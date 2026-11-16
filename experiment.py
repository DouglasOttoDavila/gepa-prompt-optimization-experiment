"""Run the synthetic acceptance-criteria experiment with a bounded native DSPy route."""
import argparse
from collections import Counter
from hashlib import sha256
from importlib import metadata, util
import json
import logging
import os
from pathlib import Path
import sys
from threading import Lock
from time import perf_counter
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
os.environ["DSPY_CACHEDIR"] = str(ROOT / "runtime" / "unused-dspy-cache")

import dspy
from dspy._vendor.lm15.transports import StdlibTransport
from dspy.utils.usage_tracker import track_usage

MODEL = "openai/qwen3.5-4b-q4-k-m"
PORT = 19387
ENDPOINT = f"http://127.0.0.1:{PORT}/v1/chat/completions"
TAXONOMY = {
    "eligibility": "Required object or request precondition",
    "boundary": "Exact limit, cutoff, threshold or inclusivity",
    "authorization": "Who is permitted to act",
    "transition": "Required state after an action",
    "idempotency": "Outcome of repeating a request with the same key",
    "audit": "What fields or events history must record",
    "display": "What users must see or what the interface must omit",
}
SEED = (
    "Review the acceptance criteria against the supplied policy. "
    "Return the category IDs for policy requirements that the criteria omit, leave untestable, "
    "or contradict. Use only the supplied policy and taxonomy. Accept equivalent wording. "
    "Return an empty list when there is no gap."
)
SPLITS = {"train": 12, "validation": 4, "test": 8}
PINNED = {"dspy": "3.5.0", "gepa": "0.1.4"}


class Review(dspy.Signature):
    taxonomy: str = dspy.InputField(desc="Definitions of allowed gap category IDs")
    requirement: str = dspy.InputField(desc="Fictional user story, policy and written acceptance criteria")
    gaps: list[str] = dspy.OutputField(desc="List of distinct category IDs for material gaps")


class GapReviewer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.review = dspy.Predict(Review.with_instructions(SEED))

    def forward(self, taxonomy, requirement):
        return self.review(taxonomy=taxonomy, requirement=requirement)


class AuditTransport:
    def __init__(self, limit=100):
        self.limit = limit
        self.stage = "preflight"
        self.calls = Counter()
        self.lock = Lock()
        self.original_stream = StdlibTransport.stream

    def stream(self, transport, request):
        self.check_request(request)
        with self.lock:
            if sum(self.calls.values()) >= self.limit:
                raise RuntimeError("Hard HTTP request budget reached")
            self.calls[self.stage] += 1
        return self.original_stream(transport, request)

    @staticmethod
    def check_request(request):
        if request.url != ENDPOINT or request.method != "POST":
            raise RuntimeError("Unexpected model endpoint or HTTP method")
        body = json.loads(request.body or b"")
        if (
            body.get("store") is not False
            or body.get("tools")
            or body.get("functions")
            or body.get("stream")
            or body.get("model") != MODEL.split("/", 1)[1]
            or not body.get("messages")
        ):
            raise RuntimeError("Model request failed experiment boundaries")
        for message in body["messages"]:
            content = message.get("content")
            text_only = isinstance(content, str) or (
                isinstance(content, list)
                and all(part.get("type") == "text" and isinstance(part.get("text"), str) for part in content)
            )
            if not text_only:
                raise RuntimeError("Model request included a non-text message")


def preflight():
    for field in ("diskcache", "litellm"):
        if field in sys.modules:
            raise RuntimeError(f"Unexpected runtime import: {field}")
    for unwanted in ("litellm", "wandb", "mlflow", "mcp", "deno", "openai"):
        if util.find_spec(unwanted):
            raise RuntimeError(f"Unexpected package in the experiment environment: {unwanted}")
    if {name: metadata.version(name) for name in PINNED} != PINNED:
        raise RuntimeError("Unreviewed DSPy or GEPA version")
    if sys.version_info[:3] != (3, 12, 16):
        raise RuntimeError("Unreviewed Python version")
    if dspy.cache.enable_disk_cache or dspy.cache.enable_memory_cache:
        raise RuntimeError("Caching is enabled")


def load_cases():
    cases = json.loads((ROOT / "fixtures" / "stories.json").read_text(encoding="utf-8-sig"))
    counts = Counter(case["split"] for case in cases)
    if counts != SPLITS or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Unexpected fixture counts or duplicate IDs")
    groups = {split: [] for split in SPLITS}
    for case in cases:
        if not set(case["gaps"]) <= TAXONOMY.keys():
            raise ValueError(f"Unrecognized label in {case['id']}")
        requirement = "\n\n".join(case[field] for field in ("title", "story", "policy", "acceptance"))
        sample = dspy.Example(
            case_id=case["id"], taxonomy=json.dumps(TAXONOMY), requirement=requirement, gaps=case["gaps"]
        ).with_inputs("taxonomy", "requirement")
        groups[case["split"]].append(sample)
    return groups


def score(gold, predicted):
    expected, actual = set(gold), set(predicted)
    tp, fp, fn = expected & actual, actual - expected, expected - actual
    return {
        "tp": sorted(tp),
        "fp": sorted(fp),
        "fn": sorted(fn),
        "f1": 2 * len(tp) / (2 * len(tp) + len(fp) + len(fn)) if expected or actual else 1.0,
    }


def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    result = score(gold.gaps, pred.gaps)
    expected = ", ".join(gold.gaps) or "none"
    missed = ", ".join(result["fn"]) or "none"
    unsupported = ", ".join(result["fp"]) or "none"
    feedback = (
        f"Expected categories: {expected}. Missed: {missed}. Unsupported: {unsupported}. "
        "Compare each explicit policy clause with the acceptance criteria, including constraints "
        "on side effects, required fields and allowed states."
    )
    return dspy.Prediction(score=result["f1"], feedback=feedback)


def assess(program, samples):
    rows = []
    for sample in samples:
        start = perf_counter()
        try:
            predicted = program(**sample.inputs()).gaps
            error = None
        except dspy.LMError:
            raise
        except Exception as issue:
            predicted = []
            error = type(issue).__name__
        result = score(sample.gaps, predicted)
        rows.append({
            "case_id": sample.case_id, "expected": sample.gaps, "predicted": predicted,
            **result, "error": error, "seconds": round(perf_counter() - start, 3),
        })
    totals = {field: sum(len(row[field]) for row in rows) for field in ("tp", "fp", "fn")}
    tp, fp, fn = (totals[field] for field in ("tp", "fp", "fn"))
    summary = {
        **totals,
        "micro_f1": round(2 * tp / (2 * tp + fp + fn), 4) if tp or fp or fn else 1.0,
        "exact_match": sum(not row["fp"] and not row["fn"] and not row["error"] for row in rows),
        "format_or_runtime_errors": sum(row["error"] is not None for row in rows),
    }
    return {"summary": summary, "cases": rows}


def bench(program, groups, lm, transport):
    transport.stage = "gepa"
    start = perf_counter()
    optimizer = dspy.GEPA(
        metric=metric, reflection_lm=lm, max_metric_calls=52,
        reflection_minibatch_size=3, num_threads=3, seed=41,
        skip_perfect_score=False, use_merge=False, log_dir=None,
        track_stats=True, use_wandb=False, use_mlflow=False,
    )
    optimized = optimizer.compile(program, trainset=groups["train"], valset=groups["validation"])
    optimization_seconds = round(perf_counter() - start, 3)
    details = optimized.detailed_results.to_dict()
    transport.stage = "baseline_held_out"
    baseline = assess(program, groups["test"])
    transport.stage = "gepa_held_out"
    result = assess(optimized, groups["test"])
    a, b = baseline["summary"], result["summary"]
    return {
        "versions": PINNED, "model": MODEL, "fixture_counts": SPLITS,
        "fixture_sha256": sha256((ROOT / "fixtures" / "stories.json").read_bytes()).hexdigest(),
        "optimization_seconds": optimization_seconds, "optimization": details,
        "held_out": {"baseline": baseline, "gepa": result},
        "advance_quality_gate": (
            b["tp"] > a["tp"] and b["fp"] <= a["fp"] and b["format_or_runtime_errors"] == 0
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "smoke", "experiment"), default="preflight")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("dspy.teleprompt.gepa").setLevel(logging.INFO)
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)
    dspy.configure(adapter=dspy.JSONAdapter(), disable_history=True)
    groups = load_cases()
    preflight()
    print(f"Preflight passed. Fixture counts {SPLITS}", flush=True)
    if args.mode == "preflight":
        return
    local_key = os.environ.get("GEPA_LOCAL_KEY")
    if not local_key:
        raise RuntimeError("Start this run with run-local.py")
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "OPENAI_API_KEY", "OPENAI_BASE_URL",
                 "OPENAI_API_BASE", "OPENAI_LOG"):
        os.environ.pop(name, None)
    transport = AuditTransport()
    lm = dspy.LM(
        MODEL, model_type="chat", engine="lm15", api_base=ENDPOINT.removesuffix("/chat/completions"),
        api_key=local_key, store=False, cache=False, temperature=0, seed=41,
        max_tokens=950, timeout=90, num_retries=0,
    )
    program = GapReviewer()
    dspy.configure(lm=lm)

    def guarded_stream(instance, request):
        return transport.stream(instance, request)

    try:
        with patch.object(StdlibTransport, "stream", guarded_stream), track_usage() as usage:
            if args.mode == "smoke":
                transport.stage = "smoke"
                row = assess(program, groups["train"][:1])
                preflight()
                print(json.dumps({"result": row, "requests": dict(transport.calls)}), flush=True)
                if row["summary"]["format_or_runtime_errors"] or not transport.calls["smoke"]:
                    raise RuntimeError("The training-fixture smoke check did not produce a parseable completion")
                return
            result = bench(program, groups, lm, transport)
            result["usage"] = usage.get_total_tokens()
        preflight()
        result["http_requests"] = dict(transport.calls)
        output_dir = ROOT / "results"
        output_dir.mkdir(exist_ok=True)
        target = output_dir / "experiment.json"
        pending = target.with_suffix(".pending.json")
        pending.write_text(json.dumps(result, indent=2), encoding="utf-8")
        pending.replace(target)
        print(json.dumps(result["held_out"]["baseline"]["summary"]), flush=True)
        print(json.dumps(result["held_out"]["gepa"]["summary"]), flush=True)
        print(f"Saved {target.relative_to(ROOT)}", flush=True)
    except dspy.LMError as error:
        raise SystemExit(f"DSPy model request failed ({type(error).__name__}).") from None
    finally:
        lm.close()


if __name__ == "__main__":
    main()
