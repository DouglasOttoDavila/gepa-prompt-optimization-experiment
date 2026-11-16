"""Start the reviewed local model, run the experiment, and stop the server on exit."""
import argparse
import os
from pathlib import Path
import secrets
import subprocess
import sys
from time import monotonic, sleep
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

ROOT = Path(__file__).resolve().parent
PORT = 19387
SERVER = ROOT / "runtime" / "llama" / "llama-server.exe"
MODEL = ROOT / "runtime" / "models" / "Qwen3.5-4B-Q4_K_M.gguf"


def wait_for_health(process):
    request = build_opener(ProxyHandler({}))
    deadline = monotonic() + 90
    while monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("The local inference process exited during startup. See the llama log in results/.")
        try:
            with request.open(f"http://127.0.0.1:{PORT}/health", timeout=2) as response:
                if response.status == 200 and process.poll() is None:
                    return
        except (URLError, TimeoutError):
            pass
        sleep(1)
    raise RuntimeError("The model did not become healthy within 90 seconds. See the llama log in results/.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "experiment"), default="experiment")
    args = parser.parse_args()
    if not SERVER.is_file() or not MODEL.is_file():
        raise SystemExit("Run setup-local.ps1 before starting inference.")
    env = {
        name: value for name, value in os.environ.items()
        if not name.upper().startswith(("LLAMA_", "GGML_", "OPENAI_"))
        and name.upper() not in {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"}
    }
    temp = ROOT / "runtime" / "tmp"
    temp.mkdir(exist_ok=True)
    env.update(TEMP=str(temp), TMP=str(temp))
    local_key = secrets.token_urlsafe(32)
    server_args = [
        str(SERVER), "--model", str(MODEL), "--alias", "qwen3.5-4b-q4-k-m",
        "--host", "127.0.0.1", "--port", str(PORT), "--n-gpu-layers", "99",
        "--ctx-size", "18432", "--parallel", "3", "--reasoning", "off", "--reasoning-budget", "0",
        "--no-ui", "--no-agent", "--no-ui-mcp-proxy", "--no-slots",
        "--no-cache-prompt", "--cors-origins", "localhost",
    ]
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    with (results / f"llama-{args.mode}.log").open("w", encoding="utf-8") as log:
        with subprocess.Popen(
            server_args, cwd=ROOT, env={**env, "LLAMA_API_KEY": local_key},
            stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW,
        ) as server:
            try:
                wait_for_health(server)
                print(f"Local model is ready on 127.0.0.1:{PORT}", flush=True)
                completed = subprocess.run(
                    [sys.executable, "-B", str(ROOT / "experiment.py"), "--mode", args.mode],
                    cwd=ROOT, env={**env, "GEPA_LOCAL_KEY": local_key}, check=False,
                )
                return completed.returncode
            finally:
                if server.poll() is None:
                    server.terminate()
                    server.wait(timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())
