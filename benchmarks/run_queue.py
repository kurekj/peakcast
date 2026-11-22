"""Run measure.py for every model of the zoo, one worker per GPU.

The queue is resumable: a model whose output file ends with a "done" record is
skipped. Each model runs in its own process, so a crash affects one model only.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from zoo import all_specs  # noqa: E402


def finished(path: str) -> bool:
    if not os.path.exists(path):
        return False
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().strip().splitlines()
    return bool(lines) and json.loads(lines[-1]).get("status") in ("done", "build_error")


def worker(gpu: str, names: list[str], lock: threading.Lock, args, log) -> None:
    while True:
        with lock:
            if not names:
                return
            name = names.pop(0)
        out = os.path.join(args.out, name + ".jsonl")
        if finished(out):
            continue
        if os.path.exists(out):
            os.remove(out)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        cmd = [sys.executable, os.path.join(HERE, args.script), "--model", name, "--out", out]
        cmd += args.extra
        t0 = time.time()
        try:
            code = subprocess.run(cmd, env=env, timeout=args.timeout, capture_output=True, text=True)
            note = f"exit={code.returncode}"
            if code.returncode != 0:
                note += " " + code.stderr.strip().splitlines()[-1][:200] if code.stderr.strip() else ""
        except subprocess.TimeoutExpired:
            note = "timeout"
            with open(out, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"model": name, "status": "done", "note": "timeout"}) + "\n")
        with lock:
            log.write(f"{time.strftime('%H:%M:%S')} gpu{gpu} {name} {time.time() - t0:.0f}s {note} left={len(names)}\n")
            log.flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--timeout", type=int, default=1500)
    ap.add_argument("--only", default="")
    ap.add_argument("--script", default="measure.py")
    ap.add_argument("extra", nargs="*")
    args = ap.parse_args()
    args.extra = [("--" + e if not e.startswith("-") and "=" in e else e) for e in args.extra]
    os.makedirs(args.out, exist_ok=True)
    names = [s.name for s in all_specs()]
    if args.only:
        keep = set(args.only.split(","))
        names = [n for n in names if n in keep]
    lock = threading.Lock()
    with open(os.path.join(args.out, "_queue.log"), "a", encoding="utf-8") as log:
        threads = [threading.Thread(target=worker, args=(g, names, lock, args, log)) for g in args.gpus.split(",")]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        log.write("QUEUE_DONE\n")


if __name__ == "__main__":
    main()
