"""Merge the per-tool result files into results.jsonl (tool, then zoo order, then batch)."""
import json
import os

from common import BATCHES, MODELS

HERE = os.path.dirname(os.path.abspath(__file__))
rows = {}
for tool in ("torchinfo", "xmem", "gpumemnet", "llmem"):
    path = f"{HERE}/results_{tool}.jsonl"
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            rows[(r["tool"], r["model"], r["batch"])] = r  # the last record wins
with open(HERE + "/results.jsonl", "w", encoding="utf-8") as fh:
    for tool in ("torchinfo", "xmem", "gpumemnet", "llmem"):
        n = {}
        for m in MODELS:
            for b in BATCHES:
                r = rows.get((tool, m, b))
                if r is None:
                    r = {"tool": tool, "model": m, "batch": b, "estimate_bytes": None,
                         "estimate_bin": None, "seconds": None, "status": "failed",
                         "note": "not run (no record produced)"}
                fh.write(json.dumps(r) + "\n")
                n[r["status"]] = n.get(r["status"], 0) + 1
        print(tool, n)
