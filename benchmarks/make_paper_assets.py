"""Write the numbers and tables of the paper from the benchmark results.

    python benchmarks/make_paper_assets.py OUTPUT_DIRECTORY

Writes OUTPUT_DIRECTORY/tables/{numbers,zoo,estimators,decision,live}.tex. Every
number in the article is a macro defined in numbers.tex or a cell of a table.
"""

from __future__ import annotations

import collections
import csv
import glob
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
GIB = 1024**3
GPU_LABEL = {"l40s": "48~GB GPU", "laptop": "24~GB GPU"}
LIVE_DEVICE = "laptop"  # the live experiments ran on this device


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}\\%"


def load_jsonl(pattern: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as fh:
            rows += [json.loads(line) for line in fh if line.strip()]
    return rows


def count_csv(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as fh:
        return sum(1 for _ in fh) - 1


def span(values: list) -> str:
    if max(values) - min(values) < 0.05:
        return f"{values[0]:.1f}"
    return f"{min(values):.1f} to {max(values):.0f}"


def models_by_domain(grid_path: str) -> dict[str, set]:
    out: dict[str, set] = collections.defaultdict(set)
    with open(grid_path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["status"] == "ok":
                out[r["domain"]].add(r["model"])
    truth_path = os.path.join(os.path.dirname(grid_path), "truth.csv")
    with open(truth_path, newline="", encoding="utf-8") as fh:
        judged = {r["model"] for r in csv.DictReader(fh) if r["status"] == "ok" and int(r["lo"]) > 0}
    return {d: models & judged for d, models in out.items()}


def zoo_table(grid_path: str, truth_grid_path: str | None = None) -> tuple[str, dict]:
    second = models_by_domain(truth_grid_path) if truth_grid_path else None
    models: dict[str, set] = collections.defaultdict(set)
    families: dict[str, set] = collections.defaultdict(set)
    params: dict[str, list] = collections.defaultdict(list)
    axis: dict[str, set] = collections.defaultdict(set)
    grids = set()
    with open(grid_path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            d = r["domain"]
            if r["status"] != "ok":
                continue
            if r["model"] not in models[d]:
                params[d].append(int(r["params"]) / 1e6)
            models[d].add(r["model"])
            families[d].add(r["family"])
            axis[d].add(r["axis_name"])
            if r["axis"] != r["axis_default"]:
                grids.add(r["model"])
    names = {"image": "image classification", "segmentation": "semantic segmentation", "video": "video classification",
             "text-encoder": "text encoders", "text-decoder": "text decoders (causal LM)",
             "seq2seq": "sequence to sequence", "audio": "audio", "recurrent": "recurrent", "multimodal": "image and text"}
    extra = " & Second GPU" if second else ""
    lines = ["\\begin{table}[H]", "\\centering", "\\small",
             "\\begin{tabular}{lr" + ("r" if second else "") + "rl}", "\\toprule",
             f"Task & Models{extra} & Parameters (M) & Second axis \\\\", "\\midrule"]
    for d in sorted(models, key=lambda d: -len(models[d])):
        p = params[d]
        cell = f" & {len(second.get(d, ()))}" if second else ""
        lines.append(f"{names.get(d, d)} & {len(models[d])}{cell} & {span(p)} & "
                     f"{', '.join(sorted(axis[d]))} \\\\")
    total = sum(len(v) for v in models.values())
    cell = f" & {sum(len(v) for v in second.values())}" if second else ""
    caption = ("Models of the benchmark. Each is trained in up to three configurations (AdamW, SGD with momentum, "
               "AdamW with mixed precision).")
    if second:
        caption += (" Models: measured on the 48~GB GPU; second GPU: the subset also measured on the 24~GB GPU.")
    lines += ["\\midrule", f"all & {total}{cell} & & \\\\", "\\bottomrule", "\\end{tabular}",
              "\\caption{" + caption + "}", "\\label{tab:zoo}", "\\end{table}"]
    return "\n".join(lines) + "\n", {"models": total, "axis_models": len(grids)}


def estimators_table(e: dict, gpu: str) -> str:
    def cell(q: dict) -> str:
        return f"{pct(q['median'])} / {pct(q['mean'])} / {pct(q['max'], 0)}"

    rows = []
    for key, name in (("torchinfo", "torchinfo~\\cite{torchinfo}"), ("xmem", "xMem~\\cite{xmem}")):
        t = e[key]
        best = min(t["vs_allocated"], t["vs_reserved"], key=lambda q: q["median"])
        rows.append(f"{name} & {t['compared']} & {cell(best)} & {t['seconds']['median']:.1f} \\\\")
    g = e["gpumemnet"]
    rows.append(f"GPUMemNet~\\cite{{gpumemnet}} & {g['compared']} & correct 8~GB class: "
                f"{g['correct_bin']} of {g['compared']} & {g['seconds']['median']:.1f} \\\\")
    rows.append("LLMem~\\cite{llmem} & 0 & could not be run & \\\\")
    p = e["peakcast"]
    rows.append(f"\\texttt{{peakcast}} (probes on the GPU) & {p['answered']} & {cell(p['vs_allocated'])} & "
                f"{p['seconds']['median']:.1f} \\\\")
    b = p["beyond_probes"]
    rows.append(f"\\quad of these, beyond the largest probe & {b['n']} & {cell(b)} & \\\\")
    lines = ["\\begin{table}[H]", "\\centering", "\\small", "\\begin{tabular}{lrlr}", "\\toprule",
             "Tool & Cases compared & Absolute relative error: median / mean / max & Time (s) \\\\", "\\midrule", *rows,
             "\\bottomrule", "\\end{tabular}",
             f"\\caption{{Estimates of peak memory against the measured peak (28 models, batch sizes 8, 32 and 128, "
             f"AdamW, single precision; {e['cases_measured']} of the {e['cases']} cases fit the {gpu} and a case "
             "is compared where the tool answered). For torchinfo and xMem the error is taken against the allocated or "
             "the reserved peak, whichever gives the tool the smaller median error. GPUMemNet predicts a memory class. "
             "Time is the median per case, for \\texttt{peakcast} the replayed time of the probe steps; xMem and "
             "GPUMemNet ran under the PyTorch version their repositories prescribe.}", "\\label{tab:estimators}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def decision_row(name: str, d: dict) -> str:
    u = d["utilization"]
    lo, hi = d["violation_ci"]
    return (f"{name} & {d['violations']} / {d['n']} & {pct(d['violation_rate'])} & "
            f"[{100 * lo:.1f}, {100 * hi:.1f}] & {pct(u['median'], 0)} & {pct(u['p05'], 0)} \\\\")


def decision_table(s: dict, gpu: str) -> str:
    rows = [decision_row("fitted law, no margin", s["margins"]["0"])]
    for m in ("0.1", "0.2", "0.3"):
        rows.append(decision_row(f"fitted law, fixed margin {int(float(m) * 100)}\\%", s["margins"][m]))
    rows.append(decision_row("AutoBatch rule, 60\\% of the budget", s["autobatch_rule"]["0.6"]))
    rows.append(decision_row("AutoBatch rule, 90\\% of the budget", s["autobatch_rule"]["0.9"]))
    rows.append("\\midrule")
    for r in ("0.1", "0.05", "0.01"):
        rows.append(decision_row(f"\\texttt{{peakcast}}, risk {100 * float(r):g}\\%", s["lofo"][r]))
    rows.append("\\midrule")
    rows.append(decision_row("halving search from a power of two (ground truth)", s["halving"]))
    lines = ["\\begin{table}[H]", "\\centering", "\\small", "\\begin{tabular}{lrrrrr}", "\\toprule",
             "Rule & \\multicolumn{3}{c}{Ran out of memory} & \\multicolumn{2}{c}{Utilisation} \\\\",
             " & cases & rate & 95\\% interval (\\%) & median & 5th pct. \\\\", "\\midrule", *rows,
             "\\bottomrule", "\\end{tabular}",
             "\\caption{Batch-size recommendations judged against the ground truth on the " + gpu + " (all workloads and "
             "budgets; replay). \\texttt{peakcast} rows use, for every workload, a calibration from which its "
             "architecture family was removed. A case runs out of memory if the recommendation exceeds the largest "
             "batch size verified to train. Utilisation is the recommended batch size over that batch size, over "
             "the other cases (zero where a rule gives no answer). The AutoBatch rule is judged where its five "
             "probes fit. The last row is an oracle: the largest power of two that trains, which a search that "
             "halves the batch size after each failure returns.}",
             "\\label{tab:decision}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def live_table(rows: list[dict], reference: list[dict] | None = None,
               second: list[dict] | None = None) -> tuple[str, dict]:
    tools = [("peakcast@0.01", "\\texttt{peakcast}, risk 1\\%"), ("peakcast@0.05", "\\texttt{peakcast}, risk 5\\%"),
             ("peakcast@0.1", "\\texttt{peakcast}, risk 10\\%"),
             ("lightning-power", "Lightning, power~\\cite{lightning}"), ("lightning-binary", "Lightning, binary search"),
             ("toma", "toma~\\cite{toma}"), ("torch-max-mem", "torch-max-mem~\\cite{torchmaxmem}"),
             ("autobatch@0.6", "Ultralytics AutoBatch~\\cite{ultralytics}")]
    best: dict = {}
    for r in [*rows, *(reference or [])]:
        if r.get("verify") == "ok":
            key = (r["model"], r["cap_label"])
            best[key] = max(best.get(key, 0), r["batch"])
    stats, body = {}, []
    for tool, name in tools:
        sub = [r for r in rows if r.get("tool") == tool]
        if not sub:
            continue
        ok = [r for r in sub if r["verify"] == "ok"]
        oom = [r for r in sub if r["verify"] == "oom"]
        other = len(sub) - len(ok) - len(oom)
        rel = [r["batch"] / best[(r["model"], r["cap_label"])] for r in ok if (r["model"], r["cap_label"]) in best]
        sec = [r["seconds"] for r in sub]
        stats[tool] = {"n": len(sub), "ok": len(ok), "oom": len(oom), "other": other,
                       "rel": statistics.median(rel) if rel else float("nan"), "sec": statistics.median(sec),
                       "sec_max": max(sec)}
        body.append(f"{name} & {len(sub)} & {len(oom)} & {other} & {pct(stats[tool]['rel'], 0)} & "
                    f"{stats[tool]['sec']:.1f} & {stats[tool]['sec_max']:.0f} \\\\")
    if second:
        other_tex, other_stats = live_table(second)
        start = other_tex.index("\\midrule\n") + len("\\midrule\n")
        other_body = other_tex[start:other_tex.index("\\bottomrule")].strip().split("\n")
        body = ["\\multicolumn{7}{l}{\\emph{24~GB GPU, budgets of 8 and 16~GiB}} \\\\", *body, "\\midrule",
                "\\multicolumn{7}{l}{\\emph{48~GB GPU, budgets of 8, 24 and 40~GiB}} \\\\", *other_body]
        stats = {**stats, "other": other_stats}
    lines = ["\\begin{table}[H]", "\\centering", "\\small", "\\begin{tabular}{lrrrrrr}", "\\toprule",
             "Tool & Cases & Failed check & No answer & Batch size & \\multicolumn{2}{c}{Time (s)} \\\\",
             " & & & & (median) & median & max \\\\", "\\midrule", *body, "\\bottomrule", "\\end{tabular}",
             "\\caption{Live comparison on both GPUs; \\texttt{peakcast} uses the calibration measured on the 48~GB GPU, without the family of the model. A recommendation fails the check if 20 training steps with it run "
             "out of memory under the budget. Batch size is relative to the largest one that passed the check for "
             "the same model and budget; time includes probing and fitting, or the search. The search tools start from a batch size of 4096 and are used through their documented interfaces.}",
             "\\label{tab:live}", "\\end{table}"]
    return "\n".join(lines) + "\n", stats


def series(items: list[str]) -> str:
    return ", ".join(items[:-1]) + " and " + items[-1] if len(items) > 1 else items[0]


def live_summary(stats: dict, earlier: dict | None = None) -> str:
    """Sentences on the live comparison, written from the table."""
    if "peakcast@0.05" not in stats:
        return "Results pending."
    risks = sorted((float(t.split("@")[1]), t) for t in stats if t.startswith("peakcast@"))
    low = stats[risks[0][1]]
    first = ""
    if earlier:
        first = (" (" + series([str(earlier[t]["oom"]) for _, t in risks if t in earlier])
                 + " with the calibration measured on the 24~GB GPU itself)")
    levels = series([f"{100 * r:g}\\%" for r, _ in risks])
    text = (f"of {low['n']} recommendations of \\texttt{{peakcast}}, "
            f"{series([str(stats[t]['oom']) for _, t in risks])} failed the check at {levels} risk{first}, "
            f"at median batch sizes of {series([pct(stats[t]['rel'], 0) for _, t in risks])} of the largest "
            f"verified one, after a median of {low['sec']:.1f}~s (at most {low['sec_max']:.1f}~s). ")
    binary, power, toma, tmm = (stats.get(t) for t in ("lightning-binary", "lightning-power", "toma", "torch-max-mem"))
    if binary and power and toma and tmm:
        text += (f"Binary search in Lightning comes closest to the limit ({pct(binary['rel'], 0)}) but takes "
                 f"{binary['sec']:.1f}~s; the "
                 f"power-of-two searches stop at {pct(power['rel'], 0)} (Lightning) and {pct(toma['rel'], 0)} (toma), "
                 f"and torch-max-mem at {pct(tmm['rel'], 0)}.")
    return text


def live_other_summary(stats: dict) -> str:
    risks = sorted((float(t.split("@")[1]), t) for t in stats if t.startswith("peakcast@"))
    low, binary, toma = stats[risks[0][1]], stats["lightning-binary"], stats["toma"]
    return (f"On the 48~GB GPU, {series([str(stats[t]['oom']) for _, t in risks])} of {low['n']} recommendations "
            f"failed, at {series([pct(stats[t]['rel'], 0) for _, t in risks])} of the largest verified batch "
            f"size and a median of {low['sec']:.1f}~s, against {binary['sec']:.1f}~s for the binary search and "
            f"{toma['sec']:.1f}~s for toma.")


def buckets_summary(rows: list[dict], short: bool = False) -> tuple[str, dict]:
    by: dict = collections.defaultdict(dict)
    for r in rows:
        if "strategy" in r:
            by[(r["model"], r["cap"])][r["strategy"]] = r
    by = {k: v for k, v in by.items() if len(v) == 3}
    if not by:
        return "", {}
    rate = lambda r: r["samples"] / r["seconds"]  # noqa: E731
    failing = sorted({m for (m, _), case in by.items() if case["buckets"]["ooms"]})
    clean = {k: v for k, v in by.items() if k[0] not in failing}
    steps = [c["fixed"]["steps_per_epoch"] / c["buckets"]["steps_per_epoch"] for c in clean.values()]
    speed_fixed = [rate(c["buckets"]) / rate(c["fixed"]) for c in clean.values()]
    speed_sorted = [rate(c["buckets"]) / rate(c["fixed-sorted"]) for c in clean.values()]
    bad = [case["buckets"] for (m, _), case in by.items() if m in failing]
    info = {"cases": len(by), "models": len({m for m, _ in by}), "clean_cases": len(clean),
            "clean_models": len({m for m, _ in clean}), "steps": statistics.median(steps),
            "speed_fixed": statistics.median(speed_fixed), "speed_sorted": statistics.median(speed_sorted),
            "failing_models": failing, "failing_ooms": sum(r["ooms"] for r in bad),
            "failing_batches": sum(r["batches_run"] for r in bad),
            "risk": next(r["risk"] for r in rows if "risk" in r)}
    text = (f"Of the {info['cases']} runs (up to 150 batches each), {info['clean_cases']}, covering "
            f"{info['clean_models']} models, completed without running out of memory. Against one batch size "
            f"chosen for the longest sequence, the plan there needs {info['steps']:.1f} times fewer steps per "
            f"epoch (median); its throughput is {info['speed_fixed']:.1f} times that of padding to the maximum "
            f"length and {info['speed_sorted']:.2f} times that of length-sorted batches with dynamic padding.")
    if failing:
        label = {"custom_lstm_lm": "the LSTM language model", "hf_t5_small": "T5-small",
                 "custom_torch_encoder_lm": "the custom transformer encoder", "hf_pythia_160m": "Pythia-160M",
                 "hf_bart_base": "BART-base", "hf_gpt2_small": "GPT-2 small"}
        parts = []
        for m in failing:
            runs = [case["buckets"] for (name, _), case in by.items() if name == m]
            parts.append(f"{label.get(m, m)} ({sum(r['ooms'] for r in runs)} of "
                         f"{sum(r['batches_run'] for r in runs)} batches failed)")
        listed = (", ".join(parts[:-1]) + " and " + parts[-1]) if len(parts) > 1 else parts[0]
        if short:
            return (f"On the 24~GB GPU ({info['models']} models, budgets of 8 and 16~GiB) {info['clean_cases']} of "
                    f"{info['cases']} runs had no failure; the exceptions are {listed}."), info
        text += f" The exception{'s are' if len(parts) > 1 else ' is'} " + listed + "."
    return text, info


def main() -> None:
    out = sys.argv[1]
    tables = os.path.join(out, "tables")
    os.makedirs(tables, exist_ok=True)
    with open(os.path.join(RESULTS, "summary.json"), encoding="utf-8") as fh:
        summary = json.load(fh)
    first = summary["main"]
    others = [d for d in summary if d not in ("main", first)]
    s = summary[first]
    zoo_tex, zoo_info = zoo_table(os.path.join(RESULTS, first, "grid.csv"))
    num: dict[str, str] = {}
    num["nZoo"] = "156"
    num["nFailedModels"] = ["No", "One", "Two", "Three", "Four", "Five"][s["failed_models"]]
    num["nModels"] = str(s["models"])
    num["nFamilies"] = str(s["families"])
    num["nWorkloads"] = str(s["workloads"])
    num["nUnits"] = str(s["units"])
    num["nAxisModels"] = str(zoo_info["axis_models"])
    grid_rows = sum(count_csv(os.path.join(RESULTS, d, "grid.csv")) for d in [first, *others])
    truth_rows = sum(count_csv(os.path.join(RESULTS, d, "truth.csv")) for d in [first, *others])
    num["nGridRows"] = f"{grid_rows:,}".replace(",", "{,}")
    num["nTruthRows"] = f"{truth_rows:,}".replace(",", "{,}")
    num["torchServer"] = s["device"]["torch"].split("+")[0]
    st = s["structure"]
    num["subhomShare"] = pct(st["sub_homogeneous_share"])
    num["convexShare"] = pct(st["convex_share"], 0)
    num["floorShare"] = pct(st["workloads_with_floor"] / st["workloads"], 0)
    num["probeCount"] = f"{s['probes']['median']:.0f}"
    num["probeSeconds"] = f"{s['probe_seconds']['median']:.1f}"
    num["settledShare"] = pct(s["settled_share"], 0)
    num["extrapMedian"] = f"{s['extrapolation']['median']:.1f}"
    num["extrapMax"] = f"{s['extrapolation']['max']:.0f}"
    acc = s["accuracy"]["abs_rel_error"]
    num["nAccuracyPoints"] = f"{acc['n']:,}".replace(",", "{,}")
    num["mapeAll"] = pct(acc["mean"])
    num["mapePninetyfive"] = pct(acc["p95"])
    num["mapeMax"] = pct(acc["max"], 0)
    num["mapeMedian"] = pct(acc["median"])
    two = s.get("two_axes", {})
    if "s" in two:
        inside, beyond = two["i"], two["s"]
        num["nAxisUnits"] = str(two["units"])
        num["nAxisDecisionModels"] = str(two["models"])
        num["mapeInside"] = pct(inside["accuracy"]["abs_rel_error"]["mean"])
        num["mapeSecond"] = pct(beyond["accuracy"]["abs_rel_error"]["mean"])
        num["mapeSecondMedian"] = pct(beyond["accuracy"]["abs_rel_error"]["median"])
        num["mapeSecondNinetyfive"] = pct(beyond["accuracy"]["abs_rel_error"]["p95"], 0)
        num["mapeSecondMax"] = pct(beyond["accuracy"]["abs_rel_error"]["max"], 0)
        num["nSecondPoints"] = f"{beyond['accuracy']['abs_rel_error']['n']:,}".replace(",", "{,}")
        five = beyond["lofo"]["0.05"]
        num["secondSummary"] = (
            f"{five['violations']} of {five['n']} cases at 5\\% risk ran out of memory "
            f"({pct(five['violation_rate'])}), at a median of {pct(five['utilization']['median'], 0)} of the "
            f"largest batch size that fits, and {five['no_answer']} cases had no batch size whose bound fits")
        import peakcast
        cal = peakcast.default_calibration()
        num["nCalibrationWorkloads"] = str(len(cal.group("b")))
        num["nCalibrationSecond"] = str(len(cal.group("s")))
        num["minRiskSecond"] = pct(1.0 / (len(cal.group("s")) + 1))
        num["maxRatio"] = f"{cal.max_ratio:.1f}"
        num["minRiskBatch"] = pct(1.0 / (len(cal.group("b")) + 1))
    second = next((d for d in others if "lofo" in summary[d]), None)
    if second:
        q = summary[second]
        zoo_tex, _ = zoo_table(os.path.join(RESULTS, first, "grid.csv"), os.path.join(RESULTS, second, "grid.csv"))
        num["secUnits"] = str(q["units"])
        num["secWorkloads"] = str(q["workloads"])
        num["secModels"] = str(q["models"])
        num["secFamilies"] = str(q["families"])
        num["secRows"] = f"{count_csv(os.path.join(RESULTS, second, 'grid.csv')):,}".replace(",", "{,}")
        num["torchSecond"] = q["device"]["torch"].split("+")[0]
        num["secMape"] = pct(q["accuracy"]["abs_rel_error"]["mean"])
        num["secViolNoMargin"] = pct(q["margins"]["0"]["violation_rate"], 0)
        for r, name in (("0.1", "Ten"), ("0.05", "Five"), ("0.01", "One")):
            num["secViol" + name] = pct(q["lofo"][r]["violation_rate"])
            num["secCount" + name] = str(q["lofo"][r]["violations"])
            for key, tag in (("transfer_from_main", "toSec"), ("transfer_to_main", "toMain")):
                t = q[key][r]
                num[tag + "Count" + name] = str(t["violations"])
                num[tag + "Rate" + name] = pct(t["violation_rate"])
                num[tag + "Cases"] = f"{t['n']:,}".replace(",", "{,}")
        num["secUtilFive"] = pct(q["lofo"]["0.05"]["utilization"]["median"], 0)
        num["toSecUtilFive"] = pct(q["transfer_from_main"]["0.05"]["utilization"]["median"], 0)
        num["toMainUtilFive"] = pct(q["transfer_to_main"]["0.05"]["utilization"]["median"], 0)
        num["toMainModelsFailing"] = str(len(q["transfer_to_main"]["0.05"]["models_out_of_memory"]))
    num["lawsBudget"] = "24" if first == "l40s" else "16"
    num["mainBudgets"] = ", ".join(str(b) for b in s["budgets_gib"][:-1]) + f" and {s['budgets_gib'][-1]}"
    num["nUnitsFmt"] = f"{s['units']:,}".replace(",", "{,}")
    grid_dev = next((d for d in others if summary[d].get("grid_only")), None)
    if grid_dev:
        g = summary[grid_dev]
        others = [d for d in others if d != grid_dev]
        ga = g["accuracy"]["abs_rel_error"]
        num["nGridModels"] = str(g["models"])
        num["nGridFamilies"] = str(g["families"])
        num["nGridWorkloads"] = str(g["workloads"])
        num["torchGrid"] = g["device"]["torch"].split("+")[0]
        num["gridPoints"] = f"{ga['n']:,}".replace(",", "{,}")
        num["gridMape"] = pct(ga["mean"])
        num["gridMapeMedian"] = pct(ga["median"])
        num["gridMapeNinetyfive"] = pct(ga["p95"])
        num["gridMapeMax"] = pct(ga["max"], 0)
        num["gridProbeSeconds"] = f"{g['probe_seconds']['median']:.1f}"
        num["gridProbeCount"] = f"{g['probes']['median']:.0f}"
        num["gridSettledShare"] = pct(g["settled_share"], 0)
        num["gridExtrapMedian"] = f"{g['extrapolation']['median']:.1f}"
        num["gridExtrapMax"] = f"{g['extrapolation']['max']:.1f}"
        gs = g["structure"]
        num["gridSubhomShare"] = pct(gs["sub_homogeneous_share"])
        num["gridConvexShare"] = pct(gs["convex_share"], 0)
        num["gridFloorShare"] = pct(gs["workloads_with_floor"] / gs["workloads"], 0)
        ax = g["accuracy_beyond_axis"]
        num["gridAxisPoints"] = f"{ax['n']:,}".replace(",", "{,}")
        num["gridAxisMape"] = pct(ax["mean"])
        num["gridAxisMapeMedian"] = pct(ax["median"])
        num["gridAxisMapeNinetyfive"] = pct(ax["p95"], 0)
        num["gridAxisMapeMax"] = pct(ax["max"], 0)
        zoo_tex, zoo_info_grid = zoo_table(os.path.join(RESULTS, grid_dev, "grid.csv"),
                                           os.path.join(RESULTS, first, "grid.csv"))
        num["nGridAxisModels"] = str(zoo_info_grid["axis_models"])
    num["nEstimatorCases"] = str(s["estimators"]["cases"])
    num["violNoMargin"] = pct(s["margins"]["0"]["violation_rate"], 0)
    num["scoreMedianPct"] = pct(s["score"]["median"] - 1)
    num["scoreNinetyfivePct"] = pct(s["score"]["p95"] - 1, 0)
    num["utilFive"] = pct(s["lofo"]["0.05"]["utilization"]["median"], 0)
    num["utilOne"] = pct(s["lofo"]["0.01"]["utilization"]["median"], 0)
    for r, name in (("0.2", "Twenty"), ("0.1", "Ten"), ("0.05", "Five"), ("0.02", "Two"), ("0.01", "One")):
        num["viol" + name] = pct(s["lofo"][r]["violation_rate"])
        num["violCount" + name] = str(s["lofo"][r]["violations"])
    num["utilHalving"] = pct(s["halving"]["utilization"]["median"], 0)
    num["violAutobatch"] = pct(s["autobatch_rule"]["0.6"]["violation_rate"])
    num["violMarginTen"] = pct(s["margins"]["0.1"]["violation_rate"])
    num["utilMarginTen"] = pct(s["margins"]["0.1"]["utilization"]["median"], 0)
    num["violMarginFifteen"] = pct(s["margins"]["0.15"]["violation_rate"])
    num["utilMarginFifteen"] = pct(s["margins"]["0.15"]["utilization"]["median"], 0)
    dom = s["by_domain"]
    hard = [d for d in dom if dom[d]["0.05"]["violation_rate"] > 0.05]
    label = {"audio": "audio models", "recurrent": "recurrent networks", "text-decoder": "text decoders",
             "text-encoder": "text encoders", "seq2seq": "sequence-to-sequence models", "image": "image classifiers"}
    hard.sort(key=lambda d: -dom[d]["0.05"]["violation_rate"])
    parts = [f"{label.get(d, d)} ({dom[d]['0.05']['violations']} of {dom[d]['0.05']['n']})" for d in hard]
    if second:
        sdom = summary[second]["by_domain"]
        shard = sorted((d for d in sdom if sdom[d]["0.05"]["violation_rate"] > 0.05),
                       key=lambda d: -sdom[d]["0.05"]["violation_rate"])
        sparts = [f"{label.get(d, d)} ({sdom[d]['0.05']['violations']} of {sdom[d]['0.05']['n']})" for d in shard]
        num["secHardDomains"] = (", ".join(sparts[:-1]) + " and " + sparts[-1]) if len(sparts) > 1 else "".join(sparts)
    num["hardRate"] = pct(sum(dom[d]["0.05"]["violations"] for d in hard)
                          / max(1, sum(dom[d]["0.05"]["n"] for d in hard)), 0)
    num["hardDomains"] = (", ".join(parts[:-1]) + " and " + parts[-1]) if len(parts) > 1 else "".join(parts) or "none"
    num["hardCount"] = str(sum(dom[d]["0.05"]["violations"] for d in hard))
    num["mainTruthRows"] = f"{count_csv(os.path.join(RESULTS, first, 'truth.csv')):,}".replace(",", "{,}")
    rest_n = sum(dom[d]["0.05"]["n"] for d in dom if d not in hard)
    rest_v = sum(dom[d]["0.05"]["violations"] for d in dom if d not in hard)
    num["restDomains"] = f"{rest_v} of {rest_n}"
    none = s["margins"]["0"]
    num["violNoMarginVerified"] = pct(none["violations_verified"] / none["n"], 0)
    num["probedAnswersFive"] = str(s["answer_is_largest_probe"]["0.05"])
    amp = s["by_config"]["adamw+amp"]["0.05"]
    num["ampFive"] = f"{amp['violations']} of {amp['n']}"
    num["ampRateFive"] = pct(amp["violation_rate"])
    need = s["required_over_allocated"]
    num["needMedianPct"] = pct(need["median"] - 1)
    num["needMaxPct"] = pct(need["max"] - 1, 0)
    num["needCases"] = str(need["n"])
    num["scoreMaxPct"] = pct(s["score"]["max"] - 1, 0)
    num["utilAutobatch"] = pct(s["autobatch_rule"]["0.6"]["utilization"]["median"], 0)
    needed = [m for m in sorted(s["margins"], key=float) if s["margins"][m]["violation_rate"] <= 0.05]
    num["marginForFive"] = (f"{int(float(needed[0]) * 100)}\\%" if needed else "more than 50\\%")
    if needed:
        num["utilMarginForFive"] = pct(s["margins"][needed[0]]["utilization"]["median"], 0)
    if grid_dev and "transfer_from_main" in summary[grid_dev]:
        g = summary[grid_dev]
        t = g["transfer_from_main"]
        num["transferModels"] = str(g["transfer_models"])
        num["transferCases"] = str(t["0.05"]["n"])
        for r, name in (("0.05", "Five"), ("0.01", "One")):
            num["transferCount" + name] = str(t[r]["violations"])
            num["transferRate" + name] = pct(t[r]["violation_rate"])
        num["transferModelsFailing"] = str(len(t["0.05"]["models_out_of_memory"]))
        num["transferUtilFive"] = pct(t["0.05"]["utilization"]["median"], 0)
    num["mainRows"] = f"{count_csv(os.path.join(RESULTS, first, 'grid.csv')):,}".replace(",", "{,}")
    if grid_dev:
        num["gridRows"] = f"{count_csv(os.path.join(RESULTS, grid_dev, 'grid.csv')):,}".replace(",", "{,}")
    num["torchMain"] = s["device"]["torch"].split("+")[0]
    live = load_jsonl(os.path.join(RESULTS, "live", "compare", "*.jsonl"))
    live_other = load_jsonl(os.path.join(RESULTS, "live_" + first, "compare", "*.jsonl"))
    live_tex, live_stats = live_table(live, second=live_other)
    if live_other:
        num["nLiveModelsMain"] = str(len({r["model"] for r in live_other if "tool" in r}))
        num["liveMainSummary"] = live_other_summary(live_stats["other"])
    num["nLiveModels"] = str(len({r["model"] for r in live if "tool" in r}))
    _, earlier = live_table(load_jsonl(os.path.join(RESULTS, "live", "compare_own_calibration", "*.jsonl")), live)
    num["liveSummary"] = live_summary(live_stats, earlier)
    buckets_text, bucket_info = buckets_summary(load_jsonl(os.path.join(RESULTS, "live", "buckets", "*.jsonl")))
    main_buckets = load_jsonl(os.path.join(RESULTS, "live_" + first, "buckets", "*.jsonl"))
    if main_buckets:  # the main device carries the description, the other one a short sentence
        num["bucketSecond"], bucket_second = buckets_summary(
            load_jsonl(os.path.join(RESULTS, "live", "buckets", "*.jsonl")), short=True)
        buckets_text, bucket_info = buckets_summary(main_buckets)
        bucket_info["second"] = bucket_second
        ten, ten_main = live_stats["peakcast@0.1"], live_stats["other"]["peakcast@0.1"]
        num["bucketVersus"] = (
            f"A batch failed in {bucket_info['cases'] - bucket_info['clean_cases']} of {bucket_info['cases']} and "
            f"{bucket_second['cases'] - bucket_second['clean_cases']} of {bucket_second['cases']} plans, more "
            f"often than single recommendations at the same risk ({ten_main['oom']} of {ten_main['n']} and "
            f"{ten['oom']} of {ten['n']}).")
    num["bucketSummary"] = buckets_text or "results pending."
    num["nBucketModels"] = str(bucket_info.get("models", 0))
    num["corpusMedian"] = "99"
    num["bucketRisk"] = f"{100 * bucket_info.get('risk', 0.05):g}\\%"
    protocol = load_jsonl(os.path.join(RESULTS, "live", "protocol", "*.jsonl"))
    num["nProtocolRuns"] = str(len(protocol))
    recorded = {}
    live_device = LIVE_DEVICE if os.path.isdir(os.path.join(RESULTS, LIVE_DEVICE)) else first
    with open(os.path.join(RESULTS, live_device, "grid.csv"), newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if (r["optimizer"], r["amp"], r["status"]) == ("adamw", "0", "ok") and r["axis"] == r["axis_default"]:
                recorded[(r["model"], int(r["batch"]))] = int(r["peak_alloc"])
    diffs = [abs(r["alloc_n"] / recorded[(r["model"], r["batch"])] - 1.0) for r in protocol
             if (r["model"], r["batch"]) in recorded]
    recorded = {}
    with open(os.path.join(RESULTS, first, "grid.csv"), newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if (r["optimizer"], r["amp"], r["status"]) == ("adamw", "0", "ok") and r["axis"] == r["axis_default"]:
                recorded[(r["model"], int(r["batch"]))] = int(r["peak_alloc"])
    if live_device != first:
        diffs += [abs(r["alloc_n"] / recorded[(r["model"], r["batch"])] - 1.0)
                  for r in load_jsonl(os.path.join(RESULTS, "live_" + first, "protocol", "*.jsonl"))
                  if (r["model"], r["batch"]) in recorded]
    num["nProtocolRuns"] = str(len(diffs))
    num["protocolMaxDiff"] = pct(max(diffs), 2) if diffs else "n/a"
    tests = 0
    for path in glob.glob(os.path.join(HERE, "..", "tests", "test_*.py")):
        with open(path, encoding="utf-8") as fh:
            tests += sum(line.startswith("def test_") for line in fh)
    num["nTests"] = str(tests)
    extra_path = os.path.join(RESULTS, "paper_text.json")
    if os.path.exists(extra_path):
        with open(extra_path, encoding="utf-8") as fh:
            num.update(json.load(fh))
    with open(os.path.join(tables, "numbers.tex"), "w", encoding="utf-8") as fh:
        fh.write("%% generated by benchmarks/make_paper_assets.py\n")
        for key in sorted(num):
            fh.write(f"\\newcommand{{\\{key}}}{{{num[key]}}}\n")
    for name, text in (("zoo", zoo_tex), ("estimators", estimators_table(s["estimators"], GPU_LABEL.get(first, first))),
                       ("decision", decision_table(s, GPU_LABEL.get(first, first))), ("live", live_tex)):
        text = text.replace("\\begin{tabular}", "\\resizebox{\\linewidth}{!}{%\n\\begin{tabular}")
        text = text.replace("\\end{tabular}", "\\end{tabular}}")
        with open(os.path.join(tables, name + ".tex"), "w", encoding="utf-8") as fh:
            fh.write(text)
    with open(os.path.join(tables, "stats.json"), "w", encoding="utf-8") as fh:
        json.dump({"live": live_stats, "buckets": bucket_info, "numbers": num}, fh, indent=1)
    print("assets written to", out)


if __name__ == "__main__":
    main()
