# code/eval_emit_inline.py
import json, re, random, sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------- Config -----------------
PROBLEMS = Path("data/problems.jsonl")
SYSTEMS = [
    ("CoT (no KG)", Path("data/predictions_cot.jsonl")),
    ("CoT + MedRule-KG", Path("data/predictions_kg.jsonl")),
    ("MedRule-KG + Verifier (Ours)", Path("data/predictions_ours.jsonl")),
]
OUT_FIGS = Path("figs"); OUT_FIGS.mkdir(exist_ok=True, parents=True)
OUT_TABLES = Path("tables"); OUT_TABLES.mkdir(exist_ok=True, parents=True)

N_BOOT = 2000
SEED = 7
random.seed(SEED); np.random.seed(SEED)
# ------------------------------------------

def load_jsonl(path: Path):
    with open(path) as f:
        return [json.loads(l) for l in f]

def parse_ans(s: str) -> int:
    """Extract the last standalone 0/1 digit as the final answer."""
    m = re.findall(r"\b([01])\b", str(s))
    return int(m[-1]) if m else 0

def facts_cues(facts):
    A = facts.get("A", {})
    B = facts.get("B", {})
    A_inh = set(A.get("inhibits", []))
    B_met = set(B.get("metabolized_by", []))
    A_qt = bool(A.get("prolongs_qt", False))
    B_qt = bool(B.get("prolongs_qt", False))
    return A_inh, B_met, A_qt, B_qt

def check_rules(facts, pred: int):
    """Return (violations list, triggered dict) for C1–C3."""
    A_inh, B_met, A_qt, B_qt = facts_cues(facts)
    viols = []
    # C1: overlap => must be 1
    if (A_inh & B_met) and pred != 1:
        viols.append("C1")
    # C2: both QT => must be 1
    if (A_qt and B_qt) and pred != 1:
        viols.append("C2")
    # C3: if not(C1 or C2) => must be 0
    if not (A_inh & B_met) and not (A_qt and B_qt) and pred == 1:
        viols.append("C3")
    trig = {"C1": int(bool(A_inh & B_met)), "C2": int(A_qt and B_qt)}
    return viols, trig

def confusion(y_true, y_pred):
    tp = sum((yt==1 and yp==1) for yt,yp in zip(y_true,y_pred))
    tn = sum((yt==0 and yp==0) for yt,yp in zip(y_true,y_pred))
    fp = sum((yt==0 and yp==1) for yt,yp in zip(y_true,y_pred))
    fn = sum((yt==1 and yp==0) for yt,yp in zip(y_true,y_pred))
    return tp, tn, fp, fn

def bootstrap_ci(vals, n_boot=N_BOOT, agg=np.mean, seed=SEED, alpha=0.05):
    vals = np.asarray(vals)
    if len(vals) == 0:
        return (float('nan'), float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    stats = []
    n = len(vals)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        stats.append(agg(vals[idx]))
    stats = np.sort(stats)
    lo = np.quantile(stats, alpha/2)
    hi = np.quantile(stats, 1-alpha/2)
    return (agg(vals), lo, hi)

# --------- Load problems ----------
if not PROBLEMS.exists():
    print("ERROR: data/problems.jsonl not found.")
    sys.exit(1)

probs = load_jsonl(PROBLEMS)
if not probs:
    print("ERROR: data/problems.jsonl is empty.")
    sys.exit(1)

ids = [d["id"] for d in probs]
gold = {d["id"]: int(d["gold"]) for d in probs}
facts = {d["id"]: d.get("facts", {}) for d in probs}

# Precompute strata labels (C1-only, C2-only, Both, None)
strata_name = {}
for i in ids:
    A_inh, B_met, A_qt, B_qt = facts_cues(facts[i])
    c1 = bool(A_inh & B_met)
    c2 = bool(A_qt and B_qt)
    if c1 and not c2:
        s = "C1-only"
    elif c2 and not c1:
        s = "C2-only"
    elif c1 and c2:
        s = "Both"
    else:
        s = "None"
    strata_name[i] = s

# ---------- Evaluate ----------
main_rows = []          # [name, em, em_lo, em_hi, vmean, v_lo, v_hi]
per_rule_rows = []      # [name, rate_C1, rate_C2, rate_C3]
conf_rows = []          # [name, tp, tn, fp, fn]
strata_rows = []        # [name, stratum, N, EM]
coverage_rows = []      # [name, covered, total, extra_count]

ALL_PER_RULE = ["C1", "C2", "C3"]

for sys_name, path in SYSTEMS:
    if not path.exists():
        print(f"ERROR: missing predictions file: {path}")
        sys.exit(1)

    preds_raw = load_jsonl(path)
    preds = {d["id"]: parse_ans(d.get("prediction", "")) for d in preds_raw}

    # coverage
    covered = sum(1 for i in ids if i in preds)
    extra_count = sum(1 for k in preds if k not in gold)
    coverage_rows.append((sys_name, covered, len(ids), extra_count))

    y_true = [gold[i] for i in ids]
    y_pred = [preds.get(i, 0) for i in ids]

    # main metrics
    exact = [int(a == b) for a, b in zip(y_true, y_pred)]
    em, em_lo, em_hi = bootstrap_ci(exact, agg=np.mean)
    viol_list = []
    per_rule_counts = Counter({k: 0 for k in ALL_PER_RULE})
    for i, yp in zip(ids, y_pred):
        viols, _ = check_rules(facts[i], yp)
        viol_list.append(len(viols))
        for v in viols:
            per_rule_counts[v] += 1
    vmean, v_lo, v_hi = bootstrap_ci(np.array(viol_list), agg=np.mean)
    main_rows.append([sys_name, em, em_lo, em_hi, vmean, v_lo, v_hi])

    # per-rule as rates
    n = len(ids)
    per_rule_rows.append([sys_name] + [per_rule_counts[k] / n for k in ALL_PER_RULE])

    # confusion
    tp, tn, fp, fn = confusion(y_true, y_pred)
    conf_rows.append([sys_name, tp, tn, fp, fn])

    # strata EM
    for s in ["C1-only", "C2-only", "Both", "None"]:
        sub = [i for i in ids if strata_name[i] == s]
        if sub:
            acc = np.mean([gold[i] == preds.get(i, 0) for i in sub])
        else:
            acc = float("nan")
        strata_rows.append([sys_name, s, len(sub), acc])

# ---------- Save figures (appendix) ----------
# A) Per-rule grouped bars
idx = np.arange(len(SYSTEMS)); width = 0.25
vals = np.array([r[1:] for r in per_rule_rows])  # shape [sys, 3]
plt.figure()
for j, k in enumerate(ALL_PER_RULE):
    plt.bar(idx + (j - 1) * width, vals[:, j], width, label=k)
plt.xticks(idx, [r[0] for r in per_rule_rows], rotation=15, ha="right")
plt.ylabel("Violation rate"); plt.title("Per-rule violation rates")
plt.legend(); plt.tight_layout()
plt.savefig(OUT_FIGS / "per_rule_violations.png", dpi=300)

# B) Stratified EM bars
systems_order = [r[0] for r in per_rule_rows]
strata_order = ["C1-only", "C2-only", "Both", "None"]
M = np.zeros((len(strata_order), len(systems_order)))
for name, s, N, em in strata_rows:
    i = strata_order.index(s)
    j = systems_order.index(name)
    M[i, j] = em
x = np.arange(len(strata_order)); width = 0.25
plt.figure()
for j, name in enumerate(systems_order):
    plt.bar(x + (j - 1) * width, M[:, j], width, label=name)
plt.xticks(x, strata_order); plt.ylabel("Exact Match")
plt.title("Stratified EM by factual condition"); plt.legend()
plt.tight_layout(); plt.savefig(OUT_FIGS / "stratified_em.png", dpi=300)

# C) Learning curve (prefix EM)
sizes = sorted({25, 50, 100, 150, len(ids)})
plt.figure()
for sys_name, path in SYSTEMS:
    preds = {d["id"]: parse_ans(d.get("prediction", "")) for d in load_jsonl(path)}
    xs, ys = [], []
    for s in sizes:
        sub = ids[:s]
        xs.append(s)
        ys.append(np.mean([gold[i] == preds.get(i, 0) for i in sub]))
    plt.plot(xs, ys, marker="o", label=sys_name)
plt.xlabel("Dataset size (prefix)"); plt.ylabel("Exact Match")
plt.title("Learning curve (prefix evaluation)"); plt.legend()
plt.tight_layout(); plt.savefig(OUT_FIGS / "learning_curve.png", dpi=300)

# ---------- Emit LaTeX blocks to console ----------
def fmt_ci(m, lo, hi):
    return f"{m:.3f} [{lo:.3f},{hi:.3f}]"

print("\n" + "=" * 70)
print("PASTE INTO MAIN TEXT (right after your existing results table)")
print("=" * 70)
print(r"""\begin{table}[t]
  \centering
  \caption{Main results with 95\% bootstrap confidence intervals on the current dataset.}
  \label{tab:main-ci}
  \begin{tabular}{lll}
\toprule
System & EM (95\% CI) & Avg Violations (95\% CI) \\
\midrule""")
for name, em, elo, ehi, vm, vlo, vhi in main_rows:
    print(f"{name} & {fmt_ci(em, elo, ehi)} & {fmt_ci(vm, vlo, vhi)} \\\\")
print(r"""\bottomrule
\end{tabular}
\end{table}""")

# Numbers-only (matches your original table style)
print("\n" + "=" * 70)
print("PASTE INTO MAIN TEXT: NUMBERS-ONLY RESULTS (matches your existing table format)")
print("=" * 70)
print(r"""\begin{table}[t]
  \centering
  \caption{Performance on the current dataset (numbers only).}
  \label{tab:main-noci}
  \begin{tabular}{lcc}
\toprule
System & EM $\uparrow$ & Avg Violations $\downarrow$ \\
\midrule""")
for name, em, elo, ehi, vm, vlo, vhi in main_rows:
    print(f"{name} & {em:.3f} & {vm:.3f} \\\\")
print(r"""\bottomrule
\end{tabular}
\end{table}""")

# Confusion matrices
print("\n" + "=" * 70)
print("PASTE INTO APPENDIX: CONFUSION MATRICES TABLE")
print("=" * 70)
print(r"""\begin{table}[h]
  \centering
  \caption{Confusion matrices by system (TP, TN, FP, FN).}
  \label{tab:confusion}
  \begin{tabular}{lrrrr}
\toprule
System & TP & TN & FP & FN \\
\midrule""")
for name, tp, tn, fp, fn in conf_rows:
    print(f"{name} & {tp} & {tn} & {fp} & {fn} \\\\")
print(r"""\bottomrule
\end{tabular}
\end{table}""")

# ---------- Optional: also save a CSV of numbers-only results ----------
pd.DataFrame(
    [{"System": n, "EM": float(f"{em:.6f}"), "AvgViolations": float(f"{vm:.6f}")}
     for n, em, _, _, vm, _, _ in main_rows]
).to_csv(OUT_TABLES / "main_numbers_only.csv", index=False)
print("\nSaved numbers-only CSV to tables/main_numbers_only.csv")

# ---------- Coverage diagnostics ----------
print("\nCoverage diagnostics:")
for name, covered, total, extra_count in coverage_rows:
    pct = 100.0 * covered / total if total else 0.0
    print(f"  {name}: matched {covered}/{total} ids ({pct:.1f}%), extra predictions not in dataset: {extra_count}")
if any(covered < len(ids) for _, covered, _, _ in coverage_rows):
    print("  [hint] Some IDs in problems.jsonl were not found in predictions_*.jsonl. Ensure IDs match (e.g., 'fda-123').")
