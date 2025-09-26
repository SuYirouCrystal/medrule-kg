import json, re, math, random
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------- Config ---------
PROBS_PATH = "data/problems.jsonl"
SYSTEMS = [
    ("CoT (no KG)", "data/predictions_cot.jsonl"),
    ("CoT + MedRule-KG", "data/predictions_kg.jsonl"),
    ("MedRule-KG + Verifier (Ours)", "data/predictions_ours.jsonl"),
]
OUT_FIGS = Path("figs"); OUT_TBLS = Path("tables")
N_BOOT = 2000   # bootstrap iterations for CIs
SEED = 7
# --------------------------

random.seed(SEED); np.random.seed(SEED)
OUT_FIGS.mkdir(exist_ok=True, parents=True)
OUT_TBLS.mkdir(exist_ok=True, parents=True)

# ---------- Helpers ----------
def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]

def parse_ans(s):
    m = re.search(r'\b([01])\b', str(s).strip())
    return int(m.group(1)) if m else 0

def check_rules(facts, pred):
    A_inh = set(facts.get("A", {}).get("inhibits", []))
    B_met = set(facts.get("B", {}).get("metabolized_by", []))
    A_qt = bool(facts.get("A", {}).get("prolongs_qt", False))
    B_qt = bool(facts.get("B", {}).get("prolongs_qt", False))
    viols = []
    if (A_inh & B_met) and pred != 1: viols.append("C1")
    if (A_qt and B_qt) and pred != 1: viols.append("C2")
    if not (A_inh & B_met) and not (A_qt and B_qt) and pred == 1: viols.append("C3")
    # which rules are "triggered" by the facts (independent of pred)
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

# ---------- Load data ----------
probs = load_jsonl(PROBS_PATH)
gold = {d["id"]: int(d["gold"]) for d in probs}
facts = {d["id"]: d.get("facts",{}) for d in probs}
ids = [d["id"] for d in probs]

# Precompute strata by facts (C1-only, C2-only, Both, None)
strata = {}
for i in ids:
    A_inh = set(facts[i].get("A",{}).get("inhibits",[]))
    B_met = set(facts[i].get("B",{}).get("metabolized_by",[]))
    A_qt  = bool(facts[i].get("A",{}).get("prolongs_qt",False))
    B_qt  = bool(facts[i].get("B",{}).get("prolongs_qt",False))
    c1 = bool(A_inh & B_met)
    c2 = bool(A_qt and B_qt)
    if c1 and not c2: strata[i] = "C1-only"
    elif c2 and not c1: strata[i] = "C2-only"
    elif c1 and c2: strata[i] = "Both"
    else: strata[i] = "None"

# ---------- Evaluate each system ----------
main_rows = []
per_rule_rows = []
strata_rows = []
conf_rows = []

for name, path in SYSTEMS:
    preds = {d["id"]: parse_ans(d.get("prediction","0")) for d in load_jsonl(path)}
    y_true = [gold[i] for i in ids]
    y_pred = [preds.get(i,0) for i in ids]

    # Main metrics + CIs
    exact = [int(a==b) for a,b in zip(y_true,y_pred)]
    em, em_lo, em_hi = bootstrap_ci(exact, agg=np.mean)
    # recompute per-example violation counts/rules
    viol_list = []
    viol_flags = []  # 0/1 if any violation
    per_rule_counts = Counter({"C1":0,"C2":0,"C3":0})
    for i, yp in zip(ids, y_pred):
        viols, _ = check_rules(facts[i], yp)
        viol_list.append(len(viols))
        viol_flags.append(int(len(viols)>0))
        for v in viols: per_rule_counts[v]+=1
    viol_mean, vm_lo, vm_hi = bootstrap_ci(np.array(viol_list), agg=np.mean)

    main_rows.append([name, f"{em:.3f} [{em_lo:.3f},{em_hi:.3f}]",
                             f"{viol_mean:.3f} [{vm_lo:.3f},{vm_hi:.3f}]"])

    # Per-rule violation rates
    n = len(ids)
    per_rule_rows.append([name,
                          per_rule_counts["C1"]/n,
                          per_rule_counts["C2"]/n,
                          per_rule_counts["C3"]/n])

    # Stratified EM by facts
    for s in ["C1-only","C2-only","Both","None"]:
        mask = [i for i in ids if strata[i]==s]
        if len(mask)==0: acc = float('nan')
        else:
            acc = np.mean([gold[i]==preds.get(i,0) for i in mask])
        strata_rows.append([name, s, len(mask), acc])

    # Confusion matrix
    tp, tn, fp, fn = confusion(y_true, y_pred)
    conf_rows.append([name, tp, tn, fp, fn])

# ---------- Save LaTeX tables ----------
# 1) Main results with CIs
df_main = pd.DataFrame(main_rows, columns=["System","EM (95% CI)","Avg Violations (95% CI)"])
df_main.to_latex(OUT_TBLS/"main_with_ci.tex", index=False, escape=False)

# 2) Per-rule violation rates
df_pr = pd.DataFrame(per_rule_rows, columns=["System","Viol(C1)","Viol(C2)","Viol(C3)"])
df_pr.to_latex(OUT_TBLS/"per_rule_violations.tex", index=False, float_format="%.3f")

# 3) Stratified EM
df_strata = pd.DataFrame(strata_rows, columns=["System","Stratum","N","EM"])
df_strata_pivot = df_strata.pivot(index="System", columns="Stratum", values="EM").reset_index()
df_strata_pivot.to_latex(OUT_TBLS/"stratified_em.tex", index=False, float_format="%.3f")

# 4) Confusion matrix
df_conf = pd.DataFrame(conf_rows, columns=["System","TP","TN","FP","FN"])
df_conf.to_latex(OUT_TBLS/"confusion_matrices.tex", index=False)

# ---------- Figures (matplotlib defaults; one plot per figure) ----------

# A) Learning curve: EM vs prefix size
sizes = sorted(set([25, 50, 100, 150, len(ids)]))
plt.figure()
for name, path in SYSTEMS:
    preds = {d["id"]: parse_ans(d.get("prediction","0")) for d in load_jsonl(path)}
    x, y = [], []
    for s in sizes:
        sub = ids[:s]
        acc = np.mean([gold[i]==preds.get(i,0) for i in sub])
        x.append(s); y.append(acc)
    plt.plot(x, y, marker="o", label=name)
plt.xlabel("Dataset size (prefix)"); plt.ylabel("Exact Match")
plt.title("Learning Curve (prefix evaluation)")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_FIGS/"learning_curve.png", dpi=300)

# B) Per-rule violations (grouped bars)
plt.figure()
idx = np.arange(len(SYSTEMS))
width = 0.22
vals_c1 = [r[1] for r in per_rule_rows]
vals_c2 = [r[2] for r in per_rule_rows]
vals_c3 = [r[3] for r in per_rule_rows]
plt.bar(idx - width, vals_c1, width, label="C1")
plt.bar(idx,           vals_c2, width, label="C2")
plt.bar(idx + width,   vals_c3, width, label="C3")
plt.xticks(idx, [r[0] for r in per_rule_rows], rotation=15, ha="right")
plt.ylabel("Violation rate")
plt.title("Per-rule violation rates")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_FIGS/"per_rule_violations.png", dpi=300)

# C) Stratified EM (grouped bars)
plt.figure()
strata_order = ["C1-only","C2-only","Both","None"]
systems_order = [r[0] for r in per_rule_rows]
# Build matrix: rows=strata, cols=systems
mat = np.zeros((len(strata_order), len(systems_order)))
for r in strata_rows:
    sname, stratum, _, em = r
    i = strata_order.index(stratum)
    j = systems_order.index(sname)
    mat[i, j] = em
x = np.arange(len(strata_order))
width = 0.22
for j, sname in enumerate(systems_order):
    plt.bar(x + j*width - width, mat[:, j], width, label=sname)
plt.xticks(x, strata_order)
plt.ylabel("Exact Match")
plt.title("Stratified EM by factual condition")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_FIGS/"stratified_em.png", dpi=300)

print("Saved tables to", OUT_TBLS, "and figures to", OUT_FIGS)
