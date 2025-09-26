# code/build_medrulekg_from_fda.py
import csv, json, random, re, sys
from pathlib import Path

RAW = Path("data/raw"); OUT = Path("data"); OUT.mkdir(parents=True, exist_ok=True)
INHIBITORS = RAW/"inhibitors.csv"
SUBSTRATES = RAW/"substrates.csv"
QT_FLAGS   = RAW/"qt_flags.csv"

# Normalize enzymes to canonical set used in paper
CANON = {
    "CYP3A4":"CYP3A4","CYP3A":"CYP3A4","3A4":"CYP3A4",
    "CYP2D6":"CYP2D6","2D6":"CYP2D6",
    "CYP2C9":"CYP2C9","2C9":"CYP2C9",
    "CYP2C19":"CYP2C19","2C19":"CYP2C19",
    "CYP1A2":"CYP1A2","1A2":"CYP1A2",
    "CYP2C8":"CYP2C8","2C8":"CYP2C8",
}
ENZ_KEEP = set(k for k in CANON if k.startswith("CYP"))

def canon_enzyme(s: str) -> str:
    u = re.sub(r"\s|-|P450", "", str(s).upper())
    for k,v in CANON.items():
        if k in u:
            return v
    return str(s).strip().upper()

def norm_drug(s): 
    return re.sub(r"\s+", " ", str(s).strip().lower())

def read_pairs(path: Path):
    """Read CSV with flexible headers; return dict drug->set(enzyme)."""
    m = {}
    if not path.exists():
        return m
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        # map any header variants to 'drug','enzyme'
        hdrs = {h.lower().strip(): h for h in r.fieldnames or []}
        drug_col = hdrs.get("drug") or hdrs.get("name") or list(hdrs.values())[0]
        enz_col  = hdrs.get("enzyme") or hdrs.get("enz")  or list(hdrs.values())[1]
        for row in r:
            d = norm_drug(row.get(drug_col, ""))
            e = canon_enzyme(row.get(enz_col, ""))
            if not d or not e: 
                continue
            if e not in ENZ_KEEP:
                continue
            m.setdefault(d, set()).add(e)
    return m

def read_qt(path: Path):
    """Read qt_flags if present; otherwise return empty dict (defaults to 0)."""
    q = {}
    if not path.exists():
        return q
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        hdrs = {h.lower().strip(): h for h in r.fieldnames or []}
        drug_col = hdrs.get("drug") or hdrs.get("name") or list(hdrs.values())[0]
        qt_col   = hdrs.get("prolongs_qt") or hdrs.get("qt") or list(hdrs.values())[1]
        for row in r:
            d = norm_drug(row.get(drug_col, ""))
            flag = str(row.get(qt_col, "0")).strip().lower()
            q[d] = 1 if flag in {"1","true","yes","y"} else 0
    return q

def build_examples(inh, sub, qt):
    drugs = sorted(set(inh) | set(sub) | set(qt))
    examples = []
    idx = 0
    for A in drugs:
        for B in drugs:
            if A == B: 
                continue
            A_inh = sorted(inh.get(A, set()))
            B_sub = sorted(sub.get(B, set()))
            A_qt  = bool(qt.get(A, 0))
            B_qt  = bool(qt.get(B, 0))
            c1 = bool(set(A_inh) & set(B_sub))
            c2 = bool(A_qt and B_qt)
            y = 1 if (c1 or c2) else 0
            facts = {
                "A": {"name": A, "inhibits": A_inh, "prolongs_qt": A_qt},
                "B": {"name": B, "metabolized_by": B_sub, "prolongs_qt": B_qt},
            }
            examples.append({"id": f"fda-{idx}", "facts": facts, "gold": y})
            idx += 1
    return examples

def seed_if_empty():
    inh = {"clarithromycin":{"CYP3A4"}, "ketoconazole":{"CYP3A4"}, "fluoxetine":{"CYP2D6"}, "fluvoxamine":{"CYP1A2"}}
    sub = {"midazolam":{"CYP3A4"}, "triazolam":{"CYP3A4"}, "codeine":{"CYP2D6"}, "warfarin":{"CYP2C9"}, "omeprazole":{"CYP2C19"}}
    qt  = {"erythromycin":1, "clarithromycin":1, "midazolam":0, "warfarin":0}
    return inh, sub, qt

if __name__ == "__main__":
    inh = read_pairs(INHIBITORS)
    sub = read_pairs(SUBSTRATES)
    qt  = read_qt(QT_FLAGS)

    print(f"[info] inhibitors: {len(inh)} drugs  | substrates: {len(sub)} drugs  | qt_flags: {len(qt)} drugs")

    if len(inh)==0 and len(sub)==0 and len(qt)==0:
        print("[warn] All inputs empty or missing. Using tiny seed so you can proceed.")
        inh, sub, qt = seed_if_empty()

    examples = build_examples(inh, sub, qt)
    print(f"[info] total drug names: {len(set(inh)|set(sub)|set(qt))}  -> pairs/examples: {len(examples)}")

    # Optional: balanced cap ~200
    random.seed(7)
    pos = [e for e in examples if e["gold"]==1]
    neg = [e for e in examples if e["gold"]==0]
    if len(examples) > 200:
        k_pos = min(len(pos), 100)
        k_neg = min(len(neg), 200 - k_pos)
        subset = random.sample(pos, k_pos) + random.sample(neg, k_neg)
        random.shuffle(subset)
    else:
        subset = examples

    out = OUT/"problems.jsonl"
    with open(out, "w") as f:
        for e in subset:
            f.write(json.dumps(e) + "\n")

    print(f"[done] Wrote {len(subset)} examples to {out}")
    if len(subset)==0:
        print("[hint] Check that CSV headers are 'drug,enzyme' and that enzymes look like CYP3A4/CYP2D6/etc. If you see 'CYP3A', rerun the FDA parser or edit the CSV to 'CYP3A4'.")
