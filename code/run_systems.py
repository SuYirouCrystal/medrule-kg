import json, os, random, re
from pathlib import Path

random.seed(11)

def noisy_reasoner(problem, facts, with_kg=False):
    """
    A stochastic 'imperfect reasoner' to simulate CoT mistakes.
    - If with_kg=True, it makes fewer errors (uses facts).
    """
    A_inh = set(facts.get("A",{}).get("inhibits",[])) if with_kg else set()
    B_met = set(facts.get("B",{}).get("metabolized_by",[])) if with_kg else set()
    A_qt = bool(facts.get("A",{}).get("prolongs_qt", False)) if with_kg else False
    B_qt = bool(facts.get("B",{}).get("prolongs_qt", False)) if with_kg else False

    # Heuristic guess
    guess = 1 if (A_inh & B_met) or (A_qt and B_qt) else 0

    # Inject mistakes: without KG ~25% flip; with KG ~12% flip
    p_flip = 0.25 if not with_kg else 0.12
    if random.random() < p_flip:
        guess = 1 - guess

    rationale = []
    if with_kg:
        rationale += ["1) Use KG facts to check shared enzymes and QT flags.",
                      f"2) Shared={bool(A_inh & B_met)}, A_QT={A_qt}, B_QT={B_qt}.",
                      "3) Apply rules C1/C2; if neither holds, predict 0."]
    else:
        rationale += ["1) Consider interaction qualitatively.",
                      "2) If any enzyme appears important or both prolong QT, predict risk.",
                      "3) Otherwise predict safe."]

    return str(guess), " ".join(rationale)

def verifier(facts, pred):
    A_inh = set(facts.get("A",{}).get("inhibits",[]))
    B_met = set(facts.get("B",{}).get("metabolized_by",[]))
    A_qt = bool(facts.get("A",{}).get("prolongs_qt", False))
    B_qt = bool(facts.get("B",{}).get("prolongs_qt", False))
    viols = []
    # C1
    if (A_inh & B_met) and pred != "1": viols.append("C1")
    # C2
    if (A_qt and B_qt) and pred != "1": viols.append("C2")
    # C3
    if not (A_inh & B_met) and not (A_qt and B_qt) and pred == "1": viols.append("C3")
    return viols

def run(split="problems"):
    data = [json.loads(l) for l in open(f"data/{split}.jsonl")]
    Path("data").mkdir(exist_ok=True, parents=True)

    # (a) CoT only (no KG)
    with open("data/predictions_cot.jsonl","w") as f:
        for ex in data:
            pred, rat = noisy_reasoner(ex["problem"], ex["facts"], with_kg=False)
            out = {"id": ex["id"], "prediction": pred, "rationale": rat, "rules_violated": []}
            json.dump(out,f); f.write("\n")

    # (b) CoT + KG
    with open("data/predictions_kg.jsonl","w") as f:
        for ex in data:
            pred, rat = noisy_reasoner(ex["problem"], ex["facts"], with_kg=True)
            out = {"id": ex["id"], "prediction": pred, "rationale": rat, "rules_violated": []}
            json.dump(out,f); f.write("\n")

    # (c) Ours: KG + Verifier (one-shot self-correct if violated)
    with open("data/predictions_ours.jsonl","w") as f:
        for ex in data:
            pred, rat = noisy_reasoner(ex["problem"], ex["facts"], with_kg=True)
            viols = verifier(ex["facts"], pred)
            if viols:
                # Self-correct once to satisfy constraints
                pred = "1" if any(v in ["C1","C2"] for v in viols) else "0"
                rat = rat + " [Corrected by verifier]"
                viols = verifier(ex["facts"], pred)
            out = {"id": ex["id"], "prediction": pred, "rationale": rat, "rules_violated": viols}
            json.dump(out,f); f.write("\n")

    print("Wrote predictions_{cot,kg,ours}.jsonl")

if __name__ == "__main__":
    run()