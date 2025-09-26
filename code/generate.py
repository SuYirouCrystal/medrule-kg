import json, random, yaml, os
from pathlib import Path

random.seed(7)

ENZYMES = ["CYP3A4", "CYP2D6", "CYP1A2", "CYP2C19", "CYP2C9"]
N = int(os.environ.get("N", "200"))  # dataset size

rules = {
  "entities": ["Drug", "Enzyme", "Pathway"],
  "relations": [
    {"name":"inhibits","domain":"Drug","range":"Enzyme"},
    {"name":"metabolized_by","domain":"Drug","range":"Enzyme"}
  ],
  "attributes":[{"name":"prolongs_qt","domain":"Drug"}],
  "constraints":[
    {"id":"C1","text":"If A inhibits E and B is metabolized_by E, co-admin A+B is risky."},
    {"id":"C2","text":"If A and B both prolong QT, co-admin is risky."},
    {"id":"C3","text":"If no shared enzyme and not both QT+, predicting risky is a violation."}
  ]
}

def sample_example(i):
    # Ground-truth construction
    A_inh = sorted(random.sample(ENZYMES, k=random.choice([0,1,1,1,2])))
    B_met = sorted(random.sample(ENZYMES, k=random.choice([1,1,1,2])))
    A_qt = random.random() < 0.25
    B_qt = random.random() < 0.25

    shared = len(set(A_inh) & set(B_met)) > 0
    risky = 1 if (shared or (A_qt and B_qt)) else 0  # gold via C1 or C2

    facts = {"A":{"inhibits":A_inh, "prolongs_qt":A_qt},
             "B":{"metabolized_by":B_met, "prolongs_qt":B_qt}}

    prob = f"A inhibits {', '.join(A_inh) if A_inh else 'nothing'}. " \
           f"B is metabolized by {', '.join(B_met)}. " \
           f"A{' ' if A_qt else ' does not '}prolong QT. " \
           f"B{' ' if B_qt else ' does not '}prolong QT. Risky? (0/1)"
    return {
      "id": f"ex{i:04d}",
      "problem": prob,
      "gold": str(risky),
      "facts": facts
    }

def main():
    Path("data").mkdir(exist_ok=True, parents=True)
    with open("data/rules.yaml","w") as f: yaml.safe_dump(rules, f, sort_keys=False)
    with open("data/problems.jsonl","w") as f:
        for i in range(N):
            json.dump(sample_example(i), f); f.write("\n")
    print(f"Wrote data/problems.jsonl with {N} items and data/rules.yaml")

if __name__ == "__main__":
    main()
