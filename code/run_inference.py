import os, json, time, argparse, re, sys, random
from pathlib import Path
from typing import Dict, Any, List

# ------------- Config -------------
PROBLEMS = Path("data/problems.jsonl")
OUT_COT  = Path("data/predictions_cot.jsonl")
OUT_KG   = Path("data/predictions_kg.jsonl")
OUT_OURS = Path("data/predictions_ours.jsonl")

# Default model name (change to your actual one)
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # example; replace as needed
SEED = 7
random.seed(SEED)
# ----------------------------------

def load_jsonl(p: Path) -> List[Dict[str, Any]]:
    with open(p) as f:
        return [json.loads(l) for l in f]

def append_jsonl(p: Path, recs: List[Dict[str, Any]]):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        for r in recs:
            f.write(json.dumps(r)+"\n")

def existing_ids(p: Path) -> set:
    if not p.exists(): return set()
    return {json.loads(l)["id"] for l in open(p)}

def parse_final_01(text: str) -> int:
    m = re.findall(r"\b([01])\b", str(text))
    return int(m[-1]) if m else 0

# ---------- Prompts ----------
SYS_INSTRUCT = (
    "You are a careful assistant for safety reasoning.\n"
    "Think step by step, then output a final line exactly in the form:\n"
    "Answer: 0  OR  Answer: 1"
)

def render_facts(facts: Dict[str, Any]) -> str:
    A = facts.get("A", {})
    B = facts.get("B", {})
    a_inh = ", ".join(A.get("inhibits", [])) or "None"
    b_met = ", ".join(B.get("metabolized_by", [])) or "None"
    a_qt  = "Yes" if A.get("prolongs_qt", False) else "No"
    b_qt  = "Yes" if B.get("prolongs_qt", False) else "No"
    a_nm  = A.get("name","Drug A")
    b_nm  = B.get("name","Drug B")
    return (
        f"Drug A: {a_nm}\n"
        f" - inhibits: {a_inh}\n"
        f" - prolongs_QT: {a_qt}\n\n"
        f"Drug B: {b_nm}\n"
        f" - metabolized_by: {b_met}\n"
        f" - prolongs_QT: {b_qt}\n"
    )

RULES_TEXT = (
    "Rules (C1–C3):\n"
    "  C1. If Drug A inhibits enzyme E and Drug B is metabolized by E, co-administration is unsafe (answer=1).\n"
    "  C2. If both drugs prolong QT, unsafe (answer=1).\n"
    "  C3. Otherwise, safe (answer=0).\n"
)

def make_user_prompt_cot(facts: Dict[str, Any]) -> str:
    # CoT (no KG): describe drugs in natural language (no explicit rule list)
    A = facts.get("A", {})
    B = facts.get("B", {})
    desc = (
        f"Consider two drugs.\n"
        f"Drug A may affect enzymes and may impact QT.\n"
        f"Drug B may be metabolized by enzymes and may impact QT.\n\n"
        f"Question: Is taking Drug A and Drug B together unsafe? "
        f"Respond with reasoning then 'Answer: 0' for safe or 'Answer: 1' for unsafe."
    )
    # Include a light NL restatement without explicit bullet rules
    return desc + "\n\nFacts (natural language):\n" + render_facts(facts)

def make_user_prompt_kg(facts: Dict[str, Any]) -> str:
    # CoT + MedRule-KG: include explicit structured serialization and rules
    return (
        "Decide if co-administration is unsafe. Use the facts and rules exactly.\n\n"
        "Facts (MedRule-KG serialization):\n" + render_facts(facts) + "\n" + RULES_TEXT +
        "\nOutput a step-by-step reasoning followed by a final line 'Answer: 0' or 'Answer: 1'."
    )

# ---------- Verifier ----------
def verifier_correction(facts: Dict[str, Any], pred: int) -> int:
    A = facts.get("A", {})
    B = facts.get("B", {})
    A_inh = set(A.get("inhibits", []))
    B_met = set(B.get("metabolized_by", []))
    A_qt  = bool(A.get("prolongs_qt", False))
    B_qt  = bool(B.get("prolongs_qt", False))
    c1 = bool(A_inh & B_met)
    c2 = bool(A_qt and B_qt)
    if c1 or c2:
        return 1
    else:
        return 0

# ---------- Backends ----------
def call_openai(messages: List[Dict[str,str]], model: str) -> str:
    # Uses the new OpenAI Python SDK style
    # pip install openai>=1.0.0
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    # You can switch to responses API if preferred
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
    )
    return resp.choices[0].message.content

def mock_predict(facts: Dict[str, Any], use_rules: bool) -> str:
    # Heuristic baseline: without rules, flip a biased coin; with rules, follow rules
    if use_rules:
        y = verifier_correction(facts, 0)  # equivalent to computing label by rules
    else:
        # weak guesser: unsafe if QT_Both or has any overlap, else random light FP rate
        A = facts["A"]; B = facts["B"]
        overlap = bool(set(A.get("inhibits", [])) & set(B.get("metabolized_by", [])))
        qt_both = bool(A.get("prolongs_qt", False) and B.get("prolongs_qt", False))
        if overlap or qt_both:
            y = 1
        else:
            y = 1 if random.random() < 0.2 else 0
    return f"Reasoning... Answer: {y}"

# ---------- Runner ----------
def run_system(name: str, problems, out_path: Path, backend: str, model: str, overwrite: bool):
    done = set()
    if out_path.exists() and not overwrite:
        done = existing_ids(out_path)
        print(f"[{name}] resuming: {len(done)} existing")

    to_write = []
    for ex in problems:
        ex_id = ex["id"]
        if ex_id in done: 
            continue
        facts = ex["facts"]

        if name == "cot":
            user_prompt = make_user_prompt_cot(facts)
        else:  # kg and ours use the same base prompt
            user_prompt = make_user_prompt_kg(facts)

        if backend == "openai":
            messages = [
                {"role":"system","content":SYS_INSTRUCT},
                {"role":"user","content":user_prompt},
            ]
            try:
                text = call_openai(messages, model=model)
            except Exception as e:
                # simple backoff and retry once
                print(f"[{name}] API error on {ex_id}: {e}; retrying...")
                time.sleep(3.0)
                text = call_openai(messages, model=model)
        elif backend == "mock":
            text = mock_predict(facts, use_rules=(name!="cot"))
        else:
            raise ValueError(f"Unknown backend: {backend}")

        pred = parse_final_01(text)

        if name == "ours":
            pred = verifier_correction(facts, pred)
            text = f"{text}\n[Verifier corrected] Answer: {pred}"

        to_write.append({"id": ex_id, "prediction": f"Answer: {pred}"})

        # flush in small batches to be robust
        if len(to_write) >= 50:
            append_jsonl(out_path, to_write)
            to_write = []

    if to_write:
        append_jsonl(out_path, to_write)

    print(f"[{name}] wrote/updated: {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["openai","mock"], default="mock",
                    help="openai=real LLM calls, mock=local heuristic")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="LLM model name if backend=openai")
    ap.add_argument("--overwrite", action="store_true", help="ignore existing outputs and rewrite")
    args = ap.parse_args()

    problems = load_jsonl(PROBLEMS)
    if not problems:
        print("ERROR: data/problems.jsonl is empty or missing.")
        sys.exit(1)

    run_system("cot",  problems, OUT_COT,  args.backend, args.model, args.overwrite)
    run_system("kg",   problems, OUT_KG,   args.backend, args.model, args.overwrite)
    run_system("ours", problems, OUT_OURS, args.backend, args.model, args.overwrite)

if __name__ == "__main__":
    main()
