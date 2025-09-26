# MedRule-KG: Knowledge-Graph Scaffold + Verifier for Safe Reasoning

> **All commands below are run from the repository root: `medrule-kg/`.**

This repo implements **MedRule-KG**, a compact typed knowledge graph plus a **lightweight verifier** to enforce rule-consistent reasoning by large language models (LLMs). It supports both a **synthetic dataset** and a **real-world FDA-derived dataset** and ships an evaluation pipeline that reproduces the paper’s tables and figures.

---

## Features

- **Small, interpretable KG** with entities `Drug`, `Enzyme`, relations `inhibits`, `metabolized_by`, and a boolean attribute `prolongs_qt`.
- **Rules (C1–C3)** and a **verifier** that corrects any violations in a single post-hoc step.
- **Two datasets**
  - **Synthetic** (controlled ablations/debugging).
  - **FDA-derived** (CYP substrates/inhibitors + QT flags from openFDA).
- **Reproducible evaluation**: EM, average violations, bootstrap CIs, confusion matrices, stratified EM, per-rule violations, learning curves.

---

## Repository Structure

medrule-kg/
├─ code/
│ ├─ build_medrulekg_from_fda.py
│ ├─ build_qt_flags_from_openfda.py
│ ├─ fda_cyp_to_csv.py
│ ├─ generate.py
│ ├─ run_inference.py
│ ├─ run_systems.py
│ ├─ eval_emit_inline.py
│ └─ eval.py
├─ data/
│ ├─ problems.jsonl
│ ├─ predictions_cot.jsonl
│ ├─ predictions_kg.jsonl
│ ├─ predictions_ours.jsonl
│ ├─ rules.yaml
│ └─ raw/
│ ├─ inhibitors.csv
│ ├─ substrates.csv
│ └─ qt_flags.csv
├─ figs/
│ ├─ per_rule_violations.png
│ ├─ stratified_em.png
│ └─ learning_curve.png
├─ tables/
│ └─ main_numbers_only.csv
├─ .gitignore
└─ README.md


---

## Installation

```bash
# from medrule-kg/
python3 -m venv venv
source venv/bin/activate

# If requirements.txt exists:
pip install -r requirements.txt

# Otherwise install common deps:
pip install pandas numpy matplotlib requests
# (Add your LLM client, e.g., openai, if you use a real backend)
```bash
export OPENAI_API_KEY="your_key"
export OPENAI_MODEL="gpt-4o-mini"   # or your preferred model
```

---

## Quick Start

# Option A — Synthetic Dataset

```bash
# 1) Generate synthetic tasks
python3 code/generate.py

# 2) Run inference (mock or real backend)
python3 code/run_inference.py --backend mock --overwrite
# or: python3 code/run_inference.py --backend openai --overwrite

# 3) Evaluate + emit LaTeX tables + save figures
python3 code/eval_emit_inline.py
```
Outputs:
Predictions: data/predictions_{cot,kg,ours}.jsonl
Figures: figs/per_rule_violations.png, figs/stratified_em.png, figs/learning_curve.png
Main numbers CSV: tables/main_numbers_only.csv
Console prints ready-to-paste LaTeX tables.

# Option B — FDA-Derived Dataset

```bash
# 1) Fetch/parse FDA CYP tables into CSVs (data/raw/)
python3 code/fda_cyp_to_csv.py

# 2) Build QT flags from openFDA labels (data/raw/qt_flags.csv)
python3 code/build_qt_flags_from_openfda.py

# 3) Construct problem instances (data/problems.jsonl)
python3 code/build_medrulekg_from_fda.py

# 4) Run inference
python3 code/run_inference.py --backend openai --overwrite
# or use --backend mock to dry-run

# 5) Evaluate + figures + tables
python3 code/eval_emit_inline.py
```
