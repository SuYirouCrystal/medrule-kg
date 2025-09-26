# code/fda_cyp_to_csv.py
import pandas as pd

FDA_URL = "https://www.fda.gov/drugs/drug-interactions-labeling/drug-development-and-drug-interactions-table-substrates-inhibitors-and-inducers"

# Pull all HTML tables on the page
tables = pd.read_html(FDA_URL, flavor="lxml")  # needs lxml installed

# Helper to normalize enzyme names
CANON = {"CYP3A": "CYP3A4", "CYP 3A4":"CYP3A4", "CYP 2D6":"CYP2D6", "CYP 2C9":"CYP2C9",
         "CYP 2C19":"CYP2C19", "CYP 1A2":"CYP1A2", "CYP2C8":"CYP2C8"}

def canon_enzyme(s):
    s = str(s).strip().upper().replace(" ", "")
    for k,v in CANON.items():
        if k.replace(" ","").upper() in s: return v
    return s

subs, inhib = [], []

# The FDA page contains multiple tables; scan for “substrate”/“inhibitor” columns
for df in tables:
    cols = [c.lower() for c in df.columns.astype(str)]
    df.columns = cols
    if any("substrate" in c for c in cols) and any("enzyme" in c for c in cols):
        # Typical layout: enzyme | example substrates | index inhibitors | ...
        for _,row in df.iterrows():
            enz = canon_enzyme(row.get("enzyme",""))
            # Split example substrates by comma/semicolon
            subs_list = pd.Series(str(row.get("example substrates",""))).str.split(r"[;,]").explode().str.strip()
            for d in subs_list.dropna():
                if d: subs.append((d.lower(), enz))
    if any("index inhibitors" in c for c in cols) and any("enzyme" in c for c in cols):
        for _,row in df.iterrows():
            enz = canon_enzyme(row.get("enzyme",""))
            inhib_list = pd.Series(str(row.get("index inhibitors",""))).str.split(r"[;,]").explode().str.strip()
            for d in inhib_list.dropna():
                if d: inhib.append((d.lower(), enz))

# Deduplicate and save
pd.DataFrame(sorted(set(inhib)), columns=["drug","enzyme"]).to_csv("data/raw/inhibitors.csv", index=False)
pd.DataFrame(sorted(set(subs)),  columns=["drug","enzyme"]).to_csv("data/raw/substrates.csv", index=False)
print("Wrote data/raw/inhibitors.csv and data/raw/substrates.csv")