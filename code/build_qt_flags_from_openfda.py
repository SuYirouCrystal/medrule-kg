# code/build_qt_flags_from_openfda.py
import requests, csv, time

API = "https://api.fda.gov/drug/label.json"
API_KEY = ""  # optional: get one from openFDA and set &api_key=YOUR_KEY

def has_qt_flag(generic):
    # Search exact generic name + QT terms across common safety sections
    terms = ['"QT prolongation"', '"torsades de pointes"', '"Torsade de pointes"']
    sections = ["boxed_warning","warnings","warnings_and_precautions","precautions","adverse_reactions"]
    query_terms = " OR ".join([f'({s}:{t})' for s in sections for t in terms])
    q = f'openfda.generic_name.exact:"{generic}" AND ({query_terms})'
    params = {"search": q, "limit": 1}
    if API_KEY: params["api_key"] = API_KEY
    try:
        r = requests.get(API, params=params, timeout=10)
        if r.status_code == 200 and r.json().get("results"):
            return 1
    except Exception:
        pass
    return 0

# Collect unique drugs from your FDA CYP CSVs
drugs = set()
for path in ["data/raw/inhibitors.csv","data/raw/substrates.csv"]:
    with open(path) as f:
        for row in csv.DictReader(f):
            drugs.add(row["drug"])

rows = []
for i,d in enumerate(sorted(drugs)):
    flag = has_qt_flag(d)
    rows.append({"drug": d, "prolongs_qt": flag})
    if i % 25 == 0: print(i, d, flag); time.sleep(0.2)  # gentle rate

with open("data/raw/qt_flags.csv","w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=["drug","prolongs_qt"])
    w.writeheader(); w.writerows(rows)

print("Wrote data/raw/qt_flags.csv")
