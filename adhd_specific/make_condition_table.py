"""ICD definitions + group prevalences for the 78 analysed conditions (RESULTS.md Appendix A).

    python make_condition_table.py   # writes results/condition_definitions_adhd78.csv + .md

Definitions come from the condition codebook used to build the event table
(BBC_Dataset/conditions/icd_condition_mapped.xlsx; codes are matched by longest prefix, and
"A00-B99" style ranges expand over 3-character stems). Prevalences come from the aggregate D1
table. Aggregate only.
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CODEBOOK = "<PROJECT_ROOT>/BBC_Dataset/conditions/icd_condition_mapped.xlsx"
KEPT51 = "<PROJECT_ROOT>/BBC_Dataset/clustering/results/nmf_kept_conditions.txt"

cb = pd.read_excel(CODEBOOK)
cb.columns = [str(c).strip() for c in cb.columns]
cb = cb.rename(columns={"Conditions by Organ/System": "condition"})
cb["condition"] = cb["condition"].astype(str).str.strip()
cb = cb.groupby("condition", as_index=False).agg(
    {"ICD-10": lambda s: ", ".join(str(x) for x in s if pd.notna(x)),
     "ICD-9": lambda s: ", ".join(str(x) for x in s if pd.notna(x))})

d1 = pd.read_csv(os.path.join(HERE, "results", "d1_adhd78_adhd573.csv"))
p51 = {l.strip() for l in open(KEPT51) if l.strip()}
t = d1[["condition", "category", "p_NT", "p_otherNDD", "p_ADHD", "p_ASD"]].merge(cb, on="condition", how="left")
t["in_paper51"] = t["condition"].isin(p51)
t.loc[t["condition"] == "Underweight", ["ICD-10", "ICD-9"]] = ["(BMI-z < -1.645, age >= 2 y; no ICD)", ""]
t.loc[t["condition"] == "Obesity & overnutrition", "ICD-10"] = (
    t.loc[t["condition"] == "Obesity & overnutrition", "ICD-10"].fillna("") +
    " [replaced by BMI-z >= 1.036 for children with BMI measurements]")
t = t.sort_values(["category", "condition"])
t.to_csv(os.path.join(HERE, "results", "condition_definitions_adhd78.csv"), index=False)

pct = lambda x: f"{100 * x:.1f}"
rows = ["| Category | Condition | ICD-10 | ICD-9 | NT % | Other NDD % | ADHD % | ASD % | In paper's 51 |",
        "|---|---|---|---|---|---|---|---|---|"]
for _, r in t.iterrows():
    rows.append(f"| {r['category']} | {r['condition']} | {r['ICD-10'] or ''} | {r['ICD-9'] or ''} | "
                f"{pct(r['p_NT'])} | {pct(r['p_otherNDD'])} | {pct(r['p_ADHD'])} | {pct(r['p_ASD'])} | "
                f"{'yes' if r['in_paper51'] else ''} |")
open(os.path.join(HERE, "results", "condition_definitions_adhd78.md"), "w").write("\n".join(rows) + "\n")
print(f"{len(t)} conditions; missing definitions: {t['ICD-10'].isna().sum()}")
