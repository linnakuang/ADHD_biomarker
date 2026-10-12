from __future__ import annotations
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import config  # noqa: E402

_ICD10_STEM = re.compile(r"^[A-Z]\d{2}")     # e.g. J45, S06
_ICD9_STEM = re.compile(r"^\d{3}")           # e.g. 493, 800
_E_ENDOCRINE = re.compile(r"^E(0\d|[1-7]\d|8\d)$")  # E00-E89 (ICD-10 endocrine)
_ICD10_ERA = pd.Timestamp("2015-10-01")
_KEEP = re.compile(r"[^A-Z0-9]")

BMI_Z_CUTS = [-np.inf, -1.645, 1.036, 1.645, np.inf]
BMI_Z_LABELS = ["Underweight", "Normal", "Overweight", "Obese"]
# Only Underweight and the two high bands become events; Normal is not a condition.
BMI_CONDITION_MAP = {"Underweight": "Underweight",
                     "Overweight": "Obesity & overnutrition",
                     "Obese": "Obesity & overnutrition"}
BMI_Z_MAX = 5.0          # CDC convention: |bmiz| > 5 is a data-entry error
BMI_MIN_MONTHS = 24      # BMI-for-age is undefined below 2y (CDC uses weight-for-length)
ENDOCRINE_CAT = "Endocrine and Metabolic"

# ── Codebook parsing: ICD tokens and ranges -> {(bin, code): name} ─────────
RANGE_RE = re.compile(r"^\s*([A-Z0-9.]+)\s*[-–—]\s*([A-Z0-9.]+)\s*$", re.IGNORECASE)
CLEAN_RE = re.compile(r"[^A-Z0-9.]")

def _clean(tok) -> str:
    s = (str(tok) or "").strip().upper()
    return CLEAN_RE.sub("", s) if s else ""

def _dot3_scalar(s: str) -> str:
    if "." in s:
        return s
    return s[:3] + "." + s[3:] if len(s) > 3 else s

def _stem3(s: str) -> str:
    s2 = (s or "").replace(".", "")
    return s2[:3] if len(s2) >= 3 else ""

def _classify_token(tok: str):
    s = _clean(tok)
    if not s:
        return None
    s_nodot = s.replace(".", "")
    if _ICD10_STEM.match(s_nodot):
        bin_id = "10"
    elif _ICD9_STEM.match(s_nodot):
        bin_id = "9"
    else:
        return None
    if "." in s or len(s_nodot) > 3:
        return bin_id, "exact", _dot3_scalar(s)
    return bin_id, "stem", s_nodot[:3]

def _split(cell) -> list[str]:
    if pd.isna(cell):
        return []
    return [p.strip().strip(",") for p in str(cell).split(",") if p.strip().strip(",")]

def _enumerate_range(lo: str, hi: str) -> list[str]:
    out = []
    if lo[0].isalpha() and hi[0].isalpha():
        for c in range(ord(lo[0]), ord(hi[0]) + 1):
            letter = chr(c)
            start = int(lo[1:]) if letter == lo[0] else 0
            end = int(hi[1:]) if letter == hi[0] else 99
            out += [f"{letter}{n:02d}" for n in range(start, end + 1)]
    elif lo.isdigit() and hi.isdigit():
        out += [f"{n:03d}" for n in range(int(lo), int(hi) + 1)]
    else:
        out.append(lo)
    return out

# ── Generic parse -> lookups ────────────────────────────────────────────────
def _build_lookups(rows, name_field, col10, col9):
    exact_lookup, stem_lookup = {}, {}
    for row in rows:
        name = str(row.get(name_field, "")).strip()
        if not name:
            continue
        for col, bin_id in ((col10, "10"), (col9, "9")):
            for tok in _split(row.get(col, "")):
                m = RANGE_RE.match(tok)
                if m:
                    a, b = _classify_token(m.group(1)), _classify_token(m.group(2))
                    if not a or not b or a[0] != bin_id or b[0] != bin_id:
                        continue
                    lo, hi = sorted([_stem3(a[2]), _stem3(b[2])])
                    if lo and hi:
                        for st in _enumerate_range(lo, hi):
                            stem_lookup.setdefault((bin_id, st), name)
                    continue
                cls = _classify_token(tok)
                if not cls or cls[0] != bin_id:
                    continue
                _, kind, val = cls
                if kind == "exact":
                    exact_lookup.setdefault((bin_id, val), name)
                else:
                    stem_lookup.setdefault((bin_id, val), name)
    return exact_lookup, stem_lookup

def assign_labels(code: pd.Series, bin_: pd.Series, exact_lookup, stem_lookup) -> pd.Series:
    lookup = {**stem_lookup, **exact_lookup}   # exact wins where both define a key
    keys = list(zip(bin_.astype(str), code.astype(str)))
    resolved = {}
    for key in set(keys):
        bin_id, c = key
        hit = ""
        for end in range(len(c), 2, -1):
            prefix = c[:end].rstrip(".")
            if (bin_id, prefix) in lookup:
                hit = lookup[(bin_id, prefix)]
                break
        resolved[key] = hit
    return pd.Series([resolved[k] for k in keys], index=code.index)

# ── Category map (sTable1: single sheet Phenotype | ICD_10_Codes | ICD_9_Codes)
def load_category_lookup():
    df = pd.read_excel(config.CODEBOOK_XLSX)
    df = df.rename(columns={c: str(c).strip() for c in df.columns})
    return _build_lookups(df.to_dict("records"), "Phenotype",
                          "ICD_10_Codes", "ICD_9_Codes")

# ── Condition map (sTable2: condition-name | ICD-10 | ICD-9) ─────────────────
def load_condition_lookup():
    path = config.CONDITION_MAP_CSV
    if path is None or not Path(path).exists():
        print(f"  note: no condition map at {path}; conditions -> category level only")
        return {}, {}
    if str(path).lower().endswith((".xls", ".xlsx")):
        df = pd.read_excel(path)
    elif str(path).lower().endswith((".tsv", ".txt")):
        df = pd.read_csv(path, sep="\t", encoding="latin-1")
    else:
        df = pd.read_csv(path, encoding="latin-1")
    cols = {str(c).strip().lower(): c for c in df.columns}
    name_field = df.columns[0]
    col10 = cols.get("icd-10") or cols.get("icd_10") or cols.get("icd10")
    col9 = cols.get("icd-9") or cols.get("icd_9") or cols.get("icd9")
    df = df.rename(columns={c: str(c).strip() for c in df.columns})
    return _build_lookups(df.to_dict("records"), str(name_field).strip(),
                          str(col10).strip() if col10 else "ICD-10",
                          str(col9).strip() if col9 else "ICD-9")

# ── Event-stream code normalization ───────────────────────────────────────
def _dot3(code_nodot: pd.Series) -> pd.Series:
    long = code_nodot.str.len() > 3
    return code_nodot.where(~long, code_nodot.str[:3] + "." + code_nodot.str[3:])

def normalize_codes(codes: pd.Series, admit: pd.Series) -> pd.DataFrame:
    s = codes.fillna("").astype(str).str.upper().str.replace(_KEEP, "", regex=True)
    admit = pd.to_datetime(admit, errors="coerce")

    is_v = s.str.startswith("V")
    is_e = s.str.startswith("E")
    stem = s.str[:3]
    e_keep = is_e & stem.str.match(_E_ENDOCRINE, na=False) & (admit >= _ICD10_ERA)

    is_icd10 = (~is_v) & s.str.match(_ICD10_STEM, na=False) & (~is_e | e_keep)
    is_icd9 = (~is_v) & (~is_e) & s.str.match(_ICD9_STEM, na=False) & (~is_icd10)

    norm = _dot3(s)
    out = pd.DataFrame({"bin": "", "code": ""}, index=s.index)
    out.loc[is_icd10, ["bin", "code"]] = pd.DataFrame(
        {"bin": "10", "code": norm[is_icd10]})
    out.loc[is_icd9, ["bin", "code"]] = pd.DataFrame(
        {"bin": "9", "code": norm[is_icd9]})
    return out

def measured_bmi_events(known_ids) -> pd.DataFrame:
    cols = ["ID", "month", "Disease_Category", "condition", "codes", "age_years"]
    if not Path(config.CHILD_BMI_CSV).exists():
        print(f"  warning: child BMI file not found ({config.CHILD_BMI_CSV}); "
              "keeping ICD-based obesity only")
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(config.CHILD_BMI_CSV)
    df.columns = [c.lower() for c in df.columns]
    df["ID"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64").astype(str).str.strip()
    df["bmiz"] = pd.to_numeric(df.get("bmiz"), errors="coerce")
    df["month"] = pd.to_numeric(df.get("age_months"), errors="coerce")
    n_raw = len(df)
    df = df.dropna(subset=["bmiz", "month"])
    df = df[df["bmiz"].abs() <= BMI_Z_MAX]
    df = df[df["month"] >= BMI_MIN_MONTHS]
    df = df[df["ID"].isin(known_ids)]
    print(f"  BMI: {n_raw} measurements -> {len(df)} usable "
          f"(need bmiz, |z|<={BMI_Z_MAX}, age>={BMI_MIN_MONTHS}mo)")

    band = pd.cut(df["bmiz"], bins=BMI_Z_CUTS, labels=BMI_Z_LABELS,
                  right=False).astype(object)
    df = df.assign(band=band).dropna(subset=["band"]).sort_values(["ID", "month"])
    # Keep only category-transition rows, so a child who crosses into overweight
    # later contributes an event at that age rather than only at first measurement.
    prev = df.groupby("ID")["band"].shift()
    df = df[prev.isna() | (df["band"] != prev)]
    df["condition"] = df["band"].map(BMI_CONDITION_MAP)
    df = df.dropna(subset=["condition"])
    df["Disease_Category"] = ENDOCRINE_CAT
    df["codes"] = ""
    df["age_years"] = df["month"] / 12.0
    return df[["ID", "month", "Disease_Category", "condition", "codes", "age_years"]]

def run():
    config.ensure(config.PREP_RESULTS)
    patients = pd.read_csv(config.PATIENTS_ANALYTIC_CSV)
    patients["ID"] = patients["ID"].astype(str).str.strip()
    patients["dob_date"] = pd.to_datetime(patients["dob_date"], errors="coerce")

    # Imported here, not at module scope, so the codebook parsers above stay
    # importable (and testable) without the R reader installed.
    import pyreadr
    dx = list(pyreadr.read_r(str(config.ICD_RDS)).values())[0]
    dx = dx.dropna(subset=["ID"]).copy()
    dx["ID"] = pd.to_numeric(dx["ID"], errors="coerce").astype("Int64").astype(str).str.strip()
    dx["admit_dt"] = pd.to_datetime(dx["serviceDate"], errors="coerce")
    dx = dx.dropna(subset=["admit_dt", "icdcode"])

    norm = normalize_codes(dx["icdcode"], dx["admit_dt"])
    dx = pd.concat([dx.reset_index(drop=True), norm.reset_index(drop=True)], axis=1)
    dx = dx[dx["bin"].isin(["9", "10"])]

    ev = dx.merge(patients[["ID", "sex", "dob_date", "label"]], on="ID", how="inner")
    ev["month"] = ((ev["admit_dt"].dt.year - ev["dob_date"].dt.year) * 12
                   + (ev["admit_dt"].dt.month - ev["dob_date"].dt.month)).astype("Int64")
    ev = ev.dropna(subset=["month", "code"])
    ev = ev[ev["month"] >= 0]
    print(f"ICD events: {len(ev)} rows, {ev['ID'].nunique()} children")

    cat_exact, cat_stem = load_category_lookup()
    ev["Disease_Category"] = assign_labels(ev["code"], ev["bin"], cat_exact, cat_stem)
    # ASD, ADHD and NDD codes are dropped, not kept as comorbidities. NT is defined
    # as ~ASD & ~ADHD & ~NDD, so any of them is present only in the ADHD arm by
    # construction -- as an outcome it would be a perfect marker of the exposure.
    EXPOSURE_CATS = ("ASD", "ADHD", "NDD")
    n_exposure = int(ev["Disease_Category"].isin(EXPOSURE_CATS).sum())
    ev = ev[~ev["Disease_Category"].isin(EXPOSURE_CATS)]
    print(f"Dropped {n_exposure} events carrying a code that defines group membership "
          f"(ASD / ADHD / other NDD)")
    mapped = ev[ev["Disease_Category"] != ""].copy()
    unmapped = ev[ev["Disease_Category"] == ""].copy()
    print(f"Category-mapped: {len(mapped)}  unmapped: {len(unmapped)}")

    cond_exact, cond_stem = load_condition_lookup()
    mapped["condition"] = assign_labels(mapped["code"], mapped["bin"],
                                        cond_exact, cond_stem).replace("", np.nan)
    n_cond = int(mapped["condition"].notna().sum())
    print(f"Condition-mapped: {n_cond} ({n_cond / max(len(mapped), 1) * 100:.1f}%)")

    keys = ["ID", "month", "Disease_Category", "condition"]
    table = (mapped[keys + ["code"]]
             .groupby(keys, as_index=False, dropna=False)
             .agg(codes=("code", lambda s: ",".join(sorted(set(s.astype(str)))))))
    table["age_years"] = table["month"] / 12.0

    # Measured child BMI replaces ICD-based obesity for children who have BMI data.
    bmi = measured_bmi_events(set(table["ID"]))
    have_bmi = set(bmi["ID"])
    drop = (table["condition"] == "Obesity & overnutrition") & table["ID"].isin(have_bmi)
    print(f"BMI: dropped {int(drop.sum())} ICD obesity rows; added {len(bmi)} measured rows")
    table = pd.concat([table[~drop], bmi[table.columns]], ignore_index=True)

    # Congenital conditions are present from birth -> keep first onset only.
    cong = table["Disease_Category"] == "Congenital"
    cong_first = (table[cong].sort_values("month")
                  .drop_duplicates(["ID", "Disease_Category", "condition"], keep="first"))
    table = pd.concat([table[~cong], cong_first], ignore_index=True)

    table = table.sort_values(["ID", "month", "condition"],
                              key=lambda c: pd.to_numeric(c, errors="coerce")
                              if c.name in ("ID", "month") else c)
    table.to_csv(config.EVENT_TABLE_CSV, index=False)
    unmapped[["code"]].drop_duplicates().to_csv(config.UNMAPPED_CODES_CSV, index=False)

    print(f"\nEvent table -> {config.EVENT_TABLE_CSV}  shape={table.shape}")
    print(f"Categories present: {sorted(table['Disease_Category'].dropna().unique())}")
    per_label = mapped.groupby("label")["ID"].nunique()
    for g in ("ADHD", "NT"):
        print(f"  {g}: {int(per_label.get(g, 0))} children with >=1 mapped event")
    return table

if __name__ == "__main__":
    run()
