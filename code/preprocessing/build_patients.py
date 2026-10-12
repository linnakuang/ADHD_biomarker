from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import config  # noqa: E402
from prenatal_panel import build_prenatal_panel, education_band  # noqa: E402

# ── ADHD presentation (subtype) from ICD codes ────────────────────────────
_ICD10 = {"F90.0": "inattentive", "F90.1": "hyp_comb", "F90.2": "hyp_comb",
          "F90.8": "unspecified", "F90.9": "unspecified"}
_ICD9 = {"314.00": "inattentive", "314.01": "hyp_comb",
         "314.0": "unspecified", "314.8": "unspecified", "314.9": "unspecified"}

SUBTYPES = ["Inattentive", "Hyperactive/combined", "Unspecified"]
NOT_CLASSIFIABLE = "Not classifiable"

D_INATT = "Inattentive"
D_HYP = "Hyperactive/combined"
D_NONSPECIFIC = "Unspecified: non-specific codes only"
D_TIED = "Unspecified: inattentive and hyperactive equally coded"
D_NOCODE = "Not classifiable: no presentation code recorded"
SUBTYPES_DETAIL = [D_INATT, D_HYP, D_NONSPECIFIC, D_TIED, D_NOCODE]
DETAIL_TO_MAIN = {D_INATT: "Inattentive", D_HYP: "Hyperactive/combined",
                  D_NONSPECIFIC: "Unspecified", D_TIED: "Unspecified",
                  D_NOCODE: NOT_CLASSIFIABLE}

def _norm_code(code):
    s = "".join(ch for ch in str(code).upper() if ch.isalnum() or ch == ".")
    if "." not in s and len(s) > 3:
        s = s[:3] + "." + s[3:]
    return s

def classify_subtype(adhd_ids, icd_df=None) -> pd.DataFrame:
    ids = [str(i).strip() for i in adhd_ids]
    if icd_df is None:
        import pyreadr
        icd_df = list(pyreadr.read_r(str(config.ICD_RDS)).values())[0]
    dx = icd_df.dropna(subset=["ID"]).copy()
    dx["ID"] = pd.to_numeric(dx["ID"], errors="coerce").astype("Int64").astype(str).str.strip()
    dx = dx[dx["ID"].isin(set(ids))].copy()
    dx["code"] = dx["icdcode"].map(_norm_code)
    dx["subtype"] = dx["code"].map(_ICD10).fillna(dx["code"].map(_ICD9))
    dx = dx.dropna(subset=["subtype"])

    base = dx.groupby(["ID", "subtype"]).size().unstack(fill_value=0)
    base = base.reindex(index=ids, fill_value=0)
    for s in ("inattentive", "hyp_comb", "unspecified"):
        if s not in base.columns:
            base[s] = 0
    ina, hc, unspec = base["inattentive"], base["hyp_comb"], base["unspecified"]
    n_codes = ina + hc + unspec
    detail = np.where(n_codes == 0, D_NOCODE,
             np.where(ina > hc, D_INATT,
             np.where(hc > ina, D_HYP,
             np.where((ina > 0) & (ina == hc), D_TIED, D_NONSPECIFIC))))
    return pd.DataFrame({"adhd_subtype": pd.Series(detail, index=ids).map(DETAIL_TO_MAIN),
                         "adhd_subtype_detail": detail,
                         "n_adhd_codes": n_codes.values}, index=ids)

# ── Patient table ─────────────────────────────────────────────────────────
def _norm_id(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64").astype(str).str.strip()

def load_labels() -> pd.DataFrame:
    df = pd.read_csv(config.LABEL_CSV)
    df = df.dropna(subset=["id"])
    df["ID"] = _norm_id(df["id"])

    asd = df["ASD"].fillna(0) == 1
    adhd = df["ADHD"].fillna(0) == 1
    ndd = df["NDD"].fillna(0).astype(int) >= 1

    label = pd.Series(pd.NA, index=df.index, dtype=object)
    label[adhd] = "ADHD"
    label[~(asd | adhd | ndd)] = "NT"

    out = pd.DataFrame({
        "ID": df["ID"],
        "label": label,
        "adhd_dx_age": pd.to_numeric(df[config.DIAG_AGE_COL], errors="coerce"),
    }).dropna(subset=["label"])
    return out.drop_duplicates("ID", keep="first")

def load_covariates() -> pd.DataFrame:
    raw = pd.read_csv(config.TELEFORM_CSV)
    lut = {c.lower(): c for c in raw.columns}

    def col(name):
        return raw[lut[name]] if name in lut else pd.Series(np.nan, index=raw.index)

    df = pd.DataFrame({"ID": _norm_id(col("id") if "id" in lut else raw["ID"])})

    sex_code = pd.to_numeric(col("sex"), errors="coerce")
    df["sex"] = np.where(sex_code == 1, "M", np.where(sex_code == 0, "F", ""))

    yr = pd.to_numeric(col("year_dod"), errors="coerce")
    mo = pd.to_numeric(col("month_dod"), errors="coerce")
    df["birth_year"] = yr
    df["dob_date"] = pd.to_datetime(dict(year=yr, month=mo, day=15), errors="coerce")

    race = pd.to_numeric(col("race_update"), errors="coerce")
    df["race_code"] = race
    df["is_black"] = race.isin(config.RACE_BLACK_CODES).astype(int)

    # Descriptive only (Table 1). Not a stratifier and not imputed.
    df["education"] = education_band(col("edu"))

    df = df[df["sex"].isin(["M", "F"])].dropna(subset=["ID", "dob_date"])
    return df.drop_duplicates("ID", keep="first")

def run():
    config.ensure(config.PREP_RESULTS)
    labels = load_labels()
    covars = load_covariates()
    panel = build_prenatal_panel()

    patients = (labels.merge(covars, on="ID", how="inner")
                      .merge(panel, on="ID", how="left"))
    patients = patients[~patients["ID"].isin([str(i) for i in config.EXCLUDE_IDS])]

    # ADHD subtype (descriptive; ICD-9/10 based). NT stays blank.
    adhd_ids = patients.loc[patients["label"] == "ADHD", "ID"]
    sub = classify_subtype(adhd_ids)
    for c in ("adhd_subtype", "adhd_subtype_detail", "n_adhd_codes"):
        patients[c] = patients["ID"].map(sub[c])

    n_adhd = int((patients["label"] == "ADHD").sum())
    n_nt = int((patients["label"] == "NT").sum())
    print(f"Patients: {len(patients)}  (ADHD={n_adhd}, NT={n_nt})")
    levels = SUBTYPES + [NOT_CLASSIFIABLE]
    sub_counts = (patients.loc[patients["label"] == "ADHD", "adhd_subtype"]
                  .value_counts().reindex(levels).fillna(0).astype(int))
    print("  ADHD presentations: "
          + ", ".join(f"{s}={sub_counts[s]}" for s in levels))

    patients.to_csv(config.PATIENTS_CSV, index=False)
    print(f"Saved -> {config.PATIENTS_CSV}")

    # Analytic cohort = all enrolled ADHD/NT children, as in the ASD project.
    # Nothing is imputed and nobody is excluded on event history.
    patients.to_csv(config.PATIENTS_ANALYTIC_CSV, index=False)
    print(f"Saved analytic cohort -> {config.PATIENTS_ANALYTIC_CSV} "
          f"({int(patients['education'].isna().sum())} with education missing, left as NA)")
    return patients

if __name__ == "__main__":
    run()
