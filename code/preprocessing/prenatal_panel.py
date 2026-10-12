from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import config  # noqa: E402

# Maternal education: source codes 1-5 -> 3 categories.
#   0 = below high school (1-2) | 1 = high school (3) | 2 = college or higher (4-5)
EDU_MAP = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}
EDU_LABELS = {0: "Less than High School", 1: "High School or equivalent",
              2: "College or higher"}
# parity: True = 3 categories (0 / 1 / >=2), False = binary (0 vs >=1).
PARITY_3CAT = True

def education_band(edu_code: pd.Series) -> pd.Series:
    return pd.to_numeric(edu_code, errors="coerce").map(EDU_MAP).map(EDU_LABELS)

def _col(df: pd.DataFrame, name: str) -> pd.Series:
    lut = {c.lower(): c for c in df.columns}
    if name.lower() in lut:
        return df[lut[name.lower()]]
    return pd.Series(np.nan, index=df.index)

def _read(path: Path, want: dict) -> pd.DataFrame:
    try:
        raw = pd.read_excel(path) if path.suffix.lower() in (".xls", ".xlsx") else pd.read_csv(path)
    except Exception as e:  # noqa: BLE001
        print(f"  warning: could not read {path.name} ({e}); factors -> NaN")
        return pd.DataFrame(columns=["ID", *want.values()])
    lut = {str(c).strip().lower(): c for c in raw.columns}
    if "id" not in lut:
        print(f"  warning: no id column in {path.name}; skipping")
        return pd.DataFrame(columns=["ID", *want.values()])
    out = pd.DataFrame({"ID": raw[lut["id"]]})
    for src, dst in want.items():
        out[dst] = raw[lut[src]] if src in lut else np.nan
    out["ID"] = out["ID"].astype("Int64").astype(str).str.strip()
    return out.dropna(subset=["ID"]).drop_duplicates("ID", keep="first")

def build_prenatal_panel() -> pd.DataFrame:
    tele = pd.read_csv(config.TELEFORM_CSV)
    tele.columns = [c.strip() for c in tele.columns]
    ids = pd.to_numeric(_col(tele, "id"), errors="coerce").astype("Int64").astype(str).str.strip()

    p = pd.DataFrame({"ID": ids})

    # ── Maternal demographics / anthropometrics ──
    p["mage"] = pd.to_numeric(_col(tele, "mage"), errors="coerce")

    # Household income: codes 1-11 are brackets (1=<$5k ... 11=$60k+), 12 = "Don't
    # know". Bracket widths are unequal, so the raw scale is collapsed at $30,000.
    # Unknown absorbs code 12 AND blanks -- over a third of the cohort -- so those
    # children are not imputed into the low-income reference.
    inc = pd.to_numeric(_col(tele, "yrincom"), errors="coerce")
    p["income_30k_plus"] = inc.between(7, 11).astype(float)
    p["income_unknown"] = (~inc.between(1, 11)).astype(float)

    edu3 = pd.to_numeric(_col(tele, "edu"), errors="coerce").map(EDU_MAP)
    p["edu_HS"] = np.where(edu3 == 1, 1.0, np.where(edu3.notna(), 0.0, np.nan))
    p["edu_college"] = np.where(edu3 == 2, 1.0, np.where(edu3.notna(), 0.0, np.nan))

    # parity: 3 categories, reference = nulliparous. PARITY_3CAT -> binary instead.
    par = pd.to_numeric(_col(tele, "parity"), errors="coerce")
    hi = (par >= 2) if PARITY_3CAT else pd.Series(False, index=par.index)
    lo = (par == 1) if PARITY_3CAT else (par >= 1)
    p["parity_1"] = np.where(lo, 1.0, np.where(par.notna(), 0.0, np.nan))
    p["parity_2plus"] = np.where(hi, 1.0, np.where(par.notna(), 0.0, np.nan))

    marry = pd.to_numeric(_col(tele, "marry"), errors="coerce")
    p["marry_married"] = np.where(marry == 1, 1.0, np.where(marry.notna(), 0.0, np.nan))
    p["maternal_bmi"] = pd.to_numeric(_col(tele, "bmi_clean_update"), errors="coerce")

    # ── Gestational age & birth size ──
    p["gest_age"] = pd.to_numeric(_col(tele, "geaa_o"), errors="coerce")
    p["bwt"] = pd.to_numeric(_col(tele, "bwt"), errors="coerce")
    # sbwt (merged below) = standardized birth weight, adjusted for race, sex and
    # gestational age. It is the birth-size term; bwt and lbw are alternatives.
    preterm = pd.to_numeric(_col(tele, "preterm1"), errors="coerce")
    p["preterm"] = np.where(preterm == 1, 1.0, np.where(preterm.notna(), 0.0, np.nan))
    p["lbw"] = pd.to_numeric(_col(tele, "lbw"), errors="coerce")

    # ── Delivery / labor ──
    p["deltype"] = pd.to_numeric(_col(tele, "deltype"), errors="coerce")  # 1=vaginal 0=C/S

    # Preterm-birth subtype: clinpres is never used on its own.
    #   preterm == 1 & clinpres 1-5  -> spontaneous PTB
    #   preterm == 1 & clinpres == 6 -> medically indicated PTB
    # Reference = not preterm. A preterm birth with an unusable clinpres cannot be
    # placed in either subtype, so it is left missing rather than called term.
    clin = pd.to_numeric(_col(tele, "clinpres"), errors="coerce")
    spont = (preterm == 1) & clin.between(1, 5)
    indic = (preterm == 1) & (clin == 6)
    undef = preterm.isna() | ((preterm == 1) & ~(clin.between(1, 5) | (clin == 6)))
    p["ptb_spontaneous"] = np.where(undef, np.nan, spont.astype(float))
    p["ptb_indicated"] = np.where(undef, np.nan, indic.astype(float))

    # ── Smoking / alcohol ──
    # 3 categories -> 2 dummies, reference = never. cont implies ever (verified:
    # no rows with cont == 1 and ever == 0).
    ever = pd.to_numeric(_col(tele, "ever_smk"), errors="coerce")
    cont = pd.to_numeric(_col(tele, "cont_smk"), errors="coerce")
    # ever missing -> undefined; ever == 1 with cont missing -> cannot be placed.
    # ever == 0 with cont missing is still "never", so it stays 0/0.
    smk_undef = (ever.isna() | ((ever == 1) & cont.isna())).values
    p["smk_quit"] = np.where(smk_undef, np.nan, ((ever == 1) & (cont == 0)).values.astype(float))
    p["smk_sustained"] = np.where(smk_undef, np.nan, (cont == 1).values.astype(float))
    p["etoh"] = pd.to_numeric(_col(tele, "etoh"), errors="coerce")

    # ── Pregnancy complications ──
    p["chtn"] = pd.to_numeric(_col(tele, "chtn"), errors="coerce")   # chronic HTN

    # Hypertensive disorder of pregnancy -> 2 dummies, reference = none.
    # GHTN is excluded: too much missingness to define a reliable lowest rung, so
    # severity rests on the three later variables. PROVISIONAL -- confirm the
    # severity definition with Nasrat before this goes in the methods.
    preeclam = pd.to_numeric(_col(tele, "preeclam"), errors="coerce")
    eclamp = pd.to_numeric(_col(tele, "eclamp"), errors="coerce")
    hellp = pd.to_numeric(_col(tele, "hellp"), errors="coerce")
    severe = (eclamp == 1) | (hellp == 1)
    hdp_undef = (preeclam.isna() & eclamp.isna() & hellp.isna()).values
    p["hdp_preeclampsia"] = np.where(hdp_undef, np.nan,
                                     (preeclam.isin([2, 3]) & ~severe).values.astype(float))
    p["hdp_severe"] = np.where(hdp_undef, np.nan, severe.values.astype(float))

    # diabetesw: 1=No 2=GDM 3=DM -> binary any diabetes
    p["diabetes_any"] = pd.to_numeric(_col(tele, "diabetesw"),
                                      errors="coerce").map({1: 0, 2: 1, 3: 1})

    p = p.dropna(subset=["ID"]).drop_duplicates("ID", keep="first")

    # ── External sources (merge on ID) ──
    inflam = _read(config.INFLAM_PATH, {"inflam": "inflam"})
    sbwt = _read(config.SBWT_CSV, {"sbwt": "sbwt", "iugr": "iugr"})
    stress = _read(config.STRESS_XLSX, {"indxstre": "indxstre"})
    pm25 = _read(config.PM25_CSV, {"pm2.5_gm_pregnancy": "pm25_pregnancy"})
    diet = _read(config.DIET_XLSX, {"msds": "MSDS"})
    cong = _read(config.CONGENITAL_CSV, {"congenital": "congenital_flag"})
    for ext in (inflam, sbwt, stress, pm25, diet, cong):
        p = p.merge(ext, on="ID", how="left")

    # SGA/LGA indicators from IUGR label
    iugr = p["iugr"] if "iugr" in p.columns else pd.Series(np.nan, index=p.index)
    p["iugr_SGA"] = np.where(iugr == "SGA", 1.0, np.where(iugr.notna(), 0.0, np.nan))
    p["iugr_LGA"] = np.where(iugr == "LGA", 1.0, np.where(iugr.notna(), 0.0, np.nan))

    for c in ("mage", "sbwt", "indxstre", "pm25_pregnancy", "MSDS", "congenital_flag"):
        if c in p.columns:
            p[c] = pd.to_numeric(p[c], errors="coerce")

    # Stress during pregnancy: 0 = mild, 1 = average, 2 = severe -> 2 dummies,
    # reference = mild. There is no "none" level, so this is a severity contrast
    # among exposed, not an ever/never comparison.
    stre = p["indxstre"] if "indxstre" in p.columns else pd.Series(np.nan, index=p.index)
    p["stress_average"] = np.where(stre == 1, 1.0, np.where(stre.notna(), 0.0, np.nan))
    p["stress_severe"] = np.where(stre == 2, 1.0, np.where(stre.notna(), 0.0, np.nan))

    return p.drop(columns=[c for c in ["iugr", "indxstre"] if c in p.columns])

# Prenatal / sociodemographic factors used to characterize the NMF profiles.
# Categorical variables are dummy-coded against an explicit reference rather than
# entered as numeric scores, so no step is assumed equal to any other.
PRENATAL_FACTORS = [
    # SES / demographic
    "income_30k_plus", "income_unknown",     # ref: < $30,000
    "edu_HS", "edu_college",                 # ref: below high school
    "marry_married", "mage",
    "parity_1", "parity_2plus",              # ref: nulliparous
    "maternal_bmi",
    # gestational age & birth size
    "gest_age", "sbwt",
    # delivery / labor
    "deltype", "ptb_spontaneous", "ptb_indicated",   # ref: not preterm
    # smoking / alcohol
    "smk_quit", "smk_sustained",             # ref: never
    "etoh",
    # pregnancy complications
    "chtn", "hdp_preeclampsia", "hdp_severe", "diabetes_any",
    # inflammation (int_fever and cata are its components; neither is analyzed)
    "inflam",
    # environmental / lifestyle
    "stress_average", "stress_severe",       # ref: mild
    "MSDS", "congenital_flag",
]

# Alternative codings kept for sensitivity, not in the primary factor list.
SENSITIVITY_FACTORS = ["preterm", "lbw", "bwt", "iugr_SGA", "iugr_LGA",
                       "pm25_pregnancy"]

if __name__ == "__main__":
    panel = build_prenatal_panel()
    print(f"Prenatal panel: {panel.shape[0]} children x {len(PRENATAL_FACTORS)} factors")
    present = [c for c in PRENATAL_FACTORS if c in panel.columns]
    print(panel[present].describe().T.to_string())
