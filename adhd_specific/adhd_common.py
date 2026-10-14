"""Shared loaders + helpers for the ADHD-focused D-series (adhd_analysis).

Import this FIRST in every D-script:

    import adhd_common as ac
    d = ac.load()            # cached: events, patients, group id lists, ADHD metadata

It wires sys.path so gus_analysis kernels (gus_common, s1/s5 helpers) are importable, and
caches the slow CSV loads to adhd_analysis/_cache/ (per-child; never leaves this box).

Groups (mutually exclusive priority labels, ASD > ADHD > other-NDD > NT, from *_full):
    NT 1,851 | Other NDD 930 | ADHD 573 (ADHD without ASD; PRIMARY) | ASD 145
Derived from the label CSV ADHD flag:
    ASD+ADHD 61 | ASD-only 84 | any-ADHD 634 (= ADHD + ASD+ADHD; main-branch definition)
"""
import os
import sys
import pickle

import numpy as np
import pandas as pd

ADHD_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ADHD_DIR)
import config  # noqa: E402  (adhd_analysis/config.py)

# gus_common inserts gus_analysis at sys.path[0], so its `import config` would pick up OUR
# config module from the cache. Load it with gus's config in place, then restore ours.
_ours = sys.modules.pop("config")
sys.path.insert(0, config.GUS_DIR)
import gus_common as gc  # noqa: E402
import s1_axis_robustness as s1  # noqa: E402
import s5_hierarchical_shrinkage as s5  # noqa: E402
sys.modules["gus_config"] = sys.modules.pop("config")
sys.modules["config"] = _ours
sys.path.remove(ADHD_DIR); sys.path.insert(0, ADHD_DIR)

OUT = config.OUT_DIR
CACHE = os.path.join(ADHD_DIR, "_cache")
os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)

GROUPS4 = ["NT", "Other NDD", "ADHD", "ASD"]
KEPT51 = s1.KEPT                         # the paper's 51 ASD-selected conditions


class Tee:
    def __init__(self, path):
        self.f = open(path, "w"); self.stdout = sys.stdout
    def write(self, s): self.stdout.write(s); self.f.write(s); self.f.flush()
    def flush(self): self.stdout.flush(); self.f.flush()


def tee(name):
    sys.stdout = Tee(os.path.join(OUT, f"{name}_run.log"))


def _norm_id(s):
    return pd.to_numeric(s, errors="coerce").astype("Int64").astype(str).str.strip()


def load(refresh=False):
    """Return dict: df (events<18y merged w/ label), patients, ids{group: sorted list}, meta."""
    path = os.path.join(CACHE, "base.pkl")
    if os.path.exists(path) and not refresh:
        with open(path, "rb") as f:
            return pickle.load(f)

    patients = gc.load_patients(config.PATIENTS_PATH)
    df = gc.load_event_data(config.EVENT_PATH, config.PATIENTS_PATH)

    lab = pd.read_csv(config.LABEL_CSV)
    lab = lab.dropna(subset=["id"]).assign(ID=lambda x: _norm_id(x["id"]))
    lab = lab.drop_duplicates("ID").set_index("ID")
    meta = pd.DataFrame(index=patients["ID"])
    meta["adhd_flag"] = lab["ADHD"].reindex(meta.index).fillna(0).astype(int)
    meta["adhd_dx_age"] = pd.to_numeric(lab[config.ADHD_DX_AGE_COL], errors="coerce").reindex(meta.index)
    meta["adhd_n_visit"] = pd.to_numeric(lab["ADHD_diag_n_of_visit"], errors="coerce").reindex(meta.index)

    lbl = patients.set_index("ID")["label"]
    ids = {g: sorted(lbl.index[lbl == g]) for g in GROUPS4}
    asd_adhd = sorted(i for i in ids["ASD"] if meta.at[i, "adhd_flag"] == 1)
    ids["ASD+ADHD"] = asd_adhd
    ids["ASD-only"] = sorted(set(ids["ASD"]) - set(asd_adhd))
    ids["any-ADHD"] = sorted(ids["ADHD"] + asd_adhd)
    ids["rest_noADHD"] = sorted(ids["ASD"] + ids["Other NDD"])          # NDD minus ADHD
    ids["spectrum"] = sorted(ids["ASD"] + ids["ADHD"] + ids["Other NDD"])

    out = {"df": df, "patients": patients, "ids": ids, "meta": meta}
    with open(path, "wb") as f:
        pickle.dump(out, f)
    return out


def presence(df, ids, conditions):
    """(n x J) 0/1 matrix, rows = ids, cols = sorted(conditions)."""
    raw, conds = gc.raw_binary_matrix(df, ids, conditions)
    return raw, conds


def prevalence(df, ids, conditions):
    return presence(df, ids, conditions)[0].mean(axis=0)


def std_prevalence(d, ids, conditions, target_ids, min_n=5):
    """Directly standardized prevalence over sex x is_black strata, weights from target_ids.

    Strata with < min_n children in `ids` are pooled (as nt_expected_rate does). Used to
    remove the ADHD (69% male) vs other-NDD (52% male) sex imbalance from contrasts.
    """
    p = d["patients"].set_index("ID")
    strat = p["sex"].astype(str) + "|" + pd.to_numeric(p["is_black"], errors="coerce").fillna(0).astype(int).astype(str)
    w = strat.reindex(target_ids).value_counts(normalize=True)
    raw = presence(d["df"], ids, conditions)[0]
    s = strat.reindex(ids).values
    valid = [k for k in w.index if (s == k).sum() >= min_n]
    pooled = [k for k in w.index if k not in valid]
    rate = np.zeros(raw.shape[1])
    for k in valid:
        rate += raw[s == k].mean(0) * w[k]
    if pooled:
        m = np.isin(s, pooled)
        rate += (raw[m].mean(0) if m.any() else raw.mean(0)) * w[pooled].sum()
    return rate


def nt_rate(d, conditions):
    """Stratum-standardized NT expected rate (identical to S1/S5's)."""
    return s1.nt_expected_rate(d["df"], d["patients"], conditions)[0]


def shrink(delta, se):
    """S5/S7 empirical-Bayes shrinkage of per-condition deltas (DerSimonian-Laird tau^2).

    vs-0 intervals use the Morris posterior SD, which includes the uncertainty in the global
    mean mu (S5 omitted it; harmless there, but with tau^2 = 0 the SD collapsed to 0 and every
    condition "survived" at mu). Beyond-mu ("surv*_mu") tests the condition-specific deviation
    ds - mu, i.e. specific rather than a whole-group level shift (e.g. utilization).
    """
    from scipy import stats
    tau2, mu, Q, m = s5.dersimonian_laird(delta, se)
    B = tau2 / (tau2 + se**2)
    ds = mu + B * (delta - mu)
    var_mu = 1.0 / np.sum(1.0 / (se**2 + tau2))
    psd = np.sqrt(B * se**2 + (1 - B)**2 * var_mu)
    psd_dev = np.sqrt(B * se**2)
    z90, z95 = stats.norm.ppf(0.95), stats.norm.ppf(0.975)
    dev = ds - mu
    return {"raw": delta, "se": se, "B": B, "shrunk": ds,
            "lo90": ds - z90 * psd, "hi90": ds + z90 * psd,
            "lo95": ds - z95 * psd, "hi95": ds + z95 * psd,
            "surv90": (ds - z90 * psd > 0) | (ds + z90 * psd < 0),
            "surv95": (ds - z95 * psd > 0) | (ds + z95 * psd < 0),
            "surv90_mu": (dev - z90 * psd_dev > 0) | (dev + z90 * psd_dev < 0),
            "surv95_mu": (dev - z95 * psd_dev > 0) | (dev + z95 * psd_dev < 0),
            "tau": float(np.sqrt(tau2)), "mu": float(mu), "mu_se": float(np.sqrt(var_mu)),
            "Q": float(Q), "m": int(m)}


def shrink_contrast(pa, na, pb, nb):
    """shrink() applied to pa - pb. SE from Agresti-Caffo-adjusted proportions
    ((x+1)/(n+2)): a 0/61 cell otherwise gets SE~0 and a spurious survival. Point
    estimates stay raw."""
    qa, qb = (pa * na + 1) / (na + 2), (pb * nb + 1) / (nb + 2)
    se = np.sqrt(qa * (1 - qa) / (na + 2) + qb * (1 - qb) / (nb + 2))
    return shrink(pa - pb, se)


def condition_sets(d):
    """{'adhd': ADHD-selected set, 'paper51': the paper's ASD-selected 51}.

    ADHD set mirrors the paper's selection rule with ADHD in ASD's place: >=5 ADHD children
    and ADHD prevalence >= 3%, but over ALL disease categories (the paper restricted to 7;
    injury/MSK is a canonical ADHD association and is one of Linna's three NMF profiles).
    """
    path = os.path.join(CACHE, "adhd_conditions.txt")
    if not os.path.exists(path):
        df = d["df"]
        adhd = df[df["ID"].isin(set(d["ids"]["ADHD"]))].drop_duplicates(["ID", "condition"])
        cnt = adhd["condition"].value_counts()
        n = len(d["ids"]["ADHD"])
        keep = sorted(c for c, k in cnt.items()
                      if k >= config.NMF_MIN_CASE_COUNT and k / n >= config.NMF_MIN_PREV)
        with open(path, "w") as f:
            f.write("\n".join(keep) + "\n")
    adhd = sorted(l.strip() for l in open(path) if l.strip())
    p51 = sorted(l.strip() for l in open(KEPT51) if l.strip())
    return {"adhd": adhd, "paper51": p51}


def category_of(d):
    return (d["df"].drop_duplicates("condition").set_index("condition")["Disease_Category"])


def utilization(d, ids):
    """Per-child utilization proxies over 0-18y: distinct event-months, distinct conditions."""
    df = d["df"][d["df"]["ID"].isin(set(ids))]
    months = df.assign(m=np.floor(df["month"])).drop_duplicates(["ID", "m"]).groupby("ID").size()
    span = df.groupby("ID")["age_years"].agg(lambda s: s.max() - s.min())
    return pd.DataFrame({"event_months": months, "record_span_y": span}).reindex(ids).fillna(0)
