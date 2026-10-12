"""S3 — Does adding the NDD/ADHD groups lift the temporal-modeling density ceiling?

Track A's tensor factorizations (A1/A2) failed as data-availability artifacts: at
n=145 the third (time) mode is too sparse — a genuine >=3-window developmental
factorization is not estimable (density <~8% / thin-slices >~37%; a0_feasibility_table).
Density scales with n, so the question is whether the spectrum pool (1,648) clears the
same floors under the same age-based binning.

Mirrors a0/slice_audit's construction (within-window NT residual, stratum-standardized,
clip at 0) but pool-agnostic via s1's helpers (a0's build_residual_matrix reads the
analytic-only patients file and cannot see the excluded children). VALIDATION: the
ASD-only column must reproduce a0_feasibility_table.csv before the spectrum column is
trusted.
"""
import os
import numpy as np
import pandas as pd

import gus_common as gc
import s1_axis_robustness as s1

OUT = gc.OUT_DIR
FLOOR = gc.config.DENSITY_FLOOR_CHILDREN          # 5
WKEEP = gc.config.WINDOW_KEEP_FRAC                 # 0.60
MEDIAN_DX = 3.253                                  # ASD median dx (fixed age boundary)


def audit(df, ids, patients, conditions, edges, labels):
    """Per-window within-window NT residual over `ids`; density + thin-slice stats."""
    slices, per_window = [], []
    for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
        df_w = df[(df["age_years"] >= lo) & (df["age_years"] < hi)]
        nt_rate, _ = s1.nt_expected_rate(df_w, patients, conditions)   # within-window NT rate
        raw, _ = gc.raw_binary_matrix(df_w, ids, conditions)
        clipped = np.clip(raw - nt_rate[None, :], 0, None)
        slices.append(clipped)
        nz = clipped > 0
        slice_n = nz.sum(axis=0)
        n_pass = int((slice_n >= FLOOR).sum())
        per_window.append(dict(window=lab, density=float(nz.mean()),
                               frac_pass=n_pass / max(1, len(conditions))))
    X = np.stack(slices, axis=2)
    nz = X > 0
    slice_n = nz.sum(axis=0)                                           # J x K
    thin = int((slice_n < FLOOR).sum())
    return dict(density=float(nz.mean()) * 100,
                thin_pct=round(thin / slice_n.size * 100),
                median_slice_n=round(float(np.median(slice_n))),
                windows_pass=sum(w["frac_pass"] >= WKEEP for w in per_window))


def main():
    conditions = sorted(l.strip() for l in open(s1.KEPT) if l.strip())
    patients = gc.load_patients(s1.FULL_PAT)
    df = gc.load_event_data(s1.FULL_EVT, s1.FULL_PAT)
    ids = {g: sorted(patients.loc[patients["label"] == g, "ID"])
           for g in ("ASD", "ADHD", "Other NDD")}
    pools = {"ASD(145)": ids["ASD"],
             "spectrum(1648)": sorted(ids["ASD"] + ids["ADHD"] + ids["Other NDD"])}

    schemes = [
        ("Full 0-18, 3y",   *gc.truncated_bins(18, 3)),
        ("Truncate @12, 3y", *gc.truncated_bins(12, 3)),
        ("Truncate @10, 2y", *gc.truncated_bins(10, 2)),
        ("Truncate @9, 3y",  *gc.truncated_bins(9, 3)),
        ("Dx-anchored 3w",   *gc.dx_anchored_bins(MEDIAN_DX, 3, 3.0)),
        ("Truncate @6, 2y",  *gc.truncated_bins(6, 2)),
        ("Two-window pre/post", *gc.two_window_bins(MEDIAN_DX)),
    ]

    rows = []
    for name, edges, labels in schemes:
        row = {"scheme": name, "n_windows": len(labels)}
        for pool, pids in pools.items():
            a = audit(df, pids, patients, conditions, edges, labels)
            row[f"{pool}_density%"] = a["density"]
            row[f"{pool}_thin%"] = a["thin_pct"]
            row[f"{pool}_medN"] = a["median_slice_n"]
            row[f"{pool}_win_pass"] = f"{a['windows_pass']}/{len(labels)}"
        rows.append(row)
        print(f"{name:<22s} win={len(labels)}  "
              f"ASD: dens={row['ASD(145)_density%']:.1f}% thin={row['ASD(145)_thin%']}% "
              f"medN={row['ASD(145)_medN']} pass={row['ASD(145)_win_pass']}   |   "
              f"SPECTRUM: dens={row['spectrum(1648)_density%']:.1f}% "
              f"thin={row['spectrum(1648)_thin%']}% medN={row['spectrum(1648)_medN']} "
              f"pass={row['spectrum(1648)_win_pass']}")
    out = pd.DataFrame(rows)
    path = os.path.join(OUT, "s3_temporal_feasibility.csv")
    out.to_csv(path, index=False)
    print(f"\nValidation: ASD(145) column should match a0_feasibility_table.csv.")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
