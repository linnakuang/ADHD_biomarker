"""Shared bootstrap + helpers for the Track-A temporal analyses (gus_analysis).

Importing this module wires sys.path so the repo's existing tooling
(clustering_utils, standardize) resolves against our LOCAL config.py (real Linux
paths) rather than the repo's Windows-path config. Import this FIRST in every
analysis script:

    import gus_common as gc
    df = gc.load_event_data(); patients = gc.load_patients()

It re-exports the reused repo functions and adds small shared helpers for the
pre-registered §2 decision rules.
"""
import os
import sys

GUS_DIR = "<PROJECT_ROOT>/gus_analysis"
CLUSTERING_DIR = "<PROJECT_ROOT>/BBC_Dataset/clustering"
ANALYSIS_DIR = "<PROJECT_ROOT>/BBC_Dataset/analysis"

# Order matters: GUS_DIR first so `import config` resolves to our fixed-path config,
# not the repo's Windows-path one.
for p in (CLUSTERING_DIR, ANALYSIS_DIR, GUS_DIR):
    if p in sys.path:
        sys.path.remove(p)
sys.path.insert(0, CLUSTERING_DIR)
sys.path.insert(0, ANALYSIS_DIR)
sys.path.insert(0, GUS_DIR)

import numpy as np
import pandas as pd

# Reused repo tooling (all now resolve config -> gus_analysis/config.py).
from clustering_utils import (  # noqa: E402
    load_event_data, load_patients, build_residual_matrix,
    select_nmf_conditions, _build_time_bins,
)
import config  # noqa: E402  (our local config)

OUT_DIR = config.OUT_DIR
os.makedirs(OUT_DIR, exist_ok=True)


# ── Raw binary matrix for an ARBITRARY set of children (Track B) ──
def raw_binary_matrix(df, ids, keep_conditions):
    """Binary (child x condition) presence matrix for the given `ids`.

    Unlike build_residual_matrix (which returns the NT-standardized residual for the
    ASD children only), this returns the RAW 0/1 co-occurrence matrix for any id set —
    needed for B1 co-occurrence and the B2 NT-side normative fit. Rows are ordered as
    `ids`; columns are sorted(keep_conditions). Children with no events are all-zero.
    """
    ids = list(ids)
    conditions = sorted(keep_conditions)
    evt = df[df["condition"].isin(conditions)].drop_duplicates(["ID", "condition"]).copy()
    evt["value"] = 1
    raw = (evt.pivot_table(index="ID", columns="condition", values="value", fill_value=0)
              .reindex(index=ids, columns=conditions, fill_value=0))
    return raw.values.astype(np.float32), conditions


def asd_nt_ids(patients):
    """Return (asd_ids, nt_ids) as sorted lists from a patients frame."""
    asd = sorted(patients.loc[patients["label"] == "ASD", "ID"])
    nt = sorted(patients.loc[patients["label"] == "NT", "ID"])
    return asd, nt


# ── Normalization used by the primary NMF (clip at 0, unit L2 rows) ──
def normalize_for_nmf(residual):
    """Clip negative residuals, L2-normalize each row. Returns (X_fit, has_burden)."""
    X = np.clip(residual, 0, None)
    has_burden = X.sum(axis=1) > 0
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms, has_burden


# ── Custom age-bin builders (truncation schemes from §3.2) ──
def dx_anchored_bins(median_dx, n_windows=3, width=None):
    """Dx-anchored windows: [0, m), [m, 2m'), ... Returns (edges, labels).

    Default reproduces §3.3 A1: [0,3.25)/[3.25,6.25)/[6.25,9.25) when median_dx=3.25
    with width=3.0.
    """
    if width is None:
        width = 3.0
    edges = [0.0, float(median_dx)]
    for _ in range(n_windows - 1):
        edges.append(edges[-1] + width)
    labels = [f"[0,{edges[1]:.2f})"] + [
        f"[{edges[i]:.2f},{edges[i+1]:.2f})" for i in range(1, len(edges) - 1)
    ]
    return edges, labels


def truncated_bins(cut, width):
    """Uniform windows of `width` years from 0 up to `cut`. Returns (edges, labels)."""
    edges = list(np.arange(0.0, cut + 1e-9, width))
    if edges[-1] < cut:
        edges.append(float(cut))
    labels = [f"[{edges[i]:.1f},{edges[i+1]:.1f})" for i in range(len(edges) - 1)]
    return edges, labels


def two_window_bins(median_dx):
    """Pre/post-diagnosis split (§3.5 A3). Returns (edges, labels)."""
    return [0.0, float(median_dx), float(config.AGE_CUTOFF)], ["Pre-dx", "Post-dx"]


# ── Density / slice-occupancy audit for a given binning (§2, §3.2) ──
def slice_audit(df, asd_patient_ids, keep_conditions, bin_edges, bin_labels):
    """Build the per-window residual slices, report density + thin-slice fraction.

    Returns dict with: slices (list of clipped arrays), feature_cols, X (I x J x K),
    density, thin_frac, median_slice_n, per_window (list of dicts), window_keep (bool
    list of windows clearing the §2 window rule).
    """
    slices, feature_cols, per_window = [], None, []
    for lo, hi, label in zip(bin_edges[:-1], bin_edges[1:], bin_labels):
        df_w = df[(df["age_years"] >= lo) & (df["age_years"] < hi)]
        resid, feature_cols, _ = build_residual_matrix(
            df_w, asd_patient_ids, kept_conditions=keep_conditions)
        clipped = np.clip(resid, 0, None)
        slices.append(clipped)
        nz = clipped > 0
        n_asd_w = df_w.loc[df_w["label"] == "ASD", "ID"].nunique()
        slice_n = nz.sum(axis=0)                      # children per condition in window
        n_pass = int((slice_n >= config.DENSITY_FLOOR_CHILDREN).sum())
        per_window.append({
            "window": label, "asd_with_event": int(n_asd_w),
            "density": float(nz.mean()), "nonzero_cells": int(nz.sum()),
            "conditions_pass_floor": n_pass, "n_conditions": len(feature_cols),
            "frac_conditions_pass": n_pass / max(1, len(feature_cols)),
        })

    X = np.stack(slices, axis=2)                       # I x J x K
    nz = X > 0
    slice_n = nz.sum(axis=0)                            # J x K children per (cond,window)
    thin = int((slice_n < config.DENSITY_FLOOR_CHILDREN).sum())
    window_keep = [w["frac_conditions_pass"] >= config.WINDOW_KEEP_FRAC for w in per_window]
    return {
        "slices": slices, "feature_cols": feature_cols, "X": X,
        "density": float(nz.mean()),
        "thin_frac": thin / slice_n.size,
        "median_slice_n": float(np.median(slice_n)),
        "per_window": per_window, "window_keep": window_keep,
    }


# ── Frobenius-norm tensor normalization (per-child unit slice) ──
def normalize_tensor(X):
    """Scale each child's whole slice to unit Frobenius norm. Returns (X_fit, has_burden)."""
    I = X.shape[0]
    slice_norm = np.linalg.norm(X.reshape(I, -1), axis=1)
    has_burden = slice_norm > 0
    X_fit = X / np.where(slice_norm, slice_norm, 1)[:, None, None]
    return X_fit, has_burden


def print_decision(name, cosine=None, concordance=None, artifact_r=None,
                   density=None, thin_frac=None):
    """Print a §2 pass/fail verdict block."""
    import config as _c
    print(f"\n=== §2 DECISION: {name} ===")
    verdict = []
    if density is not None:
        print(f"  density              = {density:.2%}")
    if thin_frac is not None:
        print(f"  thin-slice fraction  = {thin_frac:.0%}")
    if cosine is not None:
        ok = cosine >= _c.STABILITY_COSINE_FLOOR
        verdict.append(ok)
        print(f"  bootstrap cosine     = {cosine:.3f}  "
              f"({'>= ' if ok else '< '}{_c.STABILITY_COSINE_FLOOR} "
              f"{'PASS' if ok else 'FAIL'})")
    if concordance is not None:
        ok = concordance >= _c.ASSIGN_CONCORDANCE_FLOOR
        verdict.append(ok)
        print(f"  assign concordance   = {concordance:.3f}  "
              f"({'>= ' if ok else '< '}{_c.ASSIGN_CONCORDANCE_FLOOR} "
              f"{'PASS' if ok else 'FAIL'})")
    if artifact_r is not None:
        bad = artifact_r > _c.DENSITY_ARTIFACT_R
        verdict.append(not bad)
        print(f"  max temporal r(dens) = {artifact_r:.3f}  "
              f"({'ARTIFACT' if bad else 'ok'})")
    if verdict:
        print(f"  --> {'REPORTABLE' if all(verdict) else 'NOT reportable (see §7)'}")
    return all(verdict) if verdict else None
