#!/usr/bin/env python3
"""Family-drop ablation for the bundled Round 5 diagnostic model.

Builds true category-dependence summaries for the methodology cards by
dropping one diagnostic feature family at a time and retraining the same
walk-forward architecture used by the shipped Round 5 horizons.

Outputs:
  diagnostics/family_ablation_h3/ablation_summary.csv
  diagnostics/family_ablation_h3/ablation_per_fold.csv
  diagnostics/family_ablation_h6/ablation_summary.csv
  diagnostics/family_ablation_h6/ablation_per_fold.csv
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

from walk_forward_audit_fixed import PANEL, _build_folds, evaluate, prepare

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = (3, 6)
TERRITORIES = {"72", "78", "60", "66", "69"}

MODEL_PARAMS = dict(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=5,
    reg_lambda=1.0,
    tree_method="hist",
    objective="binary:logistic",
    eval_metric="aucpr",
    early_stopping_rounds=25,
    random_state=42,
    verbosity=0,
)

FAMILY_LABEL = {
    "place_rural": "Rurality / place type",
    "place_persistent_pov": "Persistent poverty",
    "regime_flag": "HMDA availability regime",
    "cra_county_concentration": "CRA county concentration",
    "fdic_concentration": "FDIC concentration",
    "fdic_other": "FDIC branch / deposit structure",
    "hmda": "HMDA lending flow",
    "acs_demographics": "ACS demographics",
    "other": "Other",
}


def family(feat: str) -> str:
    if feat in {"is_rural", "ruca_code"}:
        return "place_rural"
    if feat == "is_persistent_poverty":
        return "place_persistent_pov"
    if feat == "has_hmda":
        return "regime_flag"
    if feat.startswith("cra_county_"):
        return "cra_county_concentration"
    if feat in {"fdic_deposit_hhi", "fdic_deposit_hhi_chg1yr", "fdic_deposit_hhi_chg3yr",
                "fdic_top_bank_share", "fdic_top_bank_share_chg1yr", "fdic_top_bank_share_chg3yr"}:
        return "fdic_concentration"
    if feat.startswith("fdic_"):
        return "fdic_other"
    if feat in {"n_applications", "n_originated", "n_denied", "n_withdrawn", "n_purchased",
                "approval_rate", "denial_rate", "sum_loan_amount", "mean_loan_amount",
                "n_distinct_lenders", "n_white", "n_black", "n_asian", "n_hispanic", "n_other_race"}:
        return "hmda"
    if feat in {"population", "median_hh_income", "pct_poverty", "pct_minority",
                "pct_black", "pct_hispanic", "housing_units", "pct_vacant",
                "unemployment_rate", "pct_bachelor_plus"}:
        return "acs_demographics"
    return "other"


def load_panel() -> pd.DataFrame:
    print(f"Loading {PANEL.name}…")
    df = pd.read_parquet(PANEL)
    df["state_fips"] = df["tract_fips"].str[:2]
    territory_mask = df["state_fips"].isin(TERRITORIES)
    dropped = int(territory_mask.sum())
    df = df[~territory_mask].copy()
    print(f"  Loaded: {df.shape}, dropped {dropped:,} territory rows (PR/VI/etc)")
    return df


def summarize_rows(rows: list[dict], lever_key: str, lever_label: str) -> dict:
    if not rows:
        return {}
    aucs = np.array([r["test_auc"] for r in rows], dtype=float)
    aps = np.array([r["test_ap"] for r in rows], dtype=float)
    lifts = np.array([r["test_ap_lift"] for r in rows], dtype=float)
    briers = np.array([r["test_brier_calibrated"] for r in rows], dtype=float)
    feature_counts = np.array([r["n_features"] for r in rows], dtype=float)
    return {
        "lever_dropped": lever_key,
        "lever_label": lever_label,
        "n_features": int(round(float(np.nanmean(feature_counts)))),
        "mean_test_auc": float(np.nanmean(aucs)),
        "std_test_auc": float(np.nanstd(aucs)),
        "mean_test_ap": float(np.nanmean(aps)),
        "std_test_ap": float(np.nanstd(aps)),
        "mean_test_ap_lift": float(np.nanmean(lifts)),
        "mean_test_brier_calibrated": float(np.nanmean(briers)),
        "folds_used": len(rows),
    }


def run_horizon(df_full: pd.DataFrame, horizon: int) -> None:
    target = f"target_becomes_service_desert_h{horizon}"
    out_dir = ROOT / "diagnostics" / f"family_ablation_h{horizon}"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = df_full[df_full[target].notna()].copy()
    df[target] = df[target].astype(int)
    print(f"\n=== Horizon h{horizon} ===")
    print(f"  Labeled rows: {df.shape}")
    print(f"  Positive rate: {df[target].mean() * 100:.2f}%")

    prepared = prepare(df).drop(columns=["year"])
    families = sorted({family(col) for col in prepared.columns})
    ablations = [("none_baseline", "Baseline")] + [(fam, FAMILY_LABEL.get(fam, fam)) for fam in families]

    summary_rows = []
    per_fold_rows = []

    for lever_key, lever_label in ablations:
        print(f"\n  [{lever_key}] {lever_label}")
        fold_rows = []
        for fold_name, tr_s, tr_e, val_y, te_s, te_e in _build_folds(horizon):
            train = df[(df["year"] >= tr_s) & (df["year"] <= tr_e)]
            val = df[df["year"] == val_y]
            test = df[(df["year"] >= te_s) & (df["year"] <= te_e)]
            if len(train) == 0 or len(val) == 0 or len(test) == 0:
                print(f"    {fold_name}: skip (empty split)")
                continue

            X_tr = prepare(train).drop(columns=["year"])
            y_tr = train[target].values
            X_val = prepare(val).drop(columns=["year"])
            y_val = val[target].values
            X_te = prepare(test).drop(columns=["year"])
            y_te = test[target].values

            common = sorted(set(X_tr.columns) & set(X_val.columns) & set(X_te.columns))
            X_tr, X_val, X_te = X_tr[common], X_val[common], X_te[common]

            if lever_key != "none_baseline":
                drop_cols = [col for col in common if family(col) == lever_key]
                X_tr = X_tr.drop(columns=drop_cols, errors="ignore")
                X_val = X_val.drop(columns=drop_cols, errors="ignore")
                X_te = X_te.drop(columns=drop_cols, errors="ignore")

            if X_tr.shape[1] == 0:
                print(f"    {fold_name}: skip (no features left)")
                continue

            model = xgb.XGBClassifier(**MODEL_PARAMS)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

            val_prob = model.predict_proba(X_val)[:, 1]
            test_prob = model.predict_proba(X_te)[:, 1]
            cal = IsotonicRegression(out_of_bounds="clip")
            cal.fit(val_prob, y_val)
            test_prob_calibrated = cal.transform(test_prob)
            metrics = evaluate(y_te, test_prob_calibrated)

            row = {
                "lever_dropped": lever_key,
                "lever_label": lever_label,
                "fold": fold_name,
                "train_years": f"{tr_s}-{tr_e}",
                "val_year": int(val_y),
                "test_years": f"{te_s}-{te_e}",
                "n_features": int(X_tr.shape[1]),
                "test_auc": float(metrics["auc"]),
                "test_ap": float(metrics["ap"]),
                "test_ap_lift": float(metrics["ap_lift"]),
                "test_brier_calibrated": float(metrics["brier"]),
                "test_pos_rate": float(metrics["pos_rate"]),
            }
            fold_rows.append(row)
            per_fold_rows.append(row)
            print(f"    {fold_name}: {X_tr.shape[1]} features, AUC {metrics['auc']:.4f}, AP {metrics['ap']:.4f}")

        summary = summarize_rows(fold_rows, lever_key, lever_label)
        if summary:
            summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        raise RuntimeError(f"No family ablation rows produced for h{horizon}")

    base = summary_df.loc[summary_df["lever_dropped"] == "none_baseline"].iloc[0]
    summary_df["delta_auc_vs_full"] = summary_df["mean_test_auc"] - float(base["mean_test_auc"])
    summary_df["delta_ap_vs_full"] = summary_df["mean_test_ap"] - float(base["mean_test_ap"])

    baseline_df = summary_df[summary_df["lever_dropped"] == "none_baseline"]
    lever_df = summary_df[summary_df["lever_dropped"] != "none_baseline"].sort_values("delta_ap_vs_full")
    summary_df = pd.concat([baseline_df, lever_df], ignore_index=True)

    summary_path = out_dir / "ablation_summary.csv"
    per_fold_path = out_dir / "ablation_per_fold.csv"
    summary_df.to_csv(summary_path, index=False)
    pd.DataFrame(per_fold_rows).to_csv(per_fold_path, index=False)
    print(f"\n  → {summary_path}")
    print(f"  → {per_fold_path}")


def main() -> None:
    df_full = load_panel()
    for horizon in HORIZONS:
        run_horizon(df_full, horizon)


if __name__ == "__main__":
    main()
