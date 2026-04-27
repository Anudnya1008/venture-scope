import os
import joblib
import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "revenue_million", "revenue_growth_rate", "burn_rate_million",
    "runway_months", "funding_rounds", "team_size",
    "founder_experience_years", "has_technical_cofounder",
    "product_traction_users", "customer_growth_rate",
    "enterprise_customers", "market_size_billion",
]
CATEGORICAL_FEATURES = ["sector", "business_model", "geography"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

LOWER_IS_BETTER = {"burn_rate_million"}

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL_PATH = os.path.join(ROOT, "models", "success_model.pkl")
CSV_PATH = os.path.join(ROOT, "dataset", "startup_success_dataset.csv")

_cache = None
_medians_cache = None


def load_model():
    global _cache
    if _cache is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Run `python scripts/train_model.py` first.")
        _cache = joblib.load(MODEL_PATH)
    return _cache


def load_medians(sector: str = None) -> dict:
    """Load per-feature medians for the relevant sector (or global)."""
    global _medians_cache
    if _medians_cache is None:
        if not os.path.exists(CSV_PATH):
            _medians_cache = {}
            return {}
        df = pd.read_csv(CSV_PATH)
        # Cache both global and per-sector medians
        _medians_cache = {"_global": {}}
        for f in NUMERIC_FEATURES:
            if f in df.columns:
                _medians_cache["_global"][f] = float(df[f].median())
        for s in df["sector"].dropna().unique():
            sector_df = df[df["sector"] == s]
            _medians_cache[s] = {
                f: float(sector_df[f].median())
                for f in NUMERIC_FEATURES if f in sector_df.columns
            }

    # Prefer sector-specific medians when available
    if sector and sector in _medians_cache:
        return _medians_cache[sector]
    return _medians_cache.get("_global", {})


def _to_dataframe(csv_features: dict, feature_order: list) -> pd.DataFrame:
    row = {}
    for f in feature_order:
        if f in CATEGORICAL_FEATURES:
            row[f] = csv_features.get(f, "Unknown")
        else:
            v = csv_features.get(f)
            row[f] = float(v) if (v is not None and v != 0) else np.nan
    df = pd.DataFrame([row])
    for c in CATEGORICAL_FEATURES:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df


def _humanize(name: str) -> str:
    return (name.replace("_", " ")
            .replace("million", "($M)")
            .replace("billion", "($B)")
            .replace("rate", "rate (%)"))


def _driver_sign(feature: str, value: float, median: float) -> str:
    """Return '+' if value is favorable vs median, '-' if unfavorable."""
    if median == 0 or value is None:
        return "?"
    if feature in LOWER_IS_BETTER:
        return "+" if value < median else "-"
    return "+" if value > median else "-"


def predict_success(csv_features: dict) -> dict:
    """Return success probability + tuned label + top driver features
    with per-feature direction (+ or - vs sector median)."""
    bundle = load_model()
    model = bundle["model"]
    threshold = bundle.get("optimal_threshold", 0.5)
    feature_order = bundle["feature_order"]
    importances = dict(zip(feature_order, bundle["feature_importances_"]))
    total_imp = sum(importances.values()) or 1

    # Pull per-sector medians for honest +/- direction calls
    sector = csv_features.get("sector", None)
    medians = load_medians(sector)

    X = _to_dataframe(csv_features, feature_order)
    prob = float(model.predict_proba(X)[0][1])
    predicted_label = "Success" if prob >= threshold else "Failure"

    distance = abs(prob - threshold)
    confidence = "HIGH" if distance >= 0.25 else "MEDIUM" if distance >= 0.10 else "LOW"

    drivers = []
    for f in NUMERIC_FEATURES:
        if f not in feature_order:
            continue
        v = csv_features.get(f)
        if v is None or v == 0:
            continue   # skip missing features
        median = medians.get(f, 0)
        sign = _driver_sign(f, float(v), median)
        weight = importances.get(f, 0)
        contribution_pts = round(weight / total_imp * 100, 1)
        drivers.append({
            "feature": _humanize(f),
            "value": v,
            "vs_peer_median": round(median, 2) if median else None,
            "contribution": f"{sign}{contribution_pts} pts",
            "_w": weight,
        })
    drivers.sort(key=lambda d: d["_w"], reverse=True)
    for d in drivers: d.pop("_w")

    return {
        "success_probability": round(prob * 100, 1),
        "predicted_label": predicted_label,
        "decision_threshold": round(threshold * 100, 1),
        "model_confidence": confidence,
        "top_drivers": drivers[:3],
    }