import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, classification_report
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CSV_PATH = os.path.join(ROOT, "dataset", "startup_success_dataset.csv")
MODEL_DIR = os.path.join(ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "success_model.pkl")

NUMERIC = [
    "revenue_million", "revenue_growth_rate", "burn_rate_million",
    "runway_months", "funding_rounds", "team_size",
    "founder_experience_years", "has_technical_cofounder",
    "product_traction_users", "customer_growth_rate",
    "enterprise_customers", "market_size_billion",
]
CATEGORICAL = ["sector", "business_model", "geography"]
ALL_FEAT = NUMERIC + CATEGORICAL


def main():
    print(f"Loading {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"   {len(df):,} rows × {len(df.columns)} columns")

    available_numeric = [c for c in NUMERIC if c in df.columns]
    available_cat = [c for c in CATEGORICAL if c in df.columns]
    available = available_numeric + available_cat
    missing = [c for c in ALL_FEAT if c not in df.columns]
    if missing:
        print(f"\n Dataset missing {len(missing)} of 15 features:")
        for m in missing: print(f"     • {m}")
        print(f"   Training on {len(available)} available features.")
        print("   (Run scripts/build_dataset.py with Crunchbase data for full 15.)\n")

    df["success"] = df["outcome"].isin(["IPO", "Acquisition"]).astype(int)
    baseline = df["success"].mean()
    print(f"   Baseline success rate: {baseline:.1%}")

    for c in available_cat:
        df[c] = df[c].astype("category")

    X = df[available]
    y = df["success"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"\n Training LightGBM on {len(available)} features...")
    base = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        num_leaves=31, min_child_samples=20, random_state=42, verbose=-1)
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    clf.fit(X_train, y_train)

    y_prob = clf.predict_proba(X_test)[:, 1]
    print(f"\n ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}  (random = 0.5)")
    print("\n   Classification report on held-out test:")
    print(classification_report(y_test, clf.predict(X_test),
                                target_names=["Failure", "Success"]))

    print(" Refitting on full dataset for production use...")
    clf_final = CalibratedClassifierCV(base, method="isotonic", cv=3)
    clf_final.fit(X, y)
    base.fit(X, y)  

    bundle = {
        "model": clf_final,
        "feature_order": available,
        "feature_importances_": base.feature_importances_,
        "baseline_prob": float(baseline),
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    print(f"\n Saved model to {MODEL_PATH}")
    print(f"   File size: {os.path.getsize(MODEL_PATH) / 1024:.1f} KB")

    print("\n Top 10 features by importance:")
    for name, imp in sorted(zip(available, base.feature_importances_),
                            key=lambda x: -x[1])[:10]:
        print(f"   {name:30s}: {imp}")


if __name__ == "__main__":
    main()
