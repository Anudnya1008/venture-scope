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
MODEL_PATH = os.path.join(ROOT, "models", "success_model.pkl")

NUMERIC = [
    "revenue_million", "revenue_growth_rate", "burn_rate_million",
    "runway_months", "funding_rounds", "team_size",
    "founder_experience_years", "has_technical_cofounder",
    "product_traction_users", "customer_growth_rate",
    "enterprise_customers", "market_size_billion",
]
CATEGORICAL = ["sector", "business_model", "geography"]

def main():
    df = pd.read_csv(CSV_PATH)

    available_numeric = [c for c in NUMERIC if c in df.columns]
    available_cat = [c for c in CATEGORICAL if c in df.columns]
    available = available_numeric + available_cat

    #re-balancing code
    df["success"] = df["outcome"].isin(["IPO", "Acquisition"]).astype(int)
    baseline = df["success"].mean()
    failures = df[df["success"] == 0]
    target_successes = int(len(failures) * 0.50 / 0.50)
    successes = df[df["success"] == 1].sample(n=target_successes, random_state=42)
    df = pd.concat([failures, successes]).sample(frac=1, random_state=42).reset_index(drop=True)

    for c in available_cat:
        df[c] = df[c].astype("category")

    X = df[available]
    y = df["success"]
   
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    base = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=20,
        #class_weight="balanced",
        random_state=42,
        verbose=-1,
    )

    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    clf.fit(X_train, y_train)

    y_prob = clf.predict_proba(X_test)[:, 1]
    clf_final = CalibratedClassifierCV(base, method="isotonic", cv=3)
    clf_final.fit(X, y)
    base.fit(X, y)

    bundle = {
        "model": clf_final,
        "feature_order": available,
        "feature_importances_": base.feature_importances_,
        "baseline_prob": float(baseline),
    }

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)


if __name__ == "__main__":
    main()