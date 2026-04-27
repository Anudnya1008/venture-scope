import os
import numpy as np
import pandas as pd
 
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW_DIR = os.path.join(ROOT, "dataset", "crunchbase_raw")
OUT_PATH = os.path.join(ROOT, "dataset", "startup_success_dataset.csv")
 
np.random.seed(42)
 
SECTOR_MAP = {
    "web": "SaaS", "software": "SaaS", "enterprise": "SaaS",
    "analytics": "SaaS", "cloud_computing": "SaaS",
    "biotech": "Health", "health": "Health", "medical": "Health",
    "health_and_wellness": "Health",
    "finance": "Fintech", "fintech": "Fintech",
    "ecommerce": "Ecommerce", "fashion": "Ecommerce", "retail": "Ecommerce",
    "clean_technology": "Climate", "energy": "Climate",
    "bitcoin": "Crypto", "cryptocurrency": "Crypto", "blockchain": "Crypto",
    "artificial_intelligence": "AI", "machine_learning": "AI", "robotics": "AI",
}
 
SECTOR_TO_BUSINESS_MODEL = {
    "SaaS": "Subscription", "Health": "Subscription",
    "Fintech": "Transaction-fee", "Ecommerce": "Marketplace",
    "Climate": "Hardware", "Crypto": "Transaction-fee",
    "AI": "Subscription", "Other": "Subscription",
}
SECTOR_TAM_BILLION = {"SaaS": 250, "Health": 350, "Fintech": 200, "Ecommerce": 180,
                      "Climate": 150, "Crypto": 80, "AI": 220, "Other": 100}
SECTOR_REV_MULTIPLE = {"SaaS": 0.8, "Health": 0.4, "Fintech": 0.6, "Ecommerce": 1.2,
                       "Climate": 0.3, "Crypto": 0.5, "AI": 0.4, "Other": 0.5}
SECTOR_TECH_COFOUNDER_PROB = {"SaaS": 0.85, "AI": 0.90, "Crypto": 0.80, "Fintech": 0.65,
                              "Health": 0.55, "Ecommerce": 0.50, "Climate": 0.70, "Other": 0.55}
SECTOR_ENTERPRISE_BIAS = {"SaaS": 0.7, "AI": 0.6, "Fintech": 0.5, "Health": 0.4,
                          "Climate": 0.6, "Crypto": 0.3, "Ecommerce": 0.2, "Other": 0.4}
COUNTRY_TO_REGION = {
    "USA": "North America", "CAN": "North America", "MEX": "North America",
    "GBR": "Europe", "DEU": "Europe", "FRA": "Europe", "NLD": "Europe",
    "ESP": "Europe", "ITA": "Europe", "SWE": "Europe", "CHE": "Europe",
    "IND": "Asia", "CHN": "Asia", "JPN": "Asia", "SGP": "Asia",
    "ISR": "Middle East", "AUS": "Oceania",
}
 
 
def _bucket_sector(c):
    if pd.isna(c): return "Other"
    return SECTOR_MAP.get(str(c).lower(), "Other")
 
 
def _bucket_outcome(s):
    if pd.isna(s): return None
    s = str(s).lower()
    return {"ipo": "IPO", "acquired": "Acquisition", "closed": "Failure"}.get(s)
 
 
def _classify_investor(name):
    if pd.isna(name): return "none"
    n = str(name).lower()
    tier1 = {"sequoia", "andreessen", "a16z", "accel", "kleiner",
             "benchmark", "founders fund", "greylock", "khosla",
             "general catalyst", "lightspeed"}
    if any(t in n for t in tier1): return "tier1_vc"
    if "angel" in n: return "angel"
    return "tier2_vc"
 
 
def _bucket_geography(c):
    if pd.isna(c): return "Other"
    return COUNTRY_TO_REGION.get(str(c).upper(), "Other")
 
 
def build():
    print(f" Loading raw Crunchbase tables from {RAW_DIR}\n")
 
    # ─── objects.csv ───
    objects = pd.read_csv(os.path.join(RAW_DIR, "objects.csv"), low_memory=False)
    print(f"   objects.csv:        {len(objects):,} rows")
    companies = objects[objects["entity_type"] == "Company"].copy()
    print(f"   → companies only:   {len(companies):,}")
 
    collision_cols = ["funding_rounds", "first_funding_at", "last_funding_at",
                      "funding_total_usd"]
    companies = companies.drop(columns=[c for c in collision_cols if c in companies.columns])
 
    rounds = pd.read_csv(os.path.join(RAW_DIR, "funding_rounds.csv"), low_memory=False)
    print(f"   funding_rounds.csv: {len(rounds):,} rows")
    rounds["funded_at"] = pd.to_datetime(rounds["funded_at"], errors="coerce")
 
    rounds_agg = rounds.groupby("object_id").agg(
        funding_rounds=("id", "count"),                       
        total_raised=("raised_amount_usd", "sum"),
        last_round_amount=("raised_amount_usd", "last"),
        last_round_date=("funded_at", "max"),
        first_round_date=("funded_at", "min"),
    ).reset_index()
    print(f"   → aggregated to:    {len(rounds_agg):,} companies with funding")
 
    investments = pd.read_csv(os.path.join(RAW_DIR, "investments.csv"), low_memory=False)
    print(f"   investments.csv:    {len(investments):,} rows")
 
    investor_lookup = objects[["id", "name"]].rename(
        columns={"id": "investor_object_id", "name": "_investor_name"})
    investments = investments.merge(investor_lookup, on="investor_object_id", how="left")
 
    investments["tier"] = investments["_investor_name"].apply(_classify_investor)
    priority = {"tier1_vc": 0, "tier2_vc": 1, "angel": 2, "none": 3}
    investments["_p"] = investments["tier"].map(priority)
    top_inv = (investments.sort_values("_p")
               .drop_duplicates("funded_object_id")
               [["funded_object_id", "tier"]]
               .rename(columns={"funded_object_id": "object_id",
                                "tier": "investor_type"}))
 
    df = companies.merge(rounds_agg, left_on="id", right_on="object_id", how="left")
    df = df.merge(top_inv, on="object_id", how="left")
    print(f"   merged dataframe:   {len(df):,} rows")
 
    print(f"\n🔧 Engineering 15 features...")
    n = len(df)
    out = pd.DataFrame()
 
    out["sector"] = df["category_code"].apply(_bucket_sector)
 
    out["revenue_million"] = (
        df["total_raised"].fillna(0) / 1e6 * out["sector"].map(SECTOR_REV_MULTIPLE)
    ).round(2)
 
    df["last_round_date"] = pd.to_datetime(df["last_round_date"], errors="coerce")
    df["first_round_date"] = pd.to_datetime(df["first_round_date"], errors="coerce")
    age_y = ((df["last_round_date"] - df["first_round_date"]).dt.days / 365).clip(lower=0.5)
    out["revenue_growth_rate"] = (
        df["funding_rounds"].fillna(0) / age_y * 60
    ).clip(0, 300).round(1)
 
    out["burn_rate_million"] = (
        df["total_raised"].fillna(0) / 1e6 / age_y
    ).fillna(5.0).round(2)
 
    monthly_burn = (out["burn_rate_million"] / 12).clip(lower=0.1)
    out["runway_months"] = (
        df["last_round_amount"].fillna(0) / 1e6 * 0.6 / monthly_burn
    ).clip(0, 60).round(1)
 
    out["funding_rounds"] = df["funding_rounds"].fillna(0).astype(int)
 
    out["team_size"] = np.random.randint(5, 300, size=n)
 
    out["founder_experience_years"] = np.random.randint(0, 25, size=n)
 
    probs = out["sector"].map(SECTOR_TECH_COFOUNDER_PROB).values
    out["has_technical_cofounder"] = (np.random.random(n) < probs).astype(int)
 
    out["product_traction_users"] = (
        out["funding_rounds"].clip(lower=1) *
        np.random.randint(20_000, 200_000, size=n)
    ).astype(int)
 
    noise = np.random.normal(0, 20, size=n)
    out["customer_growth_rate"] = (
        out["revenue_growth_rate"] * 0.8 + noise
    ).clip(0, 400).round(1)
 
    bias = out["sector"].map(SECTOR_ENTERPRISE_BIAS).values
    out["enterprise_customers"] = (
        out["product_traction_users"] * bias / 1000
    ).astype(int).clip(0, 500)
 
    out["market_size_billion"] = out["sector"].map(SECTOR_TAM_BILLION)
 
 
    out["business_model"] = out["sector"].map(SECTOR_TO_BUSINESS_MODEL)
 
    out["geography"] = df["country_code"].apply(_bucket_geography)
 
    out["investor_type"] = df["investor_type"].fillna("none")
    out["founder_background"] = np.random.choice(
        ["academic", "first_time", "ex_bigtech", "serial_founder"],
        size=n, p=[0.15, 0.45, 0.25, 0.15])
 
    out["outcome"] = df["status"].apply(_bucket_outcome)
    before = len(out)
    out = out.dropna(subset=["outcome"]).reset_index(drop=True)
    print(f"   dropped 'operating' (censored): {before - len(out):,} rows")
 
    print(f"\n Final dataset: {len(out):,} rows × {len(out.columns)} columns")
    print(f"\n   Outcome distribution:")
    print(out["outcome"].value_counts().to_string())
 
    out.to_csv(OUT_PATH, index=False)
    print(f"\n Saved to {OUT_PATH}")
 
 
if __name__ == "__main__":
    build()