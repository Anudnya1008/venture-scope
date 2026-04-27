import os
import json
import pandas as pd
from google.genai import types
from config import client, MODEL

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CSV_PATH = os.path.join(ROOT, "dataset", "startup_success_dataset.csv")

METRIC_COLS = [
    "revenue_million", "revenue_growth_rate", "burn_rate_million",
    "runway_months", "funding_rounds", "team_size",
    "founder_experience_years", "has_technical_cofounder",
    "product_traction_users", "customer_growth_rate",
    "enterprise_customers", "market_size_billion",
]
LOWER_IS_BETTER = {"burn_rate_million"}


def _load_dataset() -> pd.DataFrame:
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {CSV_PATH}. "
            f"Run scripts/build_dataset.py or place the synthetic CSV there.")
    return pd.read_csv(CSV_PATH)


def _filter_peers(df: pd.DataFrame, csv_features: dict):
    """Progressive refinement: sector → business_model → geography,
    only narrowing when ≥30 peers remain."""
    sector = csv_features.get("sector", "Unknown")
    peers = df[df["sector"].str.lower() == str(sector).lower()]
    refined_by = f"sector={sector}"

    if len(peers) < 30:
        return peers, refined_by

    bm = csv_features.get("business_model")
    if bm and "business_model" in peers.columns:
        narrower = peers[peers["business_model"].str.lower() == str(bm).lower()]
        if len(narrower) >= 30:
            peers = narrower
            refined_by += f" + business_model={bm}"

    geo = csv_features.get("geography")
    if geo and "geography" in peers.columns:
        narrower = peers[peers["geography"].str.lower() == str(geo).lower()]
        if len(narrower) >= 30:
            peers = narrower
            refined_by += f" + geography={geo}"

    return peers, refined_by


def compute_peer_stats(df: pd.DataFrame, csv_features: dict) -> dict:
    """Compute median / quartile / success-rate stats."""
    peers, refined_by = _filter_peers(df, csv_features)
    if len(peers) == 0:
        return {"n_peers": 0, "error": "No peers found"}

    success = peers["outcome"].isin(["IPO", "Acquisition"])

    metrics = {}
    for col in METRIC_COLS:
        if col not in peers.columns:
            continue
        metrics[col] = {
            "median": round(float(peers[col].median()), 2),
            "p25":    round(float(peers[col].quantile(0.25)), 2),
            "p75":    round(float(peers[col].quantile(0.75)), 2),
            "lower_is_better": col in LOWER_IS_BETTER,
        }

    return {
        "n_peers": int(len(peers)),
        "refined_by": refined_by,
        "success_rate_pct": round(100 * float(success.mean()), 1),
        "n_ipo": int((peers["outcome"] == "IPO").sum()),
        "n_acquisition": int((peers["outcome"] == "Acquisition").sum()),
        "n_failure": int((peers["outcome"] == "Failure").sum()),
        "metrics": metrics,
    }


benchmark_schema = {
    "type": "OBJECT",
    "properties": {
        "benchmarks": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "metric":        {"type": "STRING"},
                    "startup_value": {"type": "STRING"},
                    "peer_median":   {"type": "NUMBER"},
                    "verdict":       {"type": "STRING"},  # ABOVE | AT | BELOW
                    "note":          {"type": "STRING"},
                },
                "required": ["metric", "verdict", "note"],
            },
        },
        "success_probability": {"type": "INTEGER"},
        "overall_fit":         {"type": "STRING"},
        "reality_check":       {"type": "STRING"},
    },
    "required": ["benchmarks", "overall_fit", "reality_check"],
}


def run_benchmark(deck_facts: dict) -> dict:
    """Compare startup against real peers from the same dataset the ML uses."""
    print(" Benchmarker analyzing (12-feature peer set)...")
    df = _load_dataset()

    csv_feats = deck_facts.get("csv_features") or {}
    peer_stats = compute_peer_stats(df, csv_feats)

    if peer_stats.get("n_peers", 0) == 0:
        return {
            "benchmarks": [],
            "success_probability": 0,
            "overall_fit": "Unknown",
            "reality_check": peer_stats.get("error", "No peer data."),
            "peer_stats": peer_stats,
        }

    startup_values = {
        m: csv_feats.get(m) for m in METRIC_COLS
        if csv_feats.get(m) is not None and csv_feats.get(m) != 0
    }

    prompt = f"""
You are "The Benchmarker" — a VC data analyst.

You have REAL peer statistics from {peer_stats['n_peers']} startups
({peer_stats['refined_by']}).

Historical outcomes:
  • Success rate: {peer_stats['success_rate_pct']}%
    ({peer_stats['n_ipo']} IPO + {peer_stats['n_acquisition']} Acq /
     {peer_stats['n_failure']} Failure)

Peer metric distributions (median | P25-P75 | lower_is_better flag):
{json.dumps(peer_stats['metrics'], indent=2)}

Startup's own values:
{json.dumps(startup_values, indent=2)}

Full deck context:
{json.dumps(deck_facts, indent=2, ensure_ascii=False)}

For each metric where the startup has a value, classify ABOVE / AT / BELOW
peer median. AT = within P25-P75. When lower_is_better=true, the startup's
LOWER value is BETTER and should be ABOVE.
Skip metrics the startup didn't report.

success_probability: 0-100. Anchor on peer success rate ({peer_stats['success_rate_pct']}%),
adjust based on how many ABOVE-median metrics the startup has.
overall_fit: "Strong" | "Average" | "Weak"
reality_check: 2-3 sentences citing actual numbers.
"""
    try:
        response = client.models.generate_content(
            model=MODEL, contents=[prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=benchmark_schema))
        result = response.parsed
    except Exception as e:
        if "429" in str(e):
            return {"error": "Quota exceeded."}
        return {"error": str(e)}

    result["peer_stats"] = peer_stats
    return result
