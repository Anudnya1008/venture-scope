
HIGH_SIGNAL_FEATURES = [
    "revenue_million", "revenue_growth_rate", "funding_rounds",
    "founder_experience_years", "product_traction_users",
]

ALL_NUMERIC_FEATURES = HIGH_SIGNAL_FEATURES + [
    "burn_rate_million", "runway_months", "team_size",
    "has_technical_cofounder", "customer_growth_rate",
    "enterprise_customers", "market_size_billion",
]


def _present(v):
    if v is None: return False
    try: return float(v) > 0
    except (TypeError, ValueError): return False


def check_completeness(facts: dict) -> dict:
    csv_features = facts.get("csv_features") or {}
    high = sum(1 for f in HIGH_SIGNAL_FEATURES if _present(csv_features.get(f)))
    total = sum(1 for f in ALL_NUMERIC_FEATURES if _present(csv_features.get(f)))
    missing = [f for f in HIGH_SIGNAL_FEATURES if not _present(csv_features.get(f))]

    if high >= 4:
        conf, can_run = "HIGH", True
        msg = f"{high}/5 key features extracted — ML reliable."
    elif high >= 2:
        conf, can_run = "MEDIUM", True
        msg = f"Partial extraction ({high}/5). ML included with lower confidence."
    else:
        conf, can_run = "LOW", False
        msg = f"Only {high}/5 key features. ML skipped — qualitative only."

    return {
        "high_signal_present": high,
        "all_features_present": total,
        "confidence": conf,
        "can_run_ml": can_run,
        "missing": missing,
        "explanation": msg,
    }
