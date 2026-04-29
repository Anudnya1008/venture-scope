import pandas as pd

METRIC_COLS = [
    "revenue_million",
    "revenue_growth_rate",
    "burn_rate_million",
    "runway_months",
    "funding_rounds",
    "team_size",
    "founder_experience_years",
    "has_technical_cofounder",
    "product_traction_users",
    "customer_growth_rate",
    "enterprise_customers",
    "market_size_billion",
]

LOWER_IS_BETTER = {"burn_rate_million"}


def get_sector_peers(df, sector):
    return df[df["sector"].str.lower() == str(sector).lower()]


def peer_success_rate(peers):
    if len(peers) == 0:
        return 50.0
    rate = peers["outcome"].isin(["IPO", "Acquisition"]).mean()
    return round(100 * rate, 1)


def compare_to_peer_median(value, median, lower_is_better):
    if median == 0:
        return None

    ratio = value / median
    if 0.9 <= ratio <= 1.1:
        return "AT"

    is_above = ratio > 1.1
    if lower_is_better:
        is_above = not is_above
    return "ABOVE" if is_above else "BELOW"


def metric_performance(peers, startup_features):
    if len(peers) == 0:
        return {"metric_score": 50.0, "above": 0, "below": 0, "comparisons": []}

    comparisons = []
    above = 0
    below = 0

    for col in METRIC_COLS:
        if col not in peers.columns:
            continue

        value = startup_features.get(col)
        if value is None or value == 0:
            continue

        median = float(peers[col].median())
        verdict = compare_to_peer_median(value, median, col in LOWER_IS_BETTER)
        if verdict is None:
            continue

        comparisons.append({
            "metric": col,
            "startup_value": value,
            "peer_median": round(median, 2),
            "verdict": verdict,
        })

        if verdict == "ABOVE":
            above += 1
        elif verdict == "BELOW":
            below += 1

    total = above + below
    metric_score = round(above / total * 100, 1) if total else 50.0

    return {
        "metric_score": metric_score,
        "above": above,
        "below": below,
        "comparisons": comparisons,
    }


def bull_bear_balance(bull, bear):
    green = len(bull.get("green_flags", []))
    red = len(bear.get("red_flags", []))
    if green + red == 0:
        return 50.0
    return round(100 * green / (green + red), 1)


def risk_level(score):
    if score >= 75:
        return "Low"
    if score >= 50:
        return "Medium"
    return "High"


RISK_PHRASES = {
    "Low": "LOW RISK, data and analysis are both favorable.",
    "Medium": "MEDIUM RISK, real upside but caveats need investigation.",
    "High": "HIGH RISK, data does not support a confident bet.",
}


def build_reasoning(score, risk, peer_rate, n_peers, sector, perf, bull, bear, ml):
    parts = []

    if ml:
        parts.append(
            f"ML model estimates {ml['success_probability']}% success "
            f"probability based on {sector} sector patterns."
        )
    elif n_peers == 0:
        parts.append(f"No peers in dataset for sector '{sector}'.")
    else:
        parts.append(
            f"Of {n_peers} {sector} peers, {peer_rate}% reached IPO or acquisition."
        )

    above, below = perf["above"], perf["below"]
    total = above + below

    if total == 0:
        parts.append("Insufficient deck data for peer benchmarking.")
    else:
        comparisons = perf["comparisons"]
        target_verdict = "ABOVE" if above >= below else "BELOW"
        notable = next(
            (c for c in comparisons if c["verdict"] == target_verdict),
            None,
        )
        sentence = f"Beats peer median on {above}/{total} metrics"
        if notable:
            metric_name = notable["metric"].replace("_", " ")
            sentence += (
                f", notably {metric_name} "
                f"({notable['startup_value']} vs {notable['peer_median']})."
            )
        else:
            sentence += "."
        parts.append(sentence)

    green = len(bull.get("green_flags", []))
    red = len(bear.get("red_flags", []))
    if green > red:
        parts.append(f"Agent debate positive: {green} green vs {red} red flags.")
    elif red > green:
        parts.append(f"Agent debate cautious: {red} red vs {green} green flags.")
    else:
        parts.append(f"Agent debate balanced: {green} green, {red} red.")

    parts.append(f"Score {score}/100, {RISK_PHRASES[risk]}")
    return " ".join(parts)


def compute_vitality(facts, df, bull, bear, ml_result=None):
    csv_features = facts.get("csv_features") or {}
    sector = csv_features.get("sector", "Unknown")

    peers = get_sector_peers(df, sector)
    n_peers = len(peers)
    peer_rate = peer_success_rate(peers)
    perf = metric_performance(peers, csv_features)
    balance = bull_bear_balance(bull, bear)

    if ml_result is not None:
        score = 0.4 * ml_result["success_probability"] + 0.3 * peer_rate + 0.3 * balance
        formula = "40% ML + 30% peer rate + 30% agent balance"
    else:
        score = 0.5 * peer_rate + 0.5 * balance
        formula = "50% peer rate + 50% agent balance (ML skipped)"

    score = round(score, 1)
    risk = risk_level(score)

    return {
        "vitality_score": score,
        "risk_level": risk,
        "reasoning": build_reasoning(
            score, risk, peer_rate, n_peers, sector, perf, bull, bear, ml_result
        ),
        "formula": formula,
        "breakdown": {
            "ml_success_probability": ml_result["success_probability"] if ml_result else None,
            "peer_success_rate": peer_rate,
            "bull_bear_balance": balance,
            "metric_performance": perf["metric_score"],
        },
        "details": {
            "sector": sector,
            "n_peers_compared": n_peers,
            "metrics_above": perf["above"],
            "metrics_below": perf["below"],
            "comparisons": perf["comparisons"],
            "green_flag_count": len(bull.get("green_flags", [])),
            "red_flag_count": len(bear.get("red_flags", [])),
        },
    }