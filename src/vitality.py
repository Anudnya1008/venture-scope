
import pandas as pd

METRIC_COLS = [
    "revenue_million", "revenue_growth_rate", "burn_rate_million",
    "runway_months", "funding_rounds", "team_size",
    "founder_experience_years", "has_technical_cofounder",
    "product_traction_users", "customer_growth_rate",
    "enterprise_customers", "market_size_billion",
]
LOWER_IS_BETTER = {"burn_rate_million"}


def peer_success_rate(df, sector):
    peers = df[df["sector"].str.lower() == str(sector).lower()]
    if len(peers) == 0:
        return 50.0, 0
    return round(100 * peers["outcome"].isin(["IPO","Acquisition"]).mean(), 1), len(peers)


def metric_performance(df, sector, startup_features):
    peers = df[df["sector"].str.lower() == str(sector).lower()]
    if len(peers) == 0:
        return {"metric_score": 50.0, "above": 0, "below": 0, "comparisons": []}
    comparisons, above, below = [], 0, 0
    for col in METRIC_COLS:
        if col not in peers.columns: continue
        v = startup_features.get(col)
        if v is None or v == 0: continue
        med = float(peers[col].median())
        if med == 0: continue
        ratio = v / med
        if 0.9 <= ratio <= 1.1:
            verdict = "AT"
        else:
            is_above = ratio > 1.1
            if col in LOWER_IS_BETTER: is_above = not is_above
            verdict = "ABOVE" if is_above else "BELOW"
        comparisons.append({"metric": col, "startup_value": v,
                            "peer_median": round(med, 2), "verdict": verdict})
        if verdict == "ABOVE": above += 1
        elif verdict == "BELOW": below += 1
    total = above + below
    return {"metric_score": round((above/total*100) if total else 50, 1),
            "above": above, "below": below, "comparisons": comparisons}


def _bull_bear(bull, bear):
    g, r = len(bull.get("green_flags", [])), len(bear.get("red_flags", []))
    return round(100 * g / (g + r), 1) if (g + r) > 0 else 50.0


def _risk(score):
    if score >= 75: return "Low"
    if score >= 50: return "Medium"
    return "High"


def _reasoning(score, risk, peer_rate, n_peers, sector, perf, bull, bear, ml):
    g, r = len(bull.get("green_flags", [])), len(bear.get("red_flags", []))

    if ml:
        s1 = (f"ML model estimates {ml['success_probability']}% success "
              f"probability based on {sector} sector patterns.")
    elif n_peers == 0:
        s1 = f"No peers in dataset for sector '{sector}'."
    else:
        s1 = f"Of {n_peers} {sector} peers, {peer_rate}% reached IPO or acquisition."

    above, below = perf["above"], perf["below"]
    total = above + below
    if total == 0:
        s2 = "Insufficient deck data for peer benchmarking."
    else:
        comps = perf["comparisons"]
        notable = (next((c for c in comps if c["verdict"] == "ABOVE"), None)
                if above >= below
                else next((c for c in comps if c["verdict"] == "BELOW"), None))
        s2 = f"Beats peer median on {above}/{total} metrics"
        if notable:
            s2 += (f", notably {notable['metric'].replace('_', ' ')} "
                f"({notable['startup_value']} vs {notable['peer_median']})")
        else:
            s2 += "."

    if g > r:
        s3 = f"Agent debate positive: {g} green vs {r} red flags."
    elif r > g:
        s3 = f"Agent debate cautious: {r} red vs {g} green flags."
    else:
        s3 = f"Agent debate balanced: {g} green, {r} red."

    risk_phrase = {
        "Low": "LOW RISK — both data and analysis favorable.",
        "Medium": "MEDIUM RISK — real upside but caveats need investigation.",
        "High": "HIGH RISK — data does not support a confident bet.",
    }[risk]
    s4 = f"Score {score}/100 → {risk_phrase}"
    return f"{s1} {s2} {s3} {s4}"


def compute_vitality(facts, df, bull, bear, ml_result=None):
    csv_features = facts.get("csv_features") or {}
    sector = csv_features.get("sector", "Unknown")
    peer_rate, n_peers = peer_success_rate(df, sector)
    perf = metric_performance(df, sector, csv_features)
    bb = _bull_bear(bull, bear)

    if ml_result is not None:
        vitality = 0.4 * ml_result["success_probability"] + 0.3 * peer_rate + 0.3 * bb
        formula = "40% ML + 30% peer rate + 30% agent balance"
    else:
        vitality = 0.5 * peer_rate + 0.5 * bb
        formula = "50% peer rate + 50% agent balance (ML skipped)"

    vitality = round(vitality, 1)
    risk = _risk(vitality)

    return {
        "vitality_score": vitality,
        "risk_level": risk,
        "reasoning": _reasoning(vitality, risk, peer_rate, n_peers, sector,
                                perf, bull, bear, ml_result),
        "formula": formula,
        "breakdown": {
            "ml_success_probability": ml_result["success_probability"] if ml_result else None,
            "peer_success_rate": peer_rate,
            "bull_bear_balance": bb,
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
