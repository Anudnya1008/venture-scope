"""
quadrant.py — Stage 9: 2D Risk × Return placement.

VCs think in 2D. "Medium Risk" alone is incomplete — we also need return potential.

  Risk axis   = 100 - ML_success_probability  (higher = riskier)
  Return axis = weighted (market_size, traction, revenue)  (higher = bigger)
"""
import math

QUADRANTS = {
    ("LOW",  "HIGH"): {"name": "Speculative",
                       "color": "#8e44ad",
                       "tagline": "Unicorn lottery — risky but potentially huge"},
    ("LOW",  "LOW"):  {"name": "Avoid",
                       "color": "#dc3545",
                       "tagline": "Low odds AND low upside — pass"},
    ("HIGH", "HIGH"): {"name": "Star",
                       "color": "#28a745",
                       "tagline": "High odds AND high upside — strongest buy case"},
    ("HIGH", "LOW"):  {"name": "Cash Cow",
                       "color": "#f4a261",
                       "tagline": "Safe but small — limited VC fit"},
}


def _return_score(facts):
    feats = facts.get("csv_features") or {}
    tam = float(feats.get("market_size_billion") or 0)
    market = min(tam, 100)

    users = float(feats.get("product_traction_users") or 0)
    traction = min(math.log10(users + 1) * 12, 100) if users > 0 else 30

    rev = float(feats.get("revenue_million") or 0)
    revenue = min(math.log10(rev + 1) * 30, 100) if rev > 0 else 30

    return round(0.4 * market + 0.3 * traction + 0.3 * revenue, 1)


def place(facts, ml_result):
    risk_score = round(100 - ml_result["success_probability"], 1) if ml_result else 50
    return_score = _return_score(facts)
    risk_band = "LOW" if risk_score <= 50 else "HIGH"
    return_band = "HIGH" if return_score >= 50 else "LOW"
    q = QUADRANTS[(risk_band, return_band)]
    return {
        "risk_score": risk_score,
        "return_score": return_score,
        "quadrant": q["name"],
        "color": q["color"],
        "tagline": q["tagline"],
        "explanation": (f"Risk {risk_score}/100 ({risk_band}) × "
                        f"Return {return_score}/100 ({return_band}) → "
                        f"{q['name']}: {q['tagline']}"),
    }
