def _agent_lean(bull, bear):
    g = len(bull.get("green_flags", []))
    r = len(bear.get("red_flags", []))
    if g + r == 0: return "NEUTRAL"
    share = g / (g + r)
    if share >= 0.6: return "BULLISH"
    if share <= 0.4: return "BEARISH"
    return "NEUTRAL"


def _ml_lean(prob):
    if prob >= 60: return "BULLISH"
    if prob <= 40: return "BEARISH"
    return "NEUTRAL"


def reconcile(ml_result, bull, bear):
    agent = _agent_lean(bull, bear)

    if ml_result is None:
        return {
            "ml_lean": "N/A", "agent_lean": agent,
            "agreement": "N/A", "needs_human_review": True,
            "explanation": "ML skipped due to insufficient data — relying on agents only.",
        }

    ml = _ml_lean(ml_result["success_probability"])

    if ml == agent:
        return {
            "ml_lean": ml, "agent_lean": agent, "agreement": "AGREE",
            "needs_human_review": False,
            "explanation": f"Both lean {ml.lower()} — high confidence.",
        }

    if "NEUTRAL" in (ml, agent):
        non_neutral = ml if ml != "NEUTRAL" else agent
        return {
            "ml_lean": ml, "agent_lean": agent, "agreement": "PARTIAL",
            "needs_human_review": False,
            "explanation": f"One source neutral, other leans {non_neutral.lower()}.",
        }

    g = len(bull.get("green_flags", []))
    r = len(bear.get("red_flags", []))
    return {
        "ml_lean": ml, "agent_lean": agent, "agreement": "DISAGREE",
        "needs_human_review": True,
        "explanation": (
            f"DISAGREEMENT: ML predicts {ml_result['success_probability']:.0f}% "
            f"({ml.lower()}), agents raised {g} green vs {r} red flags ({agent.lower()}). "
            f"Recommend human review."
        ),
    }
