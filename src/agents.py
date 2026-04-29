import json
from google.genai import types
from config import client, MODEL


NUMERIC_FEATURES = [
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

PLACEHOLDER_STRINGS = {"unknown", "other", "n/a", ""}


def clean_facts_for_agents(facts):
    if not isinstance(facts, dict):
        return facts

    cleaned = dict(facts)
    raw_features = facts.get("csv_features") or {}
    cleaned_features = {}

    for key, value in raw_features.items():
        if isinstance(value, str):
            if value.lower() not in PLACEHOLDER_STRINGS:
                cleaned_features[key] = value
            continue
        if value is None:
            continue
        try:
            if float(value) == 0:
                continue
        except (TypeError, ValueError):
            continue
        cleaned_features[key] = value

    cleaned["csv_features"] = cleaned_features
    cleaned["_features_not_in_deck"] = [
        f for f in NUMERIC_FEATURES if f not in cleaned_features
    ]
    return cleaned


def call_gemini(prompt, schema):
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
                temperature=0,
            ),
        )
        return response.parsed
    except Exception as e:
        if "429" in str(e):
            return {"error": "Quota exceeded. Wait ~60 seconds."}
        return {"error": str(e)}


def benchmark_block(benchmark):
    if not benchmark:
        return ""
    return f"\n\nBENCHMARK CONTEXT:\n{json.dumps(benchmark, indent=2)}"


bull_schema = {
    "type": "OBJECT",
    "properties": {
        "green_flags": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["green_flags"],
}


def run_bull(deck_facts, benchmark=None):
    cleaned = clean_facts_for_agents(deck_facts)
    bench_hint = "Prioritize flags backed by ABOVE-benchmark metrics." if benchmark else ""

    prompt = f"""
    You are "The Bull", an optimistic VC analyst.
    Identify 3 GREEN FLAGS. Each one should be short (4-10 words) and specific.
    {bench_hint}

    STARTUP FACTS:
    {json.dumps(cleaned, indent=2)}
    {benchmark_block(benchmark)}

    Return JSON with a "green_flags" array.
    """
    return call_gemini(prompt, bull_schema)


bear_schema = {
    "type": "OBJECT",
    "properties": {
        "red_flags": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["red_flags"],
}


def run_bear(deck_facts, benchmark=None):
    cleaned = clean_facts_for_agents(deck_facts)
    bench_hint = "Prioritize flags backed by BELOW-benchmark metrics." if benchmark else ""

    prompt = f"""
    You are "The Bear", a skeptical VC analyst.
    Identify 3 RED FLAGS. Look for high CAC, unproven monetization,
    crowded market, weak moat, team gaps. Each one should be short (4-10 words) and specific.
    {bench_hint}

    STARTUP FACTS:
    {json.dumps(cleaned, indent=2)}
    {benchmark_block(benchmark)}

    Return JSON with a "red_flags" array.
    """
    return call_gemini(prompt, bear_schema)


summarizer_schema = {
    "type": "OBJECT",
    "properties": {
        "recommendation": {"type": "STRING"},
        "risk_level":     {"type": "STRING"},
        "memo":           {"type": "STRING"},
    },
    "required": ["recommendation", "risk_level", "memo"],
}

def run_summarizer(deck_facts, bull, bear, benchmark=None, vitality=None, ml=None):
    pinned_risk = vitality.get("risk_level") if vitality else None

    data_block = ""
    if vitality:
        data_block += f"\n\nVITALITY SCORE: {vitality['vitality_score']}/100"
        data_block += f"\nRISK LEVEL: {vitality['risk_level']}"
    if ml:
        data_block += f"\nML SUCCESS PROBABILITY: {ml['success_probability']}%"
        data_block += f"\nML CONFIDENCE: {ml['model_confidence']}"
        if ml['model_confidence'] == "LOW":
            data_block += ("\nNOTE: Low ML confidence means the prediction is close "
                          "to the decision boundary. Consider HOLD over GO unless "
                          "qualitative signals are very strong.")

    risk_constraint = ""
    if pinned_risk:
        risk_constraint = (
            f'\n\nCONSTRAINT: risk_level MUST be exactly "{pinned_risk}". '
            f'This is the data-driven verdict from the Vitality engine '
            f'(ML + peer benchmark + agent balance). Do not override it. '
            f'Your job is to write the memo and pick GO/NO-GO/HOLD, '
            f'risk_level is fixed.'
        )

    prompt = f"""
    You are a senior VC partner writing the final investment call.

    STARTUP FACTS:
    {json.dumps(deck_facts, indent=2)}

    BULL CASE (green flags):
    {json.dumps(bull, indent=2)}

    BEAR CASE (red flags):
    {json.dumps(bear, indent=2)}
    {benchmark_block(benchmark)}
    {data_block}
    {risk_constraint}

    Return JSON with:
      - recommendation: "GO" | "NO-GO" | "HOLD"
      - risk_level:     "Low" | "Medium" | "High"
      - memo:           2-3 sentences naming the key caveat
    """
    result = call_gemini(prompt, summarizer_schema)

    if pinned_risk and isinstance(result, dict) and "risk_level" in result:
        result["risk_level"] = pinned_risk

    if (ml and ml.get("model_confidence") == "LOW"
        and isinstance(result, dict)
        and result.get("recommendation") == "GO"):
        result["recommendation"] = "HOLD"
        result["memo"] = (
            f"Recommended HOLD due to low ML confidence. The success probability of "
            f"{ml['success_probability']}% is near the decision boundary, indicating "
            f"the prediction could fall either way. While qualitative signals are "
            f"positive, the quantitative model lacks strong conviction. Recommend "
            f"holding pending further due diligence on missing financial details."
        )

    return result
   