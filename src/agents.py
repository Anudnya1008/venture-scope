import json
from google.genai import types
from config import client, MODEL

def clean_facts_for_agents(facts: dict) -> dict:
    
    if not isinstance(facts, dict):
        return facts
 
    cleaned = dict(facts)  # shallow copy
    raw_features = facts.get("csv_features") or {}
 
    cleaned_features = {}
    for key, value in raw_features.items():
        if isinstance(value, str):
            if value and value.lower() not in ("unknown", "other", "n/a", ""):
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
    all_numeric = ["revenue_million", "revenue_growth_rate", "burn_rate_million",
                   "runway_months", "funding_rounds", "team_size",
                   "founder_experience_years", "has_technical_cofounder",
                   "product_traction_users", "customer_growth_rate",
                   "enterprise_customers", "market_size_billion"]
    cleaned["_features_not_in_deck"] = [
        f for f in all_numeric if f not in cleaned_features
    ]
    return cleaned
 
 
def _call_agent(prompt: str, schema: dict) -> dict:
    """Generic text-in, JSON-out Gemini call."""
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
        return response.parsed
    except Exception as e:
        if "429" in str(e):
            return {"error": "Quota exceeded. Wait ~60 seconds."}
        return {"error": str(e)}

bull_schema = {
    "type": "OBJECT",
    "properties": {
        "green_flags": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["green_flags"],
}


def run_bull(deck_facts: dict, benchmark: dict = None) -> dict:
    print("Bull analyzing...")
    cleaned = clean_facts_for_agents(deck_facts)
    bench_block = ""
    if benchmark:
        bench_block = f"\n\nBENCHMARK CONTEXT:\n{json.dumps(benchmark, indent=2)}"
    prompt = f"""
    You are "The Bull" — an optimistic VC analyst.
    Identify 3-5 GREEN FLAGS. Each: short (4-10 words), specific.
    {f"Prioritize flags backed by ABOVE-benchmark metrics." if benchmark else ""}

    STARTUP FACTS:
    {json.dumps(cleaned, indent=2)}
    {bench_block}

    Return JSON with a "green_flags" array.
    """
    return _call_agent(prompt, bull_schema)


bear_schema = {
    "type": "OBJECT",
    "properties": {
        "red_flags": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["red_flags"],
}


def run_bear(deck_facts: dict, benchmark: dict = None) -> dict:
    print("Bear analyzing...")
    cleaned = clean_facts_for_agents(deck_facts)
    bench_block = ""
    if benchmark:
        bench_block = f"\n\nBENCHMARK CONTEXT:\n{json.dumps(benchmark, indent=2)}"
    prompt = f"""
    You are "The Bear" — a skeptical VC analyst.
    Identify 3-5 RED FLAGS. Look for: high CAC, unproven monetization,
    crowded market, weak moat, team gaps. Each: short (4-10 words), specific.
    {f"Prioritize flags backed by BELOW-benchmark metrics." if benchmark else ""}

    STARTUP FACTS:
    {json.dumps(cleaned, indent=2)}
    {bench_block}

    Return JSON with a "red_flags" array.
    """
    return _call_agent(prompt, bear_schema)


summarizer_schema = {
    "type": "OBJECT",
    "properties": {
        "recommendation": {"type": "STRING"},   # GO | NO-GO | HOLD
        "risk_level":     {"type": "STRING"},   # Low | Medium | High
        "memo":           {"type": "STRING"},   # 2-3 sentences
    },
    "required": ["recommendation", "risk_level", "memo"],
}


def run_summarizer(deck_facts: dict, bull: dict, bear: dict,
                   benchmark: dict = None, vitality: dict = None) -> dict:
    print("Summarizer writing memo...")
    bench_block = ""
    if benchmark:
        bench_block = f"\n\nBENCHMARK:\n{json.dumps(benchmark, indent=2)}"

   
    pinned_risk = vitality.get("risk_level") if vitality else None
    risk_constraint = (
        f'\n\nCONSTRAINT: risk_level MUST be exactly "{pinned_risk}". '
        f'This is the data-driven verdict from the Vitality engine '
        f'(ML + peer benchmark + agent balance). Do not override it. '
        f'Your job is to write the memo and pick GO/NO-GO/HOLD — '
        f'risk_level is fixed.'
        if pinned_risk else ""
    )

    prompt = f"""
    You are a senior VC partner writing the final investment call.

    STARTUP FACTS:
    {json.dumps(deck_facts, indent=2)}

    BULL CASE (green flags):
    {json.dumps(bull, indent=2)}

    BEAR CASE (red flags):
    {json.dumps(bear, indent=2)}
    {bench_block}
    {risk_constraint}

    Return JSON with:
      - recommendation: "GO" | "NO-GO" | "HOLD"
      - risk_level:     "Low" | "Medium" | "High"
      - memo:           2-3 sentences naming the key caveat
    """
    result = _call_agent(prompt, summarizer_schema)

    if pinned_risk and isinstance(result, dict) and "risk_level" in result:
        result["risk_level"] = pinned_risk

    return result