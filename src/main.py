import os
import json
from dotenv import load_dotenv
from google.genai import types
from config import client, MODEL

load_dotenv()

HERE = os.path.dirname(os.path.abspath(__file__))          
ROOT = os.path.dirname(HERE)                               
DEFAULT_PDF = os.path.join(ROOT, "pdfs", "health",
                           "Alan-Pitch-Deck-series-d-1.pdf")
CSV_PATH = os.path.join(ROOT, "dataset", "startup_success_dataset.csv")


output_schema = {
    "type": "OBJECT",
    "properties": {
        "startup_name":   {"type": "STRING"},
        "industry":       {"type": "STRING"},
        "problem":        {"type": "STRING"},
        "solution":       {"type": "STRING"},
        "funding_amount": {"type": "STRING"},
        "market_size":    {"type": "STRING"},
        "business_model": {"type": "STRING"},
        "key_metrics":    {"type": "ARRAY", "items": {"type": "STRING"}},
        "team_score":     {"type": "INTEGER"},

        "csv_features": {
            "type": "OBJECT",
            "properties": {
                "revenue_million":            {"type": "NUMBER"},
                "revenue_growth_rate":        {"type": "NUMBER"},
                "burn_rate_million":          {"type": "NUMBER"},
                "runway_months":              {"type": "NUMBER"},
                "funding_rounds":             {"type": "NUMBER"},
                "team_size":                  {"type": "NUMBER"},
                "founder_experience_years":   {"type": "NUMBER"},
                "has_technical_cofounder":    {"type": "NUMBER"},  # 0 or 1
                "product_traction_users":     {"type": "NUMBER"},
                "customer_growth_rate":       {"type": "NUMBER"},
                "enterprise_customers":       {"type": "NUMBER"},
                "market_size_billion":        {"type": "NUMBER"},
                "sector":                     {"type": "STRING"},
                "business_model":             {"type": "STRING"},
                "geography":                  {"type": "STRING"},
            },
        },
    },
    "required": ["startup_name", "industry", "problem", "solution"],
}


def analyze_deck(pdf_path: str) -> dict:
    """The Librarian — extracts 15 ML features + qualitative facts from PDF."""
    print(f"🗂️  Librarian analyzing {os.path.basename(pdf_path)}...")

    with open(pdf_path, "rb") as f:
        pdf_content = f.read()

    prompt = """
    Analyze this startup pitch deck and extract structured facts.

    For csv_features, map to these enums:
      • sector:         SaaS | Health | Fintech | Ecommerce | Climate | Crypto | AI | Other
      • business_model: Subscription | Marketplace | Transaction-fee | Hardware | Ad-based
      • geography:      North America | Europe | Asia | Middle East | Oceania | Other
      • has_technical_cofounder: 1 if yes, 0 if no/unknown

    For numeric fields: use 0 if the deck does not state the value — do NOT invent.
    revenue_growth_rate and customer_growth_rate are percentages (e.g., 50 = 50% YoY).
    runway_months is total months of runway.

    team_score: integer 0-100 based on founder pedigree as described.

    Return JSON.
    """

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=[
                types.Part.from_bytes(data=pdf_content,
                                      mime_type="application/pdf"),
                prompt
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=output_schema)
        )
        return response.parsed
    except Exception as e:
        if "429" in str(e):
            return {"error": "Quota exceeded. Wait ~60 seconds and try again."}
        return {"error": str(e)}


if __name__ == "__main__":
    from pipeline import run_pipeline, print_report

    result = run_pipeline(DEFAULT_PDF, CSV_PATH)
    print_report(result)

    with open(os.path.join(ROOT, "report.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nFull report saved to {os.path.join(ROOT, 'report.json')}")
