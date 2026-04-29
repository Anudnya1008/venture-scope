import pandas as pd

from main import analyze_deck
from benchmark import run_benchmark
from agents import run_bull, run_bear, run_summarizer
from vitality import compute_vitality
import vitality

KEY_FEATURES = [
    "revenue_million", "revenue_growth_rate", "funding_rounds",
    "founder_experience_years", "product_traction_users",
]


def run_pipeline(pdf_path, csv_path, progress=None):
    def step(msg):
        if progress:
            progress(msg)

    step("Extracting facts from deck...")
    facts = analyze_deck(pdf_path)
    if isinstance(facts, dict) and "error" in facts:
        return {"error": facts["error"]}

    csv_features = facts.get("csv_features") or {}
    can_run_ml = sum(1 for f in KEY_FEATURES if csv_features.get(f, 0) > 0) >= 2

    step("Benchmarking against peers...")
    benchmark = run_benchmark(facts)

    step("Running bull and bear agents...")
    bull = run_bull(facts, benchmark=benchmark)
    bear = run_bear(facts, benchmark=benchmark)

    ml_result = None
    if can_run_ml:
        step("Running ML prediction...")
        try:
            from ml_model import predict_success
            ml_result = predict_success(csv_features)
        except FileNotFoundError:
            pass

    step("Computing vitality score...")
    df = pd.read_csv(csv_path)
    vitality = compute_vitality(facts, df, bull, bear, ml_result)

    step("Writing investment memo...")
    verdict = run_summarizer(facts, bull, bear, benchmark=benchmark,vitality=vitality, ml=ml_result)
    return {
        "facts": facts,
        "benchmark": benchmark,
        "bull": bull,
        "bear": bear,
        "ml": ml_result,
        "vitality": vitality,
        "verdict": verdict,
    }