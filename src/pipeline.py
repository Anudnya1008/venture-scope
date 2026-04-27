import os
import pandas as pd

from main import analyze_deck       # Stage 1
from completeness import check_completeness
from benchmark import run_benchmark
from agents import run_bull, run_bear, run_summarizer
from vitality import compute_vitality
from reconciler import reconcile
from quadrant import place


def run_pipeline(pdf_path: str, csv_path: str, progress=None) -> dict:
    """Run all 10 stages sequentially."""
    def step(msg):
        if progress: progress(msg)
        else: print(msg)

    step("[1/10] Librarian extracting facts...")
    facts = analyze_deck(pdf_path)
    if isinstance(facts, dict) and "error" in facts:
        return {"error": f"Librarian failed: {facts['error']}"}

    # ── Stage 2: Completeness ──
    step(" [2/10] Checking data completeness...")
    completeness = check_completeness(facts)

    # ── Stage 3: Benchmark ──
    step("[3/10] Benchmarking against peers...")
    benchmark = run_benchmark(facts)

    # ── Stages 4 + 5: Bull and Bear ──
    step("[4/10] Bull analyzing...")
    bull = run_bull(facts, benchmark=benchmark)
    step("[5/10] Bear analyzing...")
    bear = run_bear(facts, benchmark=benchmark)

    ml_result = None
    if completeness["can_run_ml"]:
        step("[6/10] ML predicting success probability...")
        try:
            from ml_model import predict_success
            ml_result = predict_success(facts.get("csv_features", {}))
        except FileNotFoundError as e:
            step(f"ML model not trained yet — skipping. ({e})")
    else:
        step("[6/10] ML skipped (insufficient data)")

    step("[7/10] Computing vitality score...")
    df = pd.read_csv(csv_path)
    vitality = compute_vitality(facts, df, bull, bear, ml_result)

    step("[8/10] Reconciling ML vs agent verdicts...")
    reconciliation = reconcile(ml_result, bull, bear)

    step("[9/10] Placing in Risk × Return quadrant...")
    quadrant = place(facts, ml_result)

    step("[10/10] Writing investment memo...")
    verdict = run_summarizer(facts, bull, bear, benchmark=benchmark)

    return {
        "facts": facts,
        "completeness": completeness,
        "benchmark": benchmark,
        "bull": bull,
        "bear": bear,
        "ml": ml_result,
        "vitality": vitality,
        "reconciliation": reconciliation,
        "quadrant": quadrant,
        "verdict": verdict,
    }


def print_report(result: dict):
    """Pretty-print key findings to terminal."""
    if "error" in result:
        print(f"{result['error']}")
        return

    facts, v, q, rec, ver, ml = (
        result["facts"], result["vitality"], result["quadrant"],
        result["reconciliation"], result["verdict"], result["ml"])

    print("\n" + "═" * 70)
    print(f"  INVESTMENT REPORT — {facts.get('startup_name', '?')}")
    print("═" * 70)
    print(f"\n  Industry: {facts.get('industry', '—')}")
    print(f"  Sector:   {facts.get('csv_features', {}).get('sector', '—')}")

    print(f"\n  ┌─ VITALITY SCORE ──────────────────────────────────────")
    print(f"  │  {v['vitality_score']} / 100  →  Risk: {v['risk_level']}")
    print(f"  │  Formula: {v['formula']}")
    print(f"  └───────────────────────────────────────────────────────")

    if ml:
        print(f"\n  ┌─ ML PREDICTION ───────────────────────────────────────")
        print(f"  │  Success probability: {ml['success_probability']}%")
        print(f"  │  Confidence:          {ml['model_confidence']}")
        for d in ml.get("top_drivers", [])[:3]:
            print(f"  │    • {d['feature']} = {d['value']}  ({d['contribution']})")
        print(f"  └───────────────────────────────────────────────────────")
    else:
        print(f"\n  ⏭️  ML skipped — {result['completeness']['explanation']}")

    print(f"\n  ┌─ QUADRANT ────────────────────────────────────────────")
    print(f"  │  {q['quadrant']}  ({q['tagline']})")
    print(f"  │  Risk: {q['risk_score']}/100   Return: {q['return_score']}/100")
    print(f"  └───────────────────────────────────────────────────────")

    print(f"\n  ┌─ RECONCILER ──────────────────────────────────────────")
    print(f"  │  ML: {rec['ml_lean']}  |  Agents: {rec['agent_lean']}  |  → {rec['agreement']}")
    if rec['needs_human_review']:
        print(f"  │  ⚠️  Needs human review")
    print(f"  └───────────────────────────────────────────────────────")

    print(f"\n  📖 REASONING:\n     {v['reasoning']}")
    print(f"\n  📝 MEMO ({ver.get('recommendation', '?')}, "
          f"{ver.get('risk_level', '?')} risk):\n     {ver.get('memo', '')}")
    print("\n" + "═" * 70 + "\n")
