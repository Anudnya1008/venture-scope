# Venture Scope

VC startup evaluation tool: PDF in → multi-agent analysis + ML prediction → investment recommendation.

## Project structure

```
venture-scope/
├── pdfs/                              # your test pitch decks
├── dataset/
│   ├── startup_success_dataset.csv    # training data (synthetic OR Crunchbase-derived)
│   └── crunchbase_raw/                # (optional) raw Crunchbase CSVs
│       ├── objects.csv
│       ├── funding_rounds.csv
│       └── investments.csv
├── models/
│   └── success_model.pkl              # produced by train_model.py
├── scripts/
│   ├── build_dataset.py               # one-time: builds 15-feature CSV
│   └── train_model.py                 # one-time: trains LightGBM
├── src/
│   ├── config.py                      # shared Gemini client
│   ├── main.py                        # Stage 1: Librarian (analyze_deck)
│   ├── completeness.py                # Stage 2: data quality guardrail
│   ├── benchmark.py                   # Stage 3: peer comparison
│   ├── agents.py                      # Stages 4, 5, 10: Bull, Bear, Summarizer
│   ├── ml_model.py                    # Stage 6: LightGBM predictor
│   ├── vitality.py                    # Stage 7: combined score
│   ├── reconciler.py                  # Stage 8: ML vs agent agreement
│   ├── quadrant.py                    # Stage 9: 2D Risk × Return
│   ├── pipeline.py                    # orchestrator (calls all 10 stages)
│   └── app.py                         # Streamlit UI
├── requirements.txt
└── .env                               # GEMINI_API_KEY=...
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your Gemini API key
Create `.env` in the project root:
```
GEMINI_API_KEY=your_key_here
```
Get a free key at https://aistudio.google.com/apikey.

### 3. (Optional) Build dataset from Crunchbase
Skip this step if you already have `dataset/startup_success_dataset.csv` (synthetic version).

Download from https://www.kaggle.com/datasets/justinas/startup-investments and place these three files in `dataset/crunchbase_raw/`:
- `objects.csv`
- `funding_rounds.csv`
- `investments.csv`

Then build:
```bash
python scripts/build_dataset.py
```

### 4. Train the ML model
```bash
python scripts/train_model.py
```
~1 minute. Produces `models/success_model.pkl`.

### 5. Run

**CLI:**
```bash
python src/main.py
```

**Streamlit UI:**
```bash
streamlit run src/app.py
```

## The 10-stage pipeline

```
USER UPLOADS PDF
   │
   ├── 1. Librarian (main.analyze_deck)         — extract facts from PDF
   ├── 2. Completeness check                    — can ML run reliably?
   ├── 3. Benchmark (LLM + pandas)              — compare to peers
   ├── 4. Bull (LLM)                            — optimist case
   ├── 5. Bear (LLM)                            — skeptic case
   ├── 6. ML model (LightGBM)                   — success probability
   ├── 7. Vitality score                        — combined 0-100
   ├── 8. Reconciler                            — ML vs agent agreement
   ├── 9. Quadrant                              — 2D Risk × Return
   └── 10. Summarizer (LLM)                     — final memo
   │
   ▼
RESULTS RENDERED
```

## What's real vs. synthetic in the dataset

| Feature | Source |
|---|---|
| funding_rounds | ✅ Real (Crunchbase) |
| sector | ✅ Real (Crunchbase) |
| geography | ✅ Real (Crunchbase) |
| total_raised (used for derivation) | ✅ Real (Crunchbase) |
| revenue_million | ⚙️ Derived (raised × sector multiple) |
| revenue_growth_rate | ⚙️ Derived (funding velocity) |
| burn_rate_million | ⚙️ Derived (raised ÷ age) |
| runway_months | ⚙️ Derived (last round × 0.6 / monthly burn) |
| business_model | ⚙️ Derived (from sector) |
| team_size | ✅ Real if available, else 🎲 synthetic |
| founder_experience_years | 🎲 Synthetic |
| has_technical_cofounder | 🎲 Synthetic (sector-weighted) |
| product_traction_users | 🎲 Synthetic |
| customer_growth_rate | 🎲 Synthetic |
| enterprise_customers | 🎲 Synthetic |
| market_size_billion | 🎲 Per-sector default |
