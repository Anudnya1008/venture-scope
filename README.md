# Venture Scope

VC startup evaluation tool: PDF in → multi-agent analysis + ML prediction → investment recommendation.

## Setup

### 1. Install dependencies
bash
pip install -r requirements.txt

### 2. Add your Gemini API key
Create .env in the project root:
GEMINI_API_KEY=your_key_here

Get a free key at https://aistudio.google.com/apikey.

### 3. (Optional) Build dataset from Crunchbase
Skip this step if you already have `dataset/startup_success_dataset.csv` (synthetic version).

Download from https://www.kaggle.com/datasets/justinas/startup-investments and place these three files in `dataset/crunchbase_raw/`:
- `objects.csv`
- `funding_rounds.csv`
- `investments.csv`

Then build:
bash
python scripts/build_dataset.py


### 4. Train the ML model
bash
python scripts/train_model.py

~1 minute. Produces `models/success_model.pkl`.

### 5. Run

**CLI:**
bash
python src/main.py


**Streamlit UI:**
bash
streamlit run src/app.py

1. PDF parsing — Gemini reads the deck and extracts structured facts
2. Peer benchmarking — compares the startup to similar companies in the dataset
3. Bull and Bear agents — two LLMs argue opposite sides
4. ML prediction — LightGBM estimates success probability
5. Summarizer — combines everything into a final GO / NO-GO / HOLD memo


