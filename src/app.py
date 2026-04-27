import os
import json
import tempfile
import streamlit as st
from pipeline import run_pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CSV_PATH = os.path.join(ROOT, "dataset", "startup_success_dataset.csv")

st.set_page_config(page_title="Venture Scope", page_icon="📊", layout="wide")

st.markdown("""
<style>
.metric-card{padding:32px 24px;border-radius:12px;text-align:center;color:white;}
.metric-value{font-size:28px;font-weight:700;}   /* was 24px */.metric-label{font-size:14px;opacity:.9;margin-bottom:8px;}
.metric-value{font-size:24px;font-weight:700;}
.green-card{background:linear-gradient(135deg,#2d6a4f,#40916c);}
.blue-card{background:linear-gradient(135deg,#1d4e89,#2980b9);}
.yellow-card{background:linear-gradient(135deg,#d68c2a,#f4a261);}
.purple-card{background:linear-gradient(135deg,#5b2a86,#8e44ad);}
.bull-card{background:#d4edda;border-left:4px solid #28a745;padding:18px;border-radius:10px;color:#155724;}
.bear-card{background:#f8d7da;border-left:4px solid #dc3545;padding:18px;border-radius:10px;color:#721c24;}
.bench-card{background:#e2e3f3;border-left:4px solid #4b5fbd;padding:18px;border-radius:10px;color:#1a1f4b;}
.debate-title{font-weight:700;font-size:16px;margin-bottom:10px;}
.rec-box{background:#1e222a;padding:24px;border-radius:12px;color:white;margin-top:16px;}
.go-badge{background:#28a745;color:#fff;padding:10px 36px;border-radius:6px;font-weight:700;display:inline-block;}
.no-go-badge{background:#dc3545;color:#fff;padding:10px 36px;border-radius:6px;font-weight:700;display:inline-block;}
.hold-badge{background:#f4a261;color:#fff;padding:10px 36px;border-radius:6px;font-weight:700;display:inline-block;}
.disagree{background:#fff3cd;border-left:4px solid #ffc107;padding:14px;border-radius:8px;color:#856404;}
.agree{background:#d1ecf1;border-left:4px solid #0dcaf0;padding:14px;border-radius:8px;color:#0c5460;}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("📊 Venture Scope")
    st.caption("Multi-agent + ML startup evaluation")
    uploaded_pdf = st.file_uploader("Upload Pitch Deck (PDF)", type=["pdf"])
    run_btn = st.button("🚀 Analyze Deck", type="primary", use_container_width=True)

st.markdown("## AI Startup Evaluation")

if "results" not in st.session_state:
    st.session_state.results = None

if run_btn:
    if not uploaded_pdf:
        st.error("Please upload a pitch deck PDF.")
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_pdf.read())
            tmp_path = tmp.name
        try:
            status = st.empty()
            st.session_state.results = run_pipeline(
                tmp_path, CSV_PATH,
                progress=lambda msg: status.info(msg))
            status.empty()
        except Exception as e:
            st.error(f"Pipeline failed: {e}")

results = st.session_state.results
if results and "error" not in results:
    facts = results["facts"]
    bull, bear = results["bull"], results["bear"]
    verdict, vit = results["verdict"], results["vitality"]
    ml, rec = results["ml"], results["reconciliation"]

    st.markdown(f"### {facts.get('startup_name', '?')}")
    st.caption(f"**Industry:** {facts.get('industry', '—')}   •   "
               f"**Business model:** {facts.get('business_model', '—')}   •   "
               f"**Sector:** {facts.get('csv_features', {}).get('sector', '—')}")

    # ── 4 metric cards ──
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"""<div class="metric-card green-card">
        <div class="metric-label">Vitality Score</div>
        <div class="metric-value">{vit['vitality_score']}/100</div></div>""",
        unsafe_allow_html=True)
    c2.markdown(f"""<div class="metric-card blue-card">
        <div class="metric-label">Risk Level</div>
        <div class="metric-value">{vit['risk_level']}</div></div>""",
        unsafe_allow_html=True)
    ml_disp = f"{ml['success_probability']}%" if ml else "N/A"
    c3.markdown(f"""<div class="metric-card purple-card">
        <div class="metric-label">ML Success Prob.</div>
        <div class="metric-value">{ml_disp}</div></div>""",
        unsafe_allow_html=True)
   

    with st.expander("📄 About This Startup", expanded=True):
        problem = facts.get("problem", "")
        solution = facts.get("solution", "")
        market = facts.get("market_size", "")
        funding = facts.get("funding_amount", "")

        if problem:
            st.markdown(f"**Problem**  \n{problem}")
        if solution:
            st.markdown(f"**Solution**  \n{solution}")
        if market:
            st.markdown(f"**Market Size**  \n{market}")
        if funding:
            st.markdown(f"**Funding**  \n{funding}")

    st.markdown("### 🧠 Reasoning")
    if rec["agreement"] == "DISAGREE":
        st.markdown(f"<div class='disagree'>{rec['explanation']}</div>",
                    unsafe_allow_html=True)
    elif rec["agreement"] == "AGREE":
        st.markdown(f"<div class='agree'>✓ {rec['explanation']}</div>",
                    unsafe_allow_html=True)

    if ml:
        with st.expander("🤖 ML Model Details"):
            st.markdown(f"**Confidence:** {ml['model_confidence']}")
            st.markdown("**Top drivers:**")
            for d in ml.get("top_drivers", []):
                st.markdown(f"- {d['feature']} = `{d['value']}` ({d['contribution']})")
        key_metrics = facts.get("key_metrics", [])
        if key_metrics:
            with st.expander("📊 Key Metrics", expanded=False):
                for metric in key_metrics:
                    st.markdown(f"- {metric}")
    st.markdown("### Agent Debate: Bull vs. Bear")
    L, R = st.columns(2)
    bull_items = "".join(f"<li>{f}</li>" for f in bull.get("green_flags", []))
    bear_items = "".join(f"<li>{f}</li>" for f in bear.get("red_flags", []))
    L.markdown(f"""<div class="bull-card"><div class="debate-title">🐂 Bull (Optimist)</div>
        <ul>{bull_items}</ul></div>""", unsafe_allow_html=True)
    R.markdown(f"""<div class="bear-card"><div class="debate-title">🐻 Bear (Skeptic)</div>
        <ul>{bear_items}</ul></div>""", unsafe_allow_html=True)

    rec_label = verdict.get("recommendation", "HOLD").upper()
    badge = {"GO": "go-badge", "NO-GO": "no-go-badge"}.get(rec_label, "hold-badge")
    st.markdown(f"""<div class="rec-box">
        <div style="font-size:18px;font-weight:700;margin-bottom:12px;">Final Investment Recommendation</div>
        <div style="text-align:center;margin:16px 0;">
            <span class="{badge}">{rec_label}</span></div>
        <p style="opacity:.9;margin-top:12px;">{verdict.get('memo', '')}</p>
    </div>""", unsafe_allow_html=True)

    with st.expander("📋 Raw agent outputs"):
        st.json(results)
elif results and "error" in results:
    st.error(results["error"])
else:
    st.info("👈 Upload a pitch deck and click **Analyze Deck**.")
