"""
Financial Sentiment Analyzer — Streamlit Dashboard

Requires the FastAPI server to be running:
    uvicorn app.main:app --port 8000

Then run:
    streamlit run dashboard/streamlit_app.py
"""

import os

import httpx
import streamlit as st

# Configurable via env var so docker-compose can point dashboard → api container
API_BASE = os.getenv("API_BASE", "http://localhost:8000/api/v1")

SENTIMENT_COLORS = {
    "positive": "#28a745",
    "negative": "#dc3545",
    "neutral":  "#6c757d",
}

SENTIMENT_EMOJI = {
    "positive": "📈",
    "negative": "📉",
    "neutral":  "➡️",
}


def analyze(headline: str) -> dict:
    r = httpx.post(f"{API_BASE}/analyze", json={"headline": headline}, timeout=30)
    r.raise_for_status()
    return r.json()


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Sentiment Analyzer",
    page_icon="📊",
    layout="centered",
)

st.title("📊 Financial Sentiment Analyzer")
st.caption("Enter a financial news headline to get its sentiment.")

st.divider()

# ── Input ─────────────────────────────────────────────────────────────────────
headline_input = st.text_input(
    label="Headline",
    placeholder="e.g. Apple beats Q3 earnings expectations by 12%",
    label_visibility="collapsed",
)

analyze_btn = st.button("Analyze", type="primary", use_container_width=True)

# ── Result ────────────────────────────────────────────────────────────────────
if analyze_btn:
    if not headline_input.strip():
        st.warning("Please enter a headline.")
    else:
        with st.spinner("Analyzing..."):
            try:
                result = analyze(headline_input.strip())
                sentiment = result["sentiment"]
                color = SENTIMENT_COLORS[sentiment]
                emoji = SENTIMENT_EMOJI[sentiment]

                st.divider()

                # Big sentiment display
                st.markdown(
                    f"""
                    <div style="
                        text-align: center;
                        background-color: {color}18;
                        border: 2px solid {color};
                        border-radius: 12px;
                        padding: 32px 20px;
                        margin: 8px 0 20px 0;
                    ">
                        <div style="font-size: 52px; line-height: 1;">{emoji}</div>
                        <div style="font-size: 36px; font-weight: 700; color: {color}; margin-top: 8px;">
                            {sentiment.upper()}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Get More Info expander
                with st.expander("Get More Info"):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Positive", f"{result['scores']['positive']:.2%}")
                    col2.metric("Negative", f"{result['scores']['negative']:.2%}")
                    col3.metric("Neutral",  f"{result['scores']['neutral']:.2%}")

                    st.markdown("**Confidence:** `{:.2%}`".format(result["confidence"]))
                    st.markdown("**Model:** `{}`".format("VADER (vaderSentiment)"))

                    st.markdown("**Full JSON response:**")
                    st.json(result)

            except httpx.ConnectError:
                st.error("Cannot reach the API. Make sure the server is running:\n```\nuvicorn app.main:app --port 8000\n```")
            except Exception as e:
                st.error(f"Error: {e}")
