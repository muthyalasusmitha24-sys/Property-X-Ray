"""
app.py  --  Property X-Ray : Wellness Score  (Streamlit)
Run:  streamlit run app.py
"""
import io
import os
 
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
 
import theme as T
import wellness_engine as we
 
st.set_page_config(page_title="Property X-Ray – Wellness Score", page_icon="🏠", layout="wide")
T.inject()
T.hero("Wellness Score",
       "A transparent, statistically validated score for how well a property supports everyday living, "
       "from raw parameters to a final score with a confidence interval.",
       ["Preprocess", "Normalise", "Correlate", "Factor analysis", "Derive weights",
        "Bootstrap check", "Sub-factor scores", "Factor scores", "Overall score + interval"])
 
 
# ------------------------------------------------------------------ helpers
def get_token():
    try:
        t = st.secrets.get("GROQ_API_KEY")
    except Exception:
        t = None
    return t or os.environ.get("GROQ_API_KEY") or st.session_state.get("groq_key_input") or ""
 
 
@st.cache_data(show_spinner=False)
def cached_sample(n, seed):
    return we.generate_sample_data(n, seed)
 
 
@st.cache_data(show_spinner=False)
def cached_run(df, cfg, alpha, B, seed):
    return we.run_all(df, cfg, alpha, B, seed)
 
 
def read_upload(file):
    if file.name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    return pd.read_csv(file)
 
 
def short(c):
    return c.split(": ", 1)[1] if ": " in c else c
 
 
def to_csv(df):
    return df.to_csv().encode("utf-8")
 
 
# ------------------------------------------------------------------ sidebar
with st.sidebar:
    T.sidebar_brand()
    st.subheader("Settings")
    src = st.radio("Data source", ["Synthetic sample", "Upload file"])
    if src == "Synthetic sample":
        n_rows = st.number_input("Rows", 30, 1000, 100, 10)
        seed_data = st.number_input("Data seed", 0, 9999, 42)
    alpha = st.slider("Weight blend  (0 = entropy only · 1 = factor loadings only)", 0.0, 1.0, 0.5, 0.05)
    B = st.slider("Bootstrap resamples", 20, 400, 100, 20,
                  help="More = more stable weights / CI, but slower (cached after first run).")
    level = st.selectbox("Confidence level", [0.90, 0.95, 0.99], index=1, format_func=lambda v: f"{v:.0%}")
    st.divider()
    st.subheader("Buyer preference profile")
    profile = st.selectbox(
        "Overall factor-weighting method",
        ["Balanced (equal weights)", "Data-driven (explained variance)", "Custom"],
        help="Balanced gives each available wellness factor equal weight. Data-driven derives weights from PCA of factor scores. "
             "Custom lets you set relative importance explicitly; it is a preference choice, not a statistically discovered truth."
    )
    custom_w = {}
    if profile == "Custom":
        st.caption("Set relative importance from 0 to 3. Values are normalised to sum to 100%; 0 excludes a factor.")
        for code, f in we.HIERARCHY.items():
            custom_w[code] = st.slider(f["name"], 0.0, 3.0, 1.0, 0.1, key=f"custom_weight_{code}")
        if sum(custom_w.values()) <= 0:
            st.warning("Set at least one factor weight above 0. The score cannot be calculated if all custom weights are zero.")
    st.divider()
    st.subheader("LLM explanation (Groq)")
    if not (get_token() and not st.session_state.get("groq_key_input")):
        st.text_input("Groq API key (kept only in this session)", type="password", key="groq_key_input",
                      help="Better: put GROQ_API_KEY in Streamlit Secrets or an environment variable.")
    hf_model = st.text_input("Model", "llama-3.3-70b-versatile")
 
tabs = st.tabs(["Data & preprocessing", "Factor analysis & weights", "Overall score", "Score a new property", "Explanation", "Method & worked example"])
 
# ------------------------------------------------------------------ TAB 0: data
cfg_base = we.default_config()
