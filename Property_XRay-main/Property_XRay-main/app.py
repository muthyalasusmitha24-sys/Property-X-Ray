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
       ["Data preprocessing", "Normalise", "Correlate", "Factor analysis", "Entropy weights", "Bootstrap check", "Parameter weights",
        "Sub-factor scores", "Factor scores", "Wellness Score + CI"])
 
 
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
    st.subheader("Overall factor weighting")
    profile = st.selectbox(
        "Weighting method",
        ["Balanced (equal weights)", "Data-driven (explained variance)", "Custom"],
        help=("Balanced: all available wellness factors have equal importance. "
              "Data-driven: weights come from PCA of factor scores and reflect variation in this dataset. "
              "Custom: you choose relative importance; it is a preference, not a learned statistical fact.")
    )
    custom_w = {}
    if profile == "Custom":
        st.caption("Set relative importance from 0 to 3. Values are normalised automatically; 0 excludes a factor.")
        for code, f in we.HIERARCHY.items():
            custom_w[code] = st.slider(f["name"], 0.0, 3.0, 1.0, 0.1, key=f"custom_{code}")
    st.divider()
    st.subheader("LLM explanation (Groq)")
    if not (get_token() and not st.session_state.get("groq_key_input")):
        st.text_input("Groq API key (kept only in this session)", type="password", key="groq_key_input",
                      help="Better: put GROQ_API_KEY in Streamlit Secrets or an environment variable.")
    hf_model = st.text_input("Model", "llama-3.3-70b-versatile")
 
tabs = st.tabs(["Data & preprocessing", "Factor analysis & weights", "Overall score", "Score a new property", "Explanation", "Method & worked example"])
 
# ------------------------------------------------------------------ TAB 0: data
cfg_base = we.default_config()
with tabs[0]:
