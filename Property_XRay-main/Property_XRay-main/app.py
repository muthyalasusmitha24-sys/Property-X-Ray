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
       "A transparent, data-dependent wellness scoring workflow. Statistical diagnostics help assess the data; "
       "they do not by themselves prove that a score is objectively valid.",
       ["Data preprocessing", "Normalize", "Correlate", "Factor analysis", "Entropy", "Bootstrap", "Weights",
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
    st.subheader("Data and model settings")
    src = st.radio("Data source", ["Synthetic sample", "Upload file"])
    if src == "Synthetic sample":
        n_rows = st.number_input("Rows", 30, 1000, 100, 10)
        seed_data = st.number_input("Data seed", 0, 9999, 42)
    alpha = st.slider("Weight blend (α)", 0.0, 1.0, 0.5, 0.05,
                  help="α = 0 uses entropy weights only; α = 1 uses communality-based weights only; 0.5 combines both geometrically.")
    B = st.slider("Bootstrap resamples", 20, 400, 100, 20,
                  help="More = more stable weights / CI, but slower (cached after first run).")
    level = st.selectbox("Confidence level", [0.90, 0.95, 0.99], index=1, format_func=lambda v: f"{v:.0%}")
    st.divider()
    st.subheader("Overall factor weighting")
    profile_options = ["Balanced (equal weights)", "Custom", "Data-driven (explained variance)"]
    profile = st.selectbox(
        "Overall weighting method",
        profile_options,
        help=("Balanced: equal importance for each available factor. "
              "Custom: you set relative importance. Data-driven: weights are estimated from "
              "variation in the reference dataset; this does not mean they are objectively best.")
    )
    custom_w = {}
    if profile == "Custom":
        for code, f in we.HIERARCHY.items():
            custom_w[code] = st.slider(f["name"], 0.0, 3.0, 1.0, 0.1)
    st.divider()
    st.subheader("LLM explanation (Groq)")
    if not (get_token() and not st.session_state.get("groq_key_input")):
        st.text_input("Groq API key (kept only in this session)", type="password", key="groq_key_input",
                      help="Better: put GROQ_API_KEY in Streamlit Secrets or an environment variable.")
    hf_model = st.text_input("Model", "llama-3.3-70b-versatile")
 
tabs = st.tabs(["Data", "Factor pipeline", "Overall score", "Score new property", "Explanation", "Method"])
 
# ------------------------------------------------------------------ TAB 0: data
cfg_base = we.default_config()
