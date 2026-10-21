"""
app.py  --  Property X-Ray : Wellness Score  (Streamlit)
Run:  streamlit run app.py
"""
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
       "A transparent scoring workflow that converts property measurements into factor scores and an overall score. "
       "Diagnostics help assess the data; they do not by themselves prove that a score is objectively valid.",
       ["Data preprocessing", "Normalization", "Correlation", "Factor analysis", "Weight estimation", "Bootstrap",
        "Sub-factor scores", "Factor scores", "Overall Wellness Score"])
 
 
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
        data = pd.read_excel(file)
    else:
        data = pd.read_csv(file)
    return we.normalize_input_columns(data)
 
 
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
                  help="More = more bootstrap resamples for the weight-stability estimate, but slower (cached after the first run).")
    level = st.selectbox("Bootstrap interval level", [0.90, 0.95, 0.99], index=1, format_func=lambda v: f"{v:.0%}")
    st.divider()
    st.subheader("Overall factor weighting")
    profile_options = ["Balanced (equal weights)", "Custom", "Data-driven (explained variance)"]
    profile = st.selectbox(
        "How should the available wellness factors be combined?",
        profile_options,
        key="overall_weighting_v2",
        help=("Balanced: each available factor has equal weight. "
              "Custom: you assign relative importance. Data-driven: weights are estimated from "
              "the variation and relationships in this dataset. Data-driven does not mean more important to every buyer.")
    )
    if profile == "Balanced (equal weights)":
        st.caption("Each available factor contributes equally to the overall score.")
    elif profile == "Custom":
        st.caption("Set relative factor importance from 0 to 3. Values are normalized to sum to 100%; 0 excludes a factor. Sliders for factors absent from the uploaded data are ignored.")
    else:
        st.caption("Weights are derived from PCA of the available factor scores. This reflects data structure, not personal priorities.")
    custom_w = {}
    if profile == "Custom":
        for code, f in we.HIERARCHY.items():
            custom_w[code] = st.slider(f["name"], 0.0, 3.0, 1.0, 0.1)
    st.divider()
    st.subheader("LLM explanation (Groq)")
    if not (get_token() and not st.session_state.get("groq_key_input")):
