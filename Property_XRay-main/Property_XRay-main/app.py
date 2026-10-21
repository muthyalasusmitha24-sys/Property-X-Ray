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
    if src == "Synthetic sample":
        df = cached_sample(int(n_rows), int(seed_data))
        st.success(f"Generated {len(df)} synthetic properties × {df.shape[1] - 1} parameter columns "
                   "(with a few missing values and outliers injected on purpose to exercise the cleaning step).")
    else:
        up = st.file_uploader("Upload CSV / Excel (one row per property, one column per parameter)",
                              type=["csv", "xlsx", "xls"])
        tmpl = we.generate_sample_data(3, 1, messy=False).iloc[:0]
        st.download_button("⬇ Download blank template (column names)", to_csv(tmpl.set_index("Property_ID")),
                           "wellness_template.csv")
        if up is None:
            st.info("No file uploaded yet. A synthetic demonstration dataset is shown so you can explore the dashboard. "
                    "Upload your CSV/Excel file here to score your own reference dataset.")
            df = cached_sample(100, 42)
        else:
            df = read_upload(up)
    if "Property_ID" in df.columns and df["Property_ID"].is_unique:
        df = df.set_index(df["Property_ID"].astype(str)).drop(columns="Property_ID")
    else:
        df = df.drop(columns=[c for c in ["Property_ID"] if c in df.columns])
        df.index = [f"P{i + 1:03d}" for i in range(len(df))]
    T.section("Data preview", "First 20 rows of the reference dataset")
    st.dataframe(df.head(20), width="stretch")
 
    match = (cfg_base.assign(found=cfg_base["Column"].isin(df.columns))
             .groupby(["Code", "Factor"]).agg(columns_found=("found", "sum"), columns_expected=("found", "size"))
             .reset_index())
    c1, c2 = st.columns([1, 2])
    c1.markdown("##### Column coverage")
    c1.dataframe(match, hide_index=True, width="stretch")
    with c2:
        st.markdown("##### Parameter configuration")
        st.caption(
            "Direction tells the model how to interpret a raw value: **Higher is better** means larger values "
            "increase the normalised score; **Lower is better** means smaller values increase it (for example, "
            "AQI, noise, crime rate or commute time). Min/Max are optional reference bounds for scaling to 0–100. "
            "If left blank, the observed minimum and maximum in the reference dataset are used. Check each row "
            "against the real-world meaning and units of your data."
        )
        cfg = st.data_editor(
            cfg_base, key="cfg_editor", hide_index=True, width="stretch", height=300,
            disabled=["Code", "Factor", "Subfactor", "Parameter", "Column"],
            column_config={
                "Direction": st.column_config.SelectboxColumn(
                    "Scoring direction", options=["higher", "lower"],
                    help="higher = larger raw values score higher; lower = smaller raw values score higher"
                ),
                "Min": st.column_config.NumberColumn("Benchmark minimum", help="Optional lower reference bound"),
                "Max": st.column_config.NumberColumn("Benchmark maximum", help="Optional upper reference bound")
            })
    if match["columns_found"].sum() == 0:
        st.error("No matching columns found – check the template.")
        st.stop()
 
# ------------------------------------------------------------------ run engine
with st.spinner("Running the full pipeline (factor analysis + bootstrap) – cached after the first run…"):
    results = cached_run(df, cfg, float(alpha), int(B), 7)
if not results:
    st.error("No factor has ≥3 matching columns.")
    st.stop()
if profile == "Data-driven (explained variance)":
    pw = None
elif profile == "Custom":
    pw = custom_w
else:
    # Equal relative importance across all wellness factors.
    pw = {code: 1.0 for code in results}
F, fw, overall, o_lo, o_hi = we.combine_overall(results, pw, level)
names = {c: r.name for c, r in results.items()}
 
# ------------------------------------------------------------------ TAB 1: pipeline
with tabs[1]:
    code = st.selectbox("Wellness factor", list(results), format_func=lambda c: f"{c} – {names[c]}",
                        index=list(results).index("ENV") if "ENV" in results else 0, key="sel_factor")
    r = results[code]
    lo, hi = r.ci(level)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Mean factor score", f"{r.score.mean():.1f}")
    m2.metric("Parameters / sub-factors", f"{len(r.columns)} / {len(r.sub_map)}")
    m3.metric("KMO", f"{r.diagnostics['kmo']:.2f}")
    m4.metric("Factors retained", r.n_factors)
    m5.metric(f"Avg {level:.0%} CI width", f"{(hi - lo).mean():.1f}")
    if r.flags:
        with st.expander(f"{len(r.flags)} possible flaw(s) detected", expanded=True):
            for s, msg in r.flags:
                T.flag(s, msg)
 
    with st.expander("1 · Raw parameters", expanded=False):
        st.dataframe(r.raw.rename(columns=short).head(25), width="stretch")
    with st.expander("2 · Data preprocessing: missing values and outliers"):
        st.caption("Non-numeric values are treated as missing; missing values are filled with each parameter column’s median. Extreme values are capped at Tukey fences (Q1 − 1.5×IQR and Q3 + 1.5×IQR). This reduces the influence of outliers without deleting property rows.")
        st.dataframe(r.clean_stats.rename(index=short).round(2), width="stretch")
    with st.expander("3 · Normalisation to 0–100"):
        st.caption("Values are min–max scaled using the configured benchmark bounds where supplied; otherwise the reference dataset’s observed bounds are used. Lower-is-better parameters are reversed so that a higher score consistently means a more favourable result.")
        st.dataframe(r.norm_stats.rename(index=short).round(2), width="stretch")
        st.dataframe(r.Xn.rename(columns=short).head(25).round(1), width="stretch")
    with st.expander("4 · Pearson correlation matrix", expanded=False):
        st.caption("Each cell ranges from −1 to +1: +1 means two parameters rise together, −1 means they move in opposite directions, and values near 0 indicate little linear association. Correlation does not prove causation.")
        cm = r.corr.copy()
        cm.index = cm.columns = [short(c) for c in cm.columns]
        fig = px.imshow(cm, color_continuous_scale=T.DIVERGING, zmin=-1, zmax=1, aspect="auto", height=600)
        fig.update_xaxes(side="bottom", tickangle=-55, automargin=True)
        fig.update_yaxes(automargin=True)
        fig.update_layout(margin=dict(l=150, r=35, t=35, b=150))
        T.show(fig)
    with st.expander("5 · Factor analysis: how many shared patterns are present?", expanded=True):
        st.caption("Factor analysis looks for groups of parameters that tend to vary together. The scree chart shows eigenvalues; the dashed line marks 1. KMO and Bartlett’s test are diagnostics of whether correlation patterns are suitable for factor analysis, not proof that the final score is objectively valid.")
        ev = r.eigenvalues
        xs = list(range(1, len(ev) + 1))
        fig = go.Figure(go.Bar(x=xs, y=ev, name="Eigenvalue",
                               marker_color=[T.TEAL if v > 1 else "#B9C9C6" for v in ev]))
        fig.add_trace(go.Scatter(x=xs, y=ev, mode="lines", line=dict(color=T.TEAL_DARK, width=1.5), showlegend=False))
        fig.add_hline(y=1, line_dash="dash", line_color=T.CORAL, annotation_text="Kaiser criterion (λ = 1)",
                      annotation_font_color=T.CORAL)
        fig.update_layout(height=320, xaxis_title="Component", yaxis_title="Eigenvalue", showlegend=False, bargap=0.3)
        T.show(fig)
        st.write(f"**KMO** = {r.diagnostics['kmo']:.3f} · **Bartlett χ²** = {r.diagnostics['bartlett_chi2']:.1f} "
                 f"(p = {r.diagnostics['bartlett_p']:.2g}) · **{r.n_factors} factor(s)** retained, varimax rotation.")
    with st.expander("6 · Factor loadings: which parameters relate to each factor?", expanded=True):
        st.caption("Loadings describe the strength and direction of a parameter’s relationship with a retained factor. Values nearer −1 or +1 indicate a stronger relationship; values near 0 indicate a weaker one. Read row labels on the left and factor columns across the top.")
        L = r.loadings.copy()
        L.index = [short(c) for c in L.index]
        fig = px.imshow(L, color_continuous_scale=T.DIVERGING, zmin=-1, zmax=1, aspect="auto",
                        height=max(500, 34 * len(L)), text_auto=".2f", labels=dict(x="Retained factor", y="Parameter", color="Loading"))
        fig.update_xaxes(side="bottom", tickangle=0, automargin=True)
        fig.update_yaxes(automargin=True)
        fig.update_layout(margin=dict(l=190, r=35, t=45, b=80))
        T.show(fig)
    with st.expander("7 · Compare the two parameter-weight methods", expanded=False):
        st.caption("Loading/communality weights reflect how strongly a parameter relates to the retained factors. Entropy weights reflect how much the parameter varies across properties. Neither alone means a parameter is inherently more important.")
        w = r.weights
        d = pd.DataFrame({"parameter": w["parameter"], "Loading weight %": w["loading_weight"] * 100,
                          "Entropy weight %": w["entropy_weight"] * 100}).melt("parameter")
        fig = px.bar(d, x="value", y="parameter", color="variable", barmode="group", orientation="h",
                     height=max(400, 24 * len(w)), color_discrete_sequence=[T.TEAL, T.CORAL],
                     labels={"value": "Weight %", "parameter": ""})
        T.show(T.style_bars(fig))
    with st.expander("8 · Bootstrap stability of parameter weights", expanded=False):
        st.caption("Bootstrap repeatedly samples property rows with replacement and recalculates weights. The point is the estimated weight; the horizontal interval shows how much that weight varied across resamples. Wider intervals mean less stable estimates.")
        w = r.weights.sort_values("final_weight")
        a = (1 - level) / 2 * 100
        blo = np.percentile(r.boot_w, a, axis=0) if len(r.boot_w) else r.weights["final_weight"].values
        bhi = np.percentile(r.boot_w, 100 - a, axis=0) if len(r.boot_w) else r.weights["final_weight"].values
        bl = pd.Series(blo, index=r.columns)[w.index].values * 100
        bh = pd.Series(bhi, index=r.columns)[w.index].values * 100
        fv = w["final_weight"].values * 100
        fig = go.Figure(go.Scatter(x=fv, y=w["parameter"], mode="markers",
                                   marker=dict(color=T.TEAL, size=9, line=dict(color="white", width=1.5)),
                                   error_x=dict(type="data", symmetric=False, array=np.maximum(bh - fv, 0),
                                                arrayminus=np.maximum(fv - bl, 0), color=T.CORAL, thickness=1.6)))
        fig.update_layout(height=max(400, 24 * len(w)), xaxis_title=f"Weight % (bootstrap mean, {level:.0%} interval)")
        T.show(fig)
    with st.expander("9 · Final parameter weights", expanded=True):
        t = r.weights.copy()
        t.index = [short(c) for c in t.index]
        pct = ["loading_weight", "entropy_weight", "combined_weight", "bootstrap_mean", "bootstrap_std",
               "ci_lo", "ci_hi", "final_weight"]
        t[pct] = t[pct] * 100
        st.dataframe(t.sort_values("final_weight", ascending=False).round(3), width="stretch")
    with st.expander("10 · Sub-factor scores and explained-variance weights", expanded=True):
        c1, c2 = st.columns(2)
        T.show(px.box(r.sub_scores.melt(var_name="Sub-factor", value_name="Score"), x="Score",
                      y="Sub-factor", height=380, color_discrete_sequence=[T.TEAL]), c1)
        sw = (r.sub_weights * 100).sort_values()
        T.show(T.style_bars(px.bar(x=sw.values, y=sw.index, orientation="h", height=380,
                                   labels={"x": "Explained-variance weight %", "y": ""},
                                   color_discrete_sequence=[T.TEAL])), c2)
        st.dataframe(r.sub_scores.head(25).round(1), width="stretch")
    with st.expander("11 · Factor score + confidence interval", expanded=True):
        tbl = pd.DataFrame({"Factor score": r.score, f"CI low": lo, f"CI high": hi}).round(2)
        tbl["Band"] = tbl["Factor score"].map(we.band)
        c1, c2 = st.columns([1, 1])
        fig = T.add_band_zones(T.style_hist(px.histogram(tbl, x="Factor score", nbins=20, height=320)))
        fig.update_xaxes(range=[0, 100])
        T.show(fig, c1)
        c2.dataframe(T.style_df(tbl.sort_values("Factor score", ascending=False), 2), width="stretch", height=320)
 
# ------------------------------------------------------------------ TAB 2: overall
with tabs[2]:
    T.section("Overall Wellness Score", f"Weighting profile: {profile}  ·  {level:.0%} confidence interval")
    out = F.rename(columns=names).round(1)
    out["Wellness Score"] = overall.round(1)
    out[f"CI low ({level:.0%})"] = o_lo.round(1)
    out[f"CI high ({level:.0%})"] = o_hi.round(1)
    out["Band"] = overall.map(we.band)
    out = out.sort_values("Wellness Score", ascending=False)
    g, k = st.columns([1.1, 2])
    T.show(T.gauge(overall.mean(), "Mean Wellness Score", 250), g)
    with k:
        k1, k2, k3 = st.columns(3)
        k1.metric("Mean score", f"{overall.mean():.1f}")
        k2.metric("Best / worst", f"{overall.max():.1f} / {overall.min():.1f}")
        k3.metric("Avg CI width", f"{(o_hi - o_lo).mean():.1f}")
        st.markdown(f"Typical band: {T.badge(we.band(overall.mean()))}", unsafe_allow_html=True)
    left, right = st.columns([1, 1])
    fwp = (fw * 100).rename(index=names).sort_values()
    T.show(T.style_bars(px.bar(x=fwp.values, y=fwp.index, orientation="h", height=340, color_discrete_sequence=[T.TEAL],
                               labels={"x": "Factor weight %  (" + profile + ")", "y": ""})), left)
    fig = T.add_band_zones(T.style_hist(px.histogram(overall, nbins=20, height=340, labels={"value": "Wellness Score"})))
    fig.update_xaxes(range=[0, 100], title="Wellness Score")
    T.show(fig, right)
    top = out.head(25).reset_index(names="Property")
    fig = go.Figure(go.Scatter(x=top["Property"], y=top["Wellness Score"], mode="markers",
                               marker=dict(size=10, color=[T.BAND_COLORS[b] for b in top["Band"]],
                                           line=dict(color="white", width=1.5)),
                               error_y=dict(type="data", symmetric=False, color="#9AA5A8", thickness=1.4,
                                            array=top[f"CI high ({level:.0%})"] - top["Wellness Score"],
                                            arrayminus=top["Wellness Score"] - top[f"CI low ({level:.0%})"])))
    T.add_band_zones(fig, "y")
    fig.update_layout(height=340, title=f"Top 25 properties with {level:.0%} CI", yaxis_title="Wellness Score",
                      showlegend=False)
    T.show(fig)
    st.dataframe(T.style_df(out), width="stretch", height=400)
    st.download_button("⬇ Download all scores (CSV)", to_csv(out), "wellness_scores.csv")
 
    st.divider()
    T.section("Single property drill-down", "Compare one property against the dataset average")
    pid = st.selectbox("Property", list(F.index))
    cats = [names[c] for c in F.columns]
    vals = list(F.loc[pid].values)
    fig = go.Figure(go.Scatterpolar(r=vals + vals[:1], theta=cats + cats[:1], fill="toself", name=str(pid),
                                    line=dict(color=T.TEAL, width=2.5), fillcolor="rgba(15,118,110,0.22)"))
    fig.add_trace(go.Scatterpolar(r=list(F.mean().values) + [F.mean().values[0]], theta=cats + cats[:1],
                                  name="Dataset mean", line=dict(dash="dash", color=T.CORAL, width=2)))
    fig.update_layout(polar=dict(bgcolor="rgba(255,255,255,0.6)",
                                 radialaxis=dict(range=[0, 100], gridcolor=T.LINE, linecolor=T.LINE),
                                 angularaxis=dict(gridcolor=T.LINE)), height=420)
    d1, d2 = st.columns([1, 1])
    T.show(fig, d1)
    with d2:
        T.show(T.gauge(overall[pid], f"Wellness Score · {pid}", 210))
        st.markdown(f"Band: {T.badge(we.band(overall[pid]))} &nbsp; "
                    f"<span style='color:{T.MUTED}'>CI {o_lo[pid]:.1f} – {o_hi[pid]:.1f}</span>",
                    unsafe_allow_html=True)
    st.markdown("#### Why this score?")
    st.markdown(we.explain_property(results, F, fw, overall, o_lo, o_hi, pid, profile, level))
 
# ------------------------------------------------------------------ TAB 3: new property
with tabs[3]:
    T.section("Score a new property", "Estimate a comparable wellness score for a property not used to fit the reference model")
    st.markdown(
        "**What does this score mean?** Enter the raw parameter values for a property you want to assess. "
        "The app applies the reference dataset’s preprocessing rules, scaling bounds and learned parameter/factor weights, "
        "then combines the factor scores into a 0–100 Wellness Score. The confidence interval shows uncertainty from "
        "bootstrap weight variation. It is a constructed comparison index—not a probability, property valuation, "
        "legal certification, or guarantee of health/safety. For a fair comparison, use the same units and benchmark rules."
    )
    mode = st.radio("Input", ["Enter one property manually", "Upload file with new properties"], horizontal=True)
    new_df = None
    if mode.startswith("Enter"):
        all_cols = [c for r_ in results.values() for c in r_.columns]
        base = pd.DataFrame({"Column": all_cols,
                             "Value": [float(pd.to_numeric(df[c], errors="coerce").median()) for c in all_cols]})
        edit = st.data_editor(base, hide_index=True, height=350, width="stretch", disabled=["Column"],
                              key="manual_entry")
        new_df = pd.DataFrame([dict(zip(edit["Column"], edit["Value"]))], index=["Your property"])
    else:
        nf = st.file_uploader("New properties (same columns as the template)", type=["csv", "xlsx", "xls"],
                              key="newfile")
        if nf is not None:
            new_df = read_upload(nf)
            if "Property_ID" in new_df.columns:
                new_df = new_df.set_index(new_df["Property_ID"].astype(str)).drop(columns="Property_ID")
    if new_df is not None:
        Fn, on, nlo, nhi, subs = we.score_new(results, new_df, fw, level)
        res_tbl = Fn.rename(columns=names).round(1)
        res_tbl["Wellness Score"] = on.round(1)
        res_tbl[f"CI low ({level:.0%})"] = nlo.round(1)
        res_tbl[f"CI high ({level:.0%})"] = nhi.round(1)
        res_tbl["Band"] = on.map(we.band)
        st.dataframe(T.style_df(res_tbl), width="stretch")
        st.session_state["new_scored"] = (Fn, on, nlo, nhi, subs)
        first = Fn.index[0]
        rows = [(names[c], s, v) for c in results for s, v in subs[c].loc[first].items()]
        sub_df = pd.DataFrame(rows, columns=["Factor", "Sub-factor", "Score"]).sort_values("Score")
        with st.expander("Sub-factor scores for the first property"):
            st.dataframe(T.style_df(sub_df), hide_index=True, width="stretch")
        st.download_button("⬇ Download (CSV)", to_csv(res_tbl), "new_property_scores.csv")
 
# ------------------------------------------------------------------ TAB 4: explanation
with tabs[4]:
    T.section("How was the score derived?", "Plain-language walk-through of the calculation")
    scope = st.radio("Explain", ["Selected factor (Pipeline tab)", "Overall score for a property"], horizontal=True)
    if scope.startswith("Selected"):
        rr = results[st.session_state.get("sel_factor", list(results)[0])]
        text = we.explain_factor(rr, level)
    else:
        pid2 = st.selectbox("Property", list(F.index), key="expl_pid")
        text = we.explain_property(results, F, fw, overall, o_lo, o_hi, pid2, profile, level)
        flags = [f"{results[c].name}: {m}" for c in results for _, m in results[c].flags]
        if flags:
            text += "\n\n**Data-quality flags across factors:**\n" + "\n".join(f"- {f}" for f in flags[:15])
    st.markdown(text)
    st.divider()
    st.markdown("#### AI narrative + critique (Groq)")
    q = st.text_input("Optional question for the model", "")
    if st.button("Generate AI explanation", type="primary"):
        tok = get_token()
        if not tok:
            st.warning("Add your Groq API key in the sidebar, in Streamlit Secrets (GROQ_API_KEY), "
                       "or as an environment variable.")
        else:
            with st.spinner("Asking the model…"):
                try:
                    st.markdown(we.llm_explain(text, tok, hf_model, q))
                except Exception as e:
                    st.error(f"LLM call failed: {e}\n\nCheck the API key, your Groq rate limits, "
                             "or try another model name (e.g. llama-3.1-8b-instant).")
 
# ------------------------------------------------------------------ TAB 5: method
with tabs[5]:
    T.section("Method and worked example", "What goes in, what happens, and what comes out at every stage")
    st.markdown("""
### Pipeline at a glance
```text
Raw property table
      ↓
Data preprocessing (numeric conversion → median imputation → Tukey outlier capping)
      ↓
Normalise each parameter to 0–100 and align direction
      ↓
Pearson correlation → factor-analysis diagnostics and loadings
      ↓
Communality/loading weights + entropy weights → geometric blend
      ↓
Bootstrap resampling → average weights and uncertainty intervals
      ↓
Parameter weights → sub-factor scores → factor scores
      ↓
Overall factor weights → overall Wellness Score and confidence interval
```

### 1. Data preprocessing
The input is a table: each row is a property and each parameter is a column named `CODE: Parameter` (for example, `ENV: AQI value`). Text that cannot be converted to a number is treated as missing. Missing values are filled with that parameter's median. Outliers are capped—not removed—at Tukey fences: `Q1 − 1.5 × IQR` and `Q3 + 1.5 × IQR`, where `IQR = Q3 − Q1`. This keeps rows while limiting extreme values' influence.

### 2. Direction and normalisation
The **direction** is the scoring rule, not a statistical weight. For a parameter where higher is better, `score = 100 × (x − min) / (max − min)`. For lower-is-better parameters, the direction is reversed: `score = 100 − scaled_score`. For example, if AQI is lower-is-better and the reference bounds are 0 and 200, AQI 50 scales to 25, then reverses to **75/100**. Explicit benchmark Min/Max make scores comparable against those fixed bounds; blank bounds use the reference dataset's observed bounds. Values are clipped to 0–100.

### 3. Correlation and factor analysis
Pearson correlation measures linear co-movement between pairs of normalised parameters (−1 to +1). Factor analysis searches for a smaller set of shared patterns behind correlated parameters. Eigenvalues/scree and the Kaiser rule (eigenvalue > 1, subject to the implementation's cap) help choose the number of factors. KMO and Bartlett's test are suitability diagnostics. **Factor loadings** indicate how strongly each parameter relates to each retained factor; they are not scores themselves.

### 4. How parameter weights are derived
Two signals are combined:
- **Loading/communality weight:** for parameter `i`, communality is `hᵢ = Σₖ loadingᵢₖ²`. Its loading weight is `hᵢ / Σᵢ hᵢ`. Squaring means negative loadings contribute by magnitude rather than cancelling positive loadings.
- **Entropy weight:** normalised values are converted into proportions across properties. Entropy is `eⱼ = −Σᵢ pᵢⱼ ln(pᵢⱼ) / ln(n)`. A parameter that is almost identical for every property has high entropy and low information weight; one that differentiates properties more has lower entropy and more weight. The implementation handles zero proportions safely.
- **Blend:** the sidebar parameter `α` controls the balance. `α = 1` uses loading weights; `α = 0` uses entropy weights. Between them, the engine combines them geometrically: `combinedⱼ ∝ loading_weightⱼ^α × entropy_weightⱼ^(1−α)`, then normalises the combined weights to sum to 1.

**Small arithmetic example:** if a parameter has communality `0.60` and total communality across parameters is `3.00`, its loading weight is `0.60 / 3.00 = 0.20` (20%). If its entropy weight is 10% and α is 0.5, its unnormalised blend is `√(0.20 × 0.10) = 0.1414`; all parameters' blend values are then divided by their total so final weights sum to 100%.

### 5. Bootstrap validation
The engine resamples property rows with replacement and repeats the weight calculation. The final parameter weights use the bootstrap results as implemented by the engine; percentile intervals show their variability. More resamples generally stabilise estimates but take longer. The confidence interval is conditional on the observed dataset and this weighting procedure—it does not include all measurement or model uncertainty.

### 6. From parameter weights to the overall score
Within each sub-factor, its parameter weights are renormalised and used to calculate a weighted average of 0–100 parameter scores. PCA-derived sub-factor weights then combine sub-factors into each factor score. Finally, the nine wellness factors are combined using one of these choices:
- **Balanced:** equal relative weight for every available factor.
- **Data-driven:** factor weights are derived from PCA/explained variation in the reference factor scores. This reflects patterns in this dataset, not universal importance.
- **Custom:** you set relative weights; the app normalises them to sum to 100%. A zero excludes that factor.

### 7. What “Score a new property” means
This is not a new model training step. The reference dataset has already supplied medians, normalisation bounds, weights and bootstrap weight sets. For a new property, the app applies those same learned settings to its supplied raw values, calculates sub-factor and factor scores, then returns the overall 0–100 index and an interval. The score is useful for comparison under the chosen parameters and assumptions; it is not a market price, probability, certification, or objective measure of every aspect of a property.

### Important limitations
- If benchmark bounds are blank, scores depend on the properties in the reference dataset and may change when that dataset changes.
- Parameter directions must be reviewed for the actual definitions and units in your file; a wrong direction reverses the interpretation.
- Factor analysis is more informative when enough rows and meaningful correlations exist. Check the diagnostics and data-quality flags.
- Synthetic data are only for testing the pipeline and must not be interpreted as real property assessments.
""")
