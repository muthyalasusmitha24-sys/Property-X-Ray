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

st.set_page_config(page_title="Property X-Ray – Wellness Score", layout="wide")
T.inject()
T.hero("Wellness Score",
       "A transparent, statistically validated score for how well a property supports everyday living, "
       "from raw parameters to a final score with a confidence interval.",
       we.STAGES)

short = we.short_name
DIR_INV = {v: k for k, v in we.DIRECTION_LABEL.items()}


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


@st.cache_data(show_spinner=False)
def cached_toy(alpha):
    return we.toy_example(alpha)


def read_upload(file):
    if file.name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    return pd.read_csv(file)


def to_csv(df):
    return df.to_csv().encode("utf-8")


def with_index(df):
    """Use Property_ID as row label when it is present and unique; otherwise P001, P002, ..."""
    if "Property_ID" in df.columns and df["Property_ID"].is_unique:
        return df.set_index(df["Property_ID"].astype(str)).drop(columns="Property_ID")
    df = df.drop(columns=[c for c in ["Property_ID"] if c in df.columns])
    df.index = [f"P{i + 1:03d}" for i in range(len(df))]
    return df


def data_scale(df, cols):
    """Min / max that the scaling step would use for these columns when no benchmark is given."""
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.DataFrame(columns=["min", "max"])
    raw = df[cols].apply(pd.to_numeric, errors="coerce")
    nan = pd.Series(np.nan, index=cols)
    return we.preprocess(raw, pd.Series("higher", index=cols), nan, nan)["norm_stats"][["min", "max"]]


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    T.sidebar_brand()
    st.subheader("Data")
    src = st.radio("Data source", ["Synthetic sample", "Upload file"])
    if src == "Synthetic sample":
        n_rows = st.number_input("Rows", 30, 1000, 100, 10)
        seed_data = st.number_input("Data seed", 0, 9999, 42)
    st.divider()
    st.subheader("Model settings")
    alpha = st.slider("Weight blend α  (0 = entropy only · 1 = factor analysis only)", 0.0, 1.0, 0.5, 0.05,
                      help="Each parameter gets two importance scores: one from factor analysis (how strongly it "
                           "belongs to the group's hidden pattern) and one from entropy (how much it differs between "
                           "properties). α decides how much each counts. 0.5 = equal say.")
    B = st.slider("Bootstrap resamples", 20, 400, 100, 20,
                  help="How many times the weighting is repeated on randomly resampled properties to test "
                       "stability. More = steadier weights and intervals, but slower (cached after first run).")
    level = st.selectbox("Confidence level", [0.90, 0.95, 0.99], index=1, format_func=lambda v: f"{v:.0%}")
    st.divider()
    st.subheader("Overall weighting")
    profile = st.radio(
        "How should the 9 wellness factors be combined?", we.OVERALL_METHODS,
        captions=["Every factor counts the same (1/9 each).",
                  "Weights come from the data: factors that carry more of the shared pattern count more.",
                  "You decide how much each factor matters."])
    custom_w = {}
    if profile == "Custom":
        st.caption("0 = ignore this factor · 1 = normal · 3 = three times as important. "
                   "Weights are rescaled so they add up to 100%.")
        for code, f in we.HIERARCHY.items():
            custom_w[code] = st.slider(f"{f['name']} ({code})", 0.0, 3.0, 1.0, 0.1)
    st.divider()
    st.subheader("AI explanation (Groq)")
    if not (get_token() and not st.session_state.get("groq_key_input")):
        st.text_input("Groq API key (kept only in this session)", type="password", key="groq_key_input",
                      help="Better: put GROQ_API_KEY in Streamlit Secrets or an environment variable.")
    groq_model = st.text_input("Model", "llama-3.3-70b-versatile")

tabs = st.tabs(["Data", "Factor pipeline", "Overall score", "Score a new property", "Explanation", "Method"])

# ------------------------------------------------------------------ TAB: data
cfg_base = we.default_config()
with tabs[0]:
    if src == "Synthetic sample":
        df = cached_sample(int(n_rows), int(seed_data))
        st.success(f"Generated {len(df)} synthetic properties × {df.shape[1] - 1} parameter columns "
                   "(a few missing values and outliers were injected on purpose so the preprocessing step has "
                   "something to do).")
    else:
        up = st.file_uploader("Upload CSV / Excel (one row per property, one column per parameter)",
                              type=["csv", "xlsx", "xls"])
        tmpl = we.generate_sample_data(3, 1, messy=False).iloc[:0]
        st.download_button("Download blank template (column names)", to_csv(tmpl.set_index("Property_ID")),
                           "wellness_template.csv")
        if up is None:
            st.info("Upload a file to continue. Column names must match the template "
                    "(`CODE: Parameter`, e.g. `ENV: AQI value`). Missing columns are skipped; "
                    "a factor needs at least 3 columns.")
            st.stop()
        df = read_upload(up)
    df = with_index(df)

    T.section("Data preview", "First 20 rows of the reference dataset")
    st.dataframe(df.head(20), width="stretch")

    cov = cfg_base.assign(found=cfg_base["Column"].isin(df.columns))
    match = (cov.groupby(["Code", "Factor"], sort=False)
             .agg(found=("found", "sum"), expected=("found", "size")).reset_index()
             .rename(columns={"Code": "Code", "Factor": "Wellness factor",
                              "found": "Columns found", "expected": "Columns expected"}))
    match["Used in model?"] = np.where(match["Columns found"] >= 3, "Yes", "No (needs ≥ 3 columns)")
    T.section("Column coverage", "Which of the expected parameters were found in your data")
    st.dataframe(match, hide_index=True, width="stretch")
    if match["Columns found"].sum() == 0:
        st.error("No matching columns found – check the template.")
        st.stop()

    st.divider()
    T.section("Parameter settings",
              "Tell the model how to read each parameter. Open a factor below and, only if needed, change the settings.")
    with st.expander("What do these settings mean? (read this first)", expanded=False):
        st.markdown("""
Every parameter is converted to a **0-100 score where 100 is always the best**. Two settings control that conversion.

**1 · Better when  (Higher value / Lower value)**  
Says whether a *bigger raw number is good or bad* for a buyer.
- *Higher value* is better: green cover %, walk score, proximity to parks, rental income.
- *Lower value* is better: AQI, noise (dB), crime rate, commute time, property price, distance to hospital.

If a "lower is better" parameter is wrongly set to "higher", the model will reward polluted, noisy, expensive places.
The engine flags this automatically when a parameter moves *against* the others in its sub-factor.

**2 · Scale min / Scale max  (optional)**  
The raw values that should become **0** and **100** before the direction is applied.
- **Leave blank (default):** the lowest and highest value found in your data are used. Scores are then *relative*
  ("70" = better than most properties in this dataset).
- **Fill in** a benchmark range to get *absolute* scores that do not depend on which properties are in the
  dataset, e.g. AQI 0 → 500, noise 30 → 100 dB, green cover 0 → 100 %.

**Read-only columns** show whether the parameter exists in your data and the min / max that will be used when the
scale boxes are blank (extreme outliers are trimmed first, see *Factor pipeline → Data preprocessing*).

**Worked example.** AQI, *Lower value* is better, scale 0 → 500, observed AQI = 100:  
scaled = 100 × (100 − 0) ÷ (500 − 0) = 20  →  flipped because lower is better  →  score = 100 − 20 = **80**.
""")

    cfg = cfg_base.copy()
    for code, f in we.HIERARCHY.items():
        sub = cfg_base[cfg_base["Code"] == code]
        sc = data_scale(df, list(sub["Column"]))
        in_data = sub["Column"].isin(df.columns)
        view = pd.DataFrame({
            "Sub-factor": sub["Subfactor"].values,
            "Parameter": sub["Parameter"].values,
            "In your data?": np.where(in_data, "Yes", "No"),
            "Better when": sub["Direction"].map(we.DIRECTION_LABEL).values,
            "Scale min (optional)": np.nan,
            "Scale max (optional)": np.nan,
            "Default min (from data)": sc["min"].reindex(sub["Column"]).values,
            "Default max (from data)": sc["max"].reindex(sub["Column"]).values}, index=sub.index)
        with st.expander(f"{f['name']}  [{code}]  ·  {int(in_data.sum())} of {len(sub)} parameters found in data"):
            ed = st.data_editor(
                view, key=f"cfg_{code}", hide_index=True, width="stretch",
                disabled=["Sub-factor", "Parameter", "In your data?", "Default min (from data)", "Default max (from data)"],
                column_config={
                    "Better when": st.column_config.SelectboxColumn(
                        "Better when", options=list(we.DIRECTION_LABEL.values()), required=True,
                        help="Is a bigger raw number good (Higher) or bad (Lower) for the buyer?"),
                    "Scale min (optional)": st.column_config.NumberColumn(
                        "Scale min (optional)", help="Raw value that becomes 0. Blank = use the data's minimum."),
                    "Scale max (optional)": st.column_config.NumberColumn(
                        "Scale max (optional)", help="Raw value that becomes 100. Blank = use the data's maximum."),
                    "Default min (from data)": st.column_config.NumberColumn(format="%.2f"),
                    "Default max (from data)": st.column_config.NumberColumn(format="%.2f")})
            cfg.loc[sub.index, "Direction"] = ed["Better when"].map(DIR_INV).fillna("higher").values
            cfg.loc[sub.index, "Min"] = pd.to_numeric(ed["Scale min (optional)"], errors="coerce").values
            cfg.loc[sub.index, "Max"] = pd.to_numeric(ed["Scale max (optional)"], errors="coerce").values
    bad = cfg[cfg["Min"].notna() & cfg["Max"].notna() & (cfg["Min"] >= cfg["Max"])]
    if len(bad):
        st.warning("Scale min must be smaller than scale max. These rows were ignored (data min/max used instead): "
                   + ", ".join(bad["Column"]))
        cfg.loc[bad.index, ["Min", "Max"]] = np.nan

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
    pw = {c: 1.0 for c in we.HIERARCHY}
F, fw, overall, o_lo, o_hi = we.combine_overall(results, pw, level)
names = {c: r.name for c, r in results.items()}

# ------------------------------------------------------------------ TAB: pipeline
with tabs[1]:
    code = st.selectbox("Wellness factor", list(results), format_func=lambda c: f"{names[c]}  [{c}]",
                        index=list(results).index("ENV") if "ENV" in results else 0, key="sel_factor")
    st.caption(f"Factor code **{code}** = **{names[code]}**. The pipeline below is run separately for each factor; "
               "open each stage to see what happens and where its result is used.")
    r = results[code]
    W = r.weights
    cs = r.clean_stats
    lo, hi = r.ci(level)
    p_ = len(r.columns)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Mean factor score", f"{r.score.mean():.1f}")
    m2.metric("Parameters / sub-factors", f"{p_} / {len(r.sub_map)}")
    m3.metric("KMO", f"{r.diagnostics['kmo']:.2f}", we.kmo_label(r.diagnostics['kmo']), delta_color="off")
    m4.metric("Hidden factors kept", r.n_factors)
    m5.metric(f"Avg {level:.0%} CI width", f"{(hi - lo).mean():.1f}")
    if r.flags:
        with st.expander(f"{len(r.flags)} possible flaw(s) detected", expanded=True):
            for s, msg in r.flags:
                T.flag(s, msg)

    # ---- 1 raw
    with st.expander("1 · Raw parameters", expanded=False):
        T.stage_note("Your data exactly as supplied: one row per property, one column per parameter, in the units "
                     "you measured them (dB, AQI, %, price …).",
                     "Passed unchanged to stage 2, Data preprocessing.")
        st.dataframe(r.raw.rename(columns=short).head(25), width="stretch")

    # ---- 2 preprocessing
    with st.expander("2 · Data preprocessing", expanded=True):
        T.stage_note("Raw data is made safe and comparable: gaps are filled, extreme values are trimmed, "
                     "everything is rescaled to 0-100, and 'lower is better' parameters are flipped so that "
                     "100 always means best.",
                     "Every later stage (correlation, factor analysis, entropy, scoring) uses ONLY the "
                     "preprocessed 0-100 scores, never the raw values.")
        pa_, pb_, pc_, pd_ = st.tabs(["A · Missing values", "B · Outliers", "C · Direction & scaling",
                                      "D · Before vs after"])
        with pa_:
            st.markdown("**Rule:** an empty or non-numeric cell is replaced by the **median** of its column. "
                        "The median is used (not the mean) because it is not pulled around by extreme values, so "
                        "a filled-in gap looks like a typical property rather than an unusual one.")
            st.latex(r"x_{\text{missing}} \;\longrightarrow\; \operatorname{median}(\text{column})")
            t = (cs.assign(parameter=[short(c) for c in cs.index])
                 .loc[:, ["parameter", "missing", "missing_pct", "median"]]
                 .rename(columns={"parameter": "Parameter", "missing": "Missing cells",
                                  "missing_pct": "Missing %", "median": "Median used to fill"})
                 .sort_values("Missing cells", ascending=False).reset_index(drop=True))
            st.dataframe(t.round(2), hide_index=True, width="stretch")
            st.caption(f"{int(cs['missing'].sum())} of {r.raw.size} cells were filled. "
                       "Parameters with more than 10% missing are flagged above: heavy imputation pulls "
                       "properties towards the middle.")
        with pb_:
            st.markdown("**Rule:** values outside the **Tukey fences** are *clipped* (winsorised) to the nearest fence, "
                        "not deleted. Fences use the quartiles Q1 and Q3 of the observed values; IQR = Q3 − Q1.")
            st.latex(r"\text{lower}=Q_1-1.5\,\text{IQR}\qquad \text{upper}=Q_3+1.5\,\text{IQR}"
                     r"\qquad x_{\text{clipped}}=\min\!\big(\max(x,\text{lower}),\text{upper}\big)")
            st.markdown("**Why:** one absurd value (a sensor glitch, a typo) would otherwise stretch the min-max scale "
                        "and squash all normal properties into a narrow band near 0 or 100. "
                        "If a column's IQR is 0 (most values identical) no clipping is applied.")
            t = pd.DataFrame({"Parameter": [short(c) for c in cs.index], "Q1": cs["q1"].values, "Q3": cs["q3"].values,
                              "Lower fence": cs["lo"].values, "Upper fence": cs["hi"].values,
                              "Observed min": cs["raw_min"].values, "Observed max": cs["raw_max"].values,
                              "Clipped (low)": cs["clipped_low"].values, "Clipped (high)": cs["clipped_high"].values})
            st.dataframe(t.sort_values(["Clipped (high)", "Clipped (low)"], ascending=False).round(2),
                         hide_index=True, width="stretch")
            st.caption(f"{int(cs['outliers_clipped'].sum())} values were clipped. Parameters where more than 5% of "
                       "rows were clipped are flagged above.")
        with pc_:
            st.markdown("**Rule:** after cleaning, each parameter is rescaled to 0-100 with **min-max scaling**. "
                        "If the parameter is *lower is better*, the score is then flipped.")
            st.latex(r"s=100\cdot\frac{x-\min}{\max-\min}\quad(\text{kept in }0\text{–}100)\qquad "
                     r"\text{score}=\begin{cases}s & \text{higher is better}\\ 100-s & \text{lower is better}\end{cases}")
            st.markdown("**min / max** are your *benchmark* Scale min / max from the Data tab when you gave them, "
                        "otherwise the lowest / highest cleaned value in the data. If min = max the parameter has "
                        "no spread and gets 50 for everyone (it then carries no weight).")
            t = pd.DataFrame({"Parameter": [short(c) for c in r.columns],
                              "Better when": [we.DIRECTION_LABEL[d] for d in r.norm_stats["direction"]],
                              "Raw value → 0": r.norm_stats["min"].values, "Raw value → 100": r.norm_stats["max"].values,
                              "Scale source": r.norm_stats["bounds"].map({"data": "data min/max", "benchmark": "your benchmark"}).values})
            st.dataframe(t.round(2), hide_index=True, width="stretch")
        with pd_:
            pick = st.selectbox("Parameter to inspect", r.columns, format_func=short, key=f"pp_{code}")
            c1, c2 = st.columns(2)
            f1 = px.histogram(r.raw[pick].dropna().to_frame("Raw value"), x="Raw value", nbins=20, height=300,
                              title=f"Before: raw {short(pick)}")
            f1.update_layout(xaxis_title=f"{short(pick)} (raw units)", yaxis_title="Properties")
            T.show(T.style_hist(f1, T.GOLD), c1)
            f2 = px.histogram(r.Xn[pick].to_frame("Score"), x="Score", nbins=20, height=300,
                              title="After: 0-100 score (100 = best)")
            f2.update_layout(xaxis_title="Preprocessed score", yaxis_title="Properties")
            f2.update_xaxes(range=[0, 100])
            T.show(T.style_hist(f2, T.TEAL), c2)
            st.markdown("**First 25 rows after preprocessing (all parameters, 0-100):**")
            st.dataframe(r.Xn.rename(columns=short).head(25).round(1), width="stretch")

    # ---- 3 correlation
    with st.expander("3 · Correlation matrix", expanded=True):
        T.stage_note("Measures, for every pair of parameters, whether they rise and fall together "
                     "(Pearson correlation, −1 to +1) across all properties.",
                     "Stage 4 (factor analysis) is built on this matrix; it also drives the 'near-duplicate' and "
                     "'wrong direction' warnings.")
        grp = []
        sub_of = list(W["subfactor"])
        for i in range(1, p_):
            if sub_of[i] != sub_of[i - 1]:
                grp.append(i)
        ylab = [f"{i + 1} · {short(c)}" for i, c in enumerate(r.columns)]
        fig = T.heatmap(r.corr.values, [str(i + 1) for i in range(p_)], ylab, text=p_ <= 20, text_size=9,
                        cbar_title="r", boundaries=grp, xhover=[short(c) for c in r.columns], hover_label="r")
        T.show(fig)
        st.caption("Read it like a map: **teal** = the two parameters move together, **red** = they move in opposite "
                   "directions, **pale** = unrelated. Columns are numbered to match the rows; black lines separate "
                   "sub-factors. Groups: " + " · ".join(
                       f"{int(np.where(np.array(sub_of) == s)[0][0]) + 1}–{int(np.where(np.array(sub_of) == s)[0][-1]) + 1} {s}"
                       for s in dict.fromkeys(sub_of)))
        iu = np.triu_indices(p_, 1)
        pairs = pd.DataFrame({"Parameter A": [short(r.columns[i]) for i in iu[0]],
                              "Parameter B": [short(r.columns[j]) for j in iu[1]], "r": r.corr.values[iu]})
        c1, c2 = st.columns(2)
        c1.markdown("**Strongest positive pairs**")
        c1.dataframe(pairs.sort_values("r", ascending=False).head(5).round(2), hide_index=True, width="stretch")
        c2.markdown("**Strongest negative pairs** (negative values deserve a direction check)")
        c2.dataframe(pairs.sort_values("r").head(5).round(2), hide_index=True, width="stretch")

    # ---- 4 factor analysis
    with st.expander("4 · Factor analysis", expanded=True):
        T.stage_note("Looks for a small number of hidden themes ('factors') that explain why groups of parameters "
                     "move together, and checks that the data are suitable for this.",
                     "The number of hidden factors found here is passed to stage 5, where each parameter's tie "
                     "to those factors becomes its importance weight.")
        fa_a, fa_b = st.tabs(["What is factor analysis?", "Scree chart & suitability checks"])
        with fa_a:
            st.markdown("""
**In one sentence:** factor analysis asks *"are the many columns I measured really just a few underlying themes?"*

**Everyday example.** Suppose five parameters are AQI, pollutant level, ventilation, tree cover and open space.
Neighbourhoods with clean air usually also have good ventilation and many trees, so these columns rise and fall
together. Factor analysis spots that shared movement and says: *there is one hidden theme (call it "how green and clean
the area is") and all five columns are partly measuring it.*

| Term | Meaning |
|---|---|
| **Hidden factor** | A theme that is not measured directly but shows up in several columns at once. |
| **Loading** (−1 … +1) | How strongly one parameter is tied to one hidden factor (like a correlation). |
| **Communality h** | Add up the *squared* loadings of a parameter: the share (0-1) of its ups and downs explained by the hidden themes. High h = it agrees with its group; low h = it behaves on its own. |
| **Eigenvalue** | How much of the total variation a hidden factor captures. The **Kaiser rule** keeps factors with eigenvalue > 1 (they explain more than one parameter would alone). |
| **Varimax rotation** | A tidy-up that makes each parameter load mainly on one factor, so the factors are easier to read. Used only when more than one factor is kept. |
| **KMO** | Are the correlations strong enough for factor analysis? 0.6+ is acceptable, 0.8+ is good, below 0.5 is unsuitable. |
| **Bartlett's test** | p < 0.05 means the parameters are correlated enough to be worth analysing (not just random noise). |

**Why we use it for weights.** A parameter that is strongly tied to the group's hidden theme is a good
representative of the factor, so it earns more weight (through its communality). A parameter unrelated to
everything else earns less. **Caution:** this reveals patterns in the data, not what a buyer values.
""")
        with fa_b:
            ev = r.eigenvalues
            xs = list(range(1, len(ev) + 1))
            fig = go.Figure(go.Bar(x=xs, y=ev, marker_color=[T.TEAL if v > 1 else "#B9C9C6" for v in ev],
                                   text=[f"{v:.2f}" for v in ev], textposition="outside", textfont=dict(size=10),
                                   hovertemplate="Component %{x}<br>Eigenvalue %{y:.2f}<extra></extra>"))
            fig.add_hline(y=1, line_dash="dash", line_color=T.CORAL, annotation_text="Kaiser cut-off (eigenvalue = 1)",
                          annotation_position="top right", annotation_font_color=T.CORAL)
            fig.update_layout(height=340, showlegend=False, bargap=0.3, title="Scree chart: eigenvalue of each component",
                              xaxis_title="Component number", yaxis_title="Eigenvalue")
            fig.update_xaxes(dtick=1)
            fig.update_yaxes(range=[0, float(ev.max()) * 1.18])
            T.show(fig)
            st.caption("Teal bars (above the dashed line) are the components kept as hidden factors; grey bars are treated "
                       "as noise. The number kept is also capped at 4 and at one-third of the parameter count.")
            evr = ev / ev.sum()
            tbl = pd.DataFrame({"Component": xs, "Eigenvalue": ev, "% of variance": evr * 100,
                                "Cumulative %": np.cumsum(evr) * 100, "Kept?": ["Yes" if i < r.n_factors else "No" for i in range(len(ev))]})
            st.dataframe(tbl.head(max(r.n_factors + 3, 6)).round(2), hide_index=True, width="stretch")
            d = r.diagnostics
            st.markdown(f"**KMO** = {d['kmo']:.3f} ({we.kmo_label(d['kmo'])}) · **Bartlett χ²** = {d['bartlett_chi2']:.1f}, "
                        f"p = {d['bartlett_p']:.2g} · **{r.n_factors} hidden factor(s)** kept"
                        f"{' with varimax rotation' if r.n_factors > 1 else ''}, explaining about "
                        f"**{d['var_explained'] * 100:.0f}%** of the parameters' variance (sum of communalities ÷ parameters).")

    # ---- 5 loadings
    with st.expander("5 · Factor loadings and communality", expanded=True):
        T.stage_note("Shows how strongly each parameter is tied to each hidden factor, and adds up the squared "
                     "loadings into one number per parameter: its communality h.",
                     "Communality becomes the 'loading weight' of each parameter: h ÷ (sum of all h). Parameters "
                     "that agree with their group's hidden theme get more weight.")
        Lm = r.loadings.copy()
        ylab = [f"{short(c)}" for c in Lm.index]
        T.show(T.heatmap(Lm.values, list(Lm.columns), ylab, text=True, text_size=11, cbar_title="loading",
                         height=max(340, 30 * len(Lm) + 130), hover_label="loading"))
        st.caption("**+1 / −1** (strong teal / red) = tightly tied to that hidden factor; **near 0** = not related. "
                   "The sign only says which way the factor points and is irrelevant for weights, because "
                   "communality uses squared loadings.")
        top_f = Lm.abs().idxmax(axis=1)
        tbl = pd.DataFrame({"Parameter": ylab, "Sub-factor": W["subfactor"].values,
                            "Mainly tied to": top_f.values, "Communality h": W["communality"].values,
                            "Loading weight %": W["loading_weight"].values * 100})
        st.dataframe(tbl.round(3), hide_index=True, width="stretch")

    # ---- 6 entropy
    with st.expander("6 · Entropy weights vs loading weights", expanded=False):
        T.stage_note("Gives each parameter a second importance score: entropy measures how much a parameter "
                     "DIFFERS from property to property. A parameter that is nearly identical everywhere cannot "
                     "rank properties, so it earns little weight.",
                     "Blended with the loading weight using α (sidebar) into the combined weight, which then "
                     "goes into the bootstrap.")
        st.latex(r"p_{ij}=\frac{x_{ij}}{\sum_i x_{ij}}\quad e_j=-\frac{1}{\ln n}\sum_i p_{ij}\ln p_{ij}"
                 r"\quad d_j=1-e_j\quad w^{\text{ent}}_j=\frac{d_j}{\sum_k d_k}")
        st.latex(r"w^{\text{load}}_j=\frac{h_j}{\sum_k h_k}\qquad "
                 r"w_j\;\propto\;\big(w^{\text{load}}_j\big)^{\alpha}\,\big(w^{\text{ent}}_j\big)^{1-\alpha}")
        order = W.sort_values("combined_weight", ascending=False)["parameter"].tolist()
        fig = go.Figure()
        fig.add_bar(y=W["parameter"], x=W["loading_weight"] * 100, name="Loading weight (factor analysis)",
                    orientation="h", marker_color=T.TEAL)
        fig.add_bar(y=W["parameter"], x=W["entropy_weight"] * 100, name="Entropy weight", orientation="h",
                    marker_color=T.CORAL)
        fig.add_scatter(y=W["parameter"], x=W["combined_weight"] * 100, mode="markers",
                        name=f"Combined (α = {alpha:.2f})", marker=dict(symbol="diamond", size=9, color=T.INK,
                                                                      line=dict(color="white", width=1)))
        fig.update_layout(barmode="group", bargap=0.25, height=max(440, 46 * p_ + 120),
                          xaxis_title="Weight % within this factor")
        fig.update_yaxes(categoryorder="array", categoryarray=order, autorange="reversed")
        T.show(fig)
        st.caption(f"Parameters are sorted by combined weight. With α = {alpha:.2f} the combined weight is "
                   f"{'a geometric blend of both' if 0 < alpha < 1 else ('factor analysis only' if alpha == 1 else 'entropy only')}. "
                   "These are the weights from the full dataset, before bootstrap averaging.")

    # ---- 7 bootstrap
    with st.expander("7 · Bootstrap validation of parameter weights", expanded=False):
        T.stage_note(f"Tests whether the weights are stable. Properties are randomly resampled with replacement "
                     f"{len(r.boot_w)} times and stages 3-6 are redone each time, giving {len(r.boot_w)} different "
                     "weight sets.",
                     "The average of those weight sets is the final parameter weight (stage 8); their spread gives "
                     "the interval below and, later, the confidence interval of every score.")
        a = (1 - level) / 2 * 100
        if len(r.boot_w):
            blo, bhi = np.percentile(r.boot_w, a, axis=0), np.percentile(r.boot_w, 100 - a, axis=0)
        else:
            blo = bhi = W["final_weight"].values
        ws = W.assign(blo=blo, bhi=bhi).sort_values("final_weight")
        fv = ws["final_weight"].values * 100
        fig = go.Figure(go.Scatter(
            x=fv, y=ws["parameter"], mode="markers", name="Bootstrap mean",
            marker=dict(color=T.TEAL, size=9, line=dict(color="white", width=1.5)),
            error_x=dict(type="data", symmetric=False, array=np.maximum(ws["bhi"].values * 100 - fv, 0),
                         arrayminus=np.maximum(fv - ws["blo"].values * 100, 0), color=T.CORAL, thickness=1.6)))
        fig.update_layout(height=max(420, 34 * p_ + 120), showlegend=False,
                          xaxis_title=f"Parameter weight % (dot = bootstrap mean, bar = {level:.0%} interval)")
        T.show(fig)
        st.caption("A **short bar** means the weight hardly changes when the sample changes (stable). A **long bar** "
                   "means the weight depends on which properties happen to be in the sample (unreliable). "
                   "CV = standard deviation ÷ mean; CV above 0.6 is flagged.")
        st.dataframe(pd.DataFrame({"Parameter": W["parameter"], "Mean weight %": W["bootstrap_mean"] * 100,
                                   "Std dev %": W["bootstrap_std"] * 100, "CV": W["cv"]})
                     .sort_values("CV", ascending=False).head(6).round(3), hide_index=True, width="stretch")

    # ---- 8 weights
    with st.expander("8 · Final parameter weights", expanded=True):
        T.stage_note("Collects the weights into one table. The bootstrap mean is the PARAMETER WEIGHT. It is then "
                     "shared out inside each sub-factor and multiplied by that sub-factor's weight to give the "
                     "EFFECTIVE weight, the share of the factor score each parameter really controls.",
                     "Parameter weights build the sub-factor scores (stage 9); effective weights are what a "
                     "parameter truly contributes to the factor score, and sum to 100%.")
        st.latex(r"\text{effective}_j=W_s\cdot\frac{w_j}{\sum_{k\in s}w_k}\qquad "
                 r"\text{factor score}=\sum_j \text{effective}_j\cdot x_j")
        t = pd.DataFrame({
            "Parameter": W["parameter"].values, "Sub-factor": W["subfactor"].values,
            "Communality h": W["communality"].values,
            "Loading weight %": W["loading_weight"].values * 100,
            "Entropy weight %": W["entropy_weight"].values * 100,
            "Combined %": W["combined_weight"].values * 100,
            "Parameter weight % (bootstrap mean)": W["final_weight"].values * 100,
            "95% low %": W["ci_lo"].values * 100, "95% high %": W["ci_hi"].values * 100,
            "Share in sub-factor %": W["within_sub_weight"].values * 100,
            "Sub-factor weight %": W["subfactor_weight"].values * 100,
            "Effective weight %": W["effective_weight"].values * 100})
        st.dataframe(t.sort_values("Effective weight %", ascending=False).round(2), hide_index=True, width="stretch")
        fig = px.bar(t, x="Effective weight %", y="Parameter", color="Sub-factor", orientation="h",
                     height=max(420, 30 * p_ + 130), color_discrete_sequence=T.COLORWAY,
                     title="Effective weight of each parameter in the factor score (adds up to 100%)")
        fig.update_traces(texttemplate="%{x:.1f}%", textposition="outside", cliponaxis=False,
                          textfont=dict(size=10))
        fig.update_yaxes(autorange="reversed", categoryorder="array", categoryarray=list(t["Parameter"]))
        fig.update_xaxes(range=[0, float(t["Effective weight %"].max()) * 1.2])
        fig.update_layout(bargap=0.25, margin=dict(t=90))
        T.show(fig)

    # ---- 9 sub-factors
    with st.expander("9 · Sub-factor scores and explained-variance weights", expanded=True):
        T.stage_note("Each sub-factor score is the weighted average of its parameters (weights re-scaled to add up to 1 "
                     "inside the sub-factor). Sub-factors are then weighted by how much of the shared pattern "
                     "among sub-factors each one carries (PCA / explained variance).",
                     "The sub-factor weights combine the sub-factor scores into the factor score (stage 10).")
        st.latex(r"S_s=\frac{\sum_{j\in s} w_j x_j}{\sum_{j\in s} w_j}\qquad "
                 r"W_s\;\propto\;\sum_{k:\,\lambda_k>1}\text{EVR}_k\, v_{sk}^{2},\quad "
                 r"\text{EVR}_k=\frac{\lambda_k}{\sum\lambda}")
        st.markdown("PCA is run on the standardised sub-factor scores. Sub-factors that load heavily on the "
                    "components carrying most of the variance get more weight; the weights add up to 100%.")
        c1, c2 = st.columns(2)
        melt = r.sub_scores.melt(var_name="Sub-factor", value_name="Score")
        melt["Sub-factor"] = melt["Sub-factor"].map(lambda s: T.wrap(s, 22))
        fb = px.box(melt, x="Score", y="Sub-factor", height=max(380, 62 * len(r.sub_map) + 110),
                    color_discrete_sequence=[T.TEAL], title="Sub-factor scores across properties")
        fb.update_xaxes(range=[0, 100])
        fb.update_yaxes(autorange="reversed")
        T.show(fb, c1)
        sw = (r.sub_weights * 100)
        fw_ = px.bar(x=sw.values, y=[T.wrap(s, 22) for s in sw.index], orientation="h",
                     height=max(380, 62 * len(r.sub_map) + 110), color_discrete_sequence=[T.TEAL],
                     title="Explained-variance weight of each sub-factor",
                     labels={"x": "Weight %", "y": ""})
        fw_.update_traces(texttemplate="%{x:.0f}%", textposition="outside", cliponaxis=False)
        fw_.update_xaxes(range=[0, max(float(sw.max()) * 1.25, 10)])
        fw_.update_yaxes(autorange="reversed")
        fw_.update_layout(bargap=0.3)
        T.show(T.style_bars(fw_), c2)
        st.dataframe(r.sub_scores.head(25).round(1), width="stretch")

    # ---- 10 factor score
    with st.expander("10 · Factor score and confidence interval", expanded=True):
        T.stage_note("Factor score = sum of (sub-factor weight × sub-factor score), on a 0-100 scale. The confidence "
                     "interval comes from applying each of the bootstrap weight sets to every property and taking "
                     f"the {a:.1f}th and {100 - a:.1f}th percentiles of the resulting scores.",
                     "The factor scores feed the overall Wellness Score (Overall score tab); the bootstrap scores feed "
                     "its confidence interval.")
        tbl = pd.DataFrame({"Factor score": r.score, "CI low": lo, "CI high": hi}).round(2)
        tbl["Band"] = tbl["Factor score"].map(we.band)
        c1, c2 = st.columns([1, 1])
        fig = px.histogram(tbl, x="Factor score", nbins=20, height=340, title="Distribution of factor scores")
        T.add_band_zones(T.style_hist(fig))
        fig.update_xaxes(range=[0, 100])
        fig.update_yaxes(title="Properties")
        T.show(fig, c1)
        c2.dataframe(T.style_df(tbl.sort_values("Factor score", ascending=False), 2), width="stretch", height=340)
        st.caption("This interval reflects uncertainty in the WEIGHTS only. It does not include measurement error "
                   "in the input data.")

    # ---- 11 trace
    with st.expander("11 · Trace one property end to end", expanded=False):
        T.stage_note("Follows one property through every stage of this factor with real numbers: raw value, what "
                     "preprocessing did to it, its 0-100 score, its weight, and how much it adds to the factor score.",
                     "A check that the maths adds up: the contributions below sum exactly to the factor score.")
        pid_t = st.selectbox("Property", list(r.score.index), key=f"trace_{code}")
        ptab, stab = we.trace_property(r, pid_t)
        ptab_show = ptab.copy()
        ptab_show["Weight inside sub-factor"] *= 100
        ptab_show["Effective weight"] *= 100
        ptab_show = ptab_show.rename(columns={"Weight inside sub-factor": "Weight inside sub-factor %",
                                              "Effective weight": "Effective weight %"})
        st.markdown("**Parameter level** (score × effective weight = contribution)")
        st.dataframe(ptab_show.round(2), hide_index=True, width="stretch")
        stab_show = stab.copy()
        stab_show["Sub-factor weight"] *= 100
        stab_show = stab_show.rename(columns={"Sub-factor weight": "Sub-factor weight %"})
        st.markdown("**Sub-factor level** (sub-factor score × sub-factor weight = contribution)")
        st.dataframe(stab_show.round(2), hide_index=True, width="stretch")
        st.success(f"Sum of contributions = {ptab['Contribution to factor score'].sum():.2f}  =  "
                   f"factor score of {pid_t} = {r.score[pid_t]:.2f}")

# ------------------------------------------------------------------ TAB: overall
with tabs[2]:
    T.section("Overall Wellness Score", f"Weighting: {profile}  ·  {level:.0%} confidence interval")
    st.latex(r"\text{Wellness Score}=\sum_{f=1}^{9} \text{factor weight}_f \times \text{factor score}_f")
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
    fig = px.bar(x=fwp.values, y=[T.wrap(n, 20) for n in fwp.index], orientation="h", height=360,
                 color_discrete_sequence=[T.TEAL], title="Weight of each factor in the overall score",
                 labels={"x": "Factor weight %", "y": ""})
    fig.update_traces(texttemplate="%{x:.1f}%", textposition="outside", cliponaxis=False)
    fig.update_xaxes(range=[0, max(float(fwp.max()) * 1.25, 10)])
    T.show(T.style_bars(fig), left)
    fig = T.add_band_zones(T.style_hist(px.histogram(overall.to_frame("Wellness Score"), x="Wellness Score", nbins=20,
                                                     height=360, title="Distribution of overall scores")))
    fig.update_xaxes(range=[0, 100], title="Wellness Score")
    fig.update_yaxes(title="Properties")
    T.show(fig, right)
    top = out.head(25).reset_index(names="Property")
    fig = go.Figure(go.Scatter(x=top["Property"], y=top["Wellness Score"], mode="markers",
                               marker=dict(size=10, color=[T.BAND_COLORS[b] for b in top["Band"]],
                                           line=dict(color="white", width=1.5)),
                               error_y=dict(type="data", symmetric=False, color="#9AA5A8", thickness=1.4,
                                            array=top[f"CI high ({level:.0%})"] - top["Wellness Score"],
                                            arrayminus=top["Wellness Score"] - top[f"CI low ({level:.0%})"])))
    T.add_band_zones(fig, "y")
    fig.update_layout(height=380, title=f"Top 25 properties with {level:.0%} confidence interval",
                      yaxis_title="Wellness Score", showlegend=False)
    fig.update_xaxes(tickangle=-60, type="category")
    T.show(fig)
    st.dataframe(T.style_df(out), width="stretch", height=400)
    st.download_button("Download all scores (CSV)", to_csv(out), "wellness_scores.csv")

    st.divider()
    T.section("Single property drill-down", "Compare one property against the dataset average")
    pid = st.selectbox("Property", list(F.index))
    cats = [T.wrap(names[c], 14) for c in F.columns]
    vals = list(F.loc[pid].values)
    means = list(F.mean().values)
    fig = go.Figure(go.Scatterpolar(r=vals + vals[:1], theta=cats + cats[:1], fill="toself", name=str(pid),
                                    line=dict(color=T.TEAL, width=2.5), fillcolor="rgba(15,118,110,0.22)"))
    fig.add_trace(go.Scatterpolar(r=means + means[:1], theta=cats + cats[:1],
                                  name="Dataset mean", line=dict(dash="dash", color=T.CORAL, width=2)))
    fig.update_layout(polar=dict(bgcolor="rgba(255,255,255,0.6)",
                                 radialaxis=dict(range=[0, 100], gridcolor=T.LINE, linecolor=T.LINE),
                                 angularaxis=dict(gridcolor=T.LINE, tickfont=dict(size=11))),
                      height=460, margin=dict(l=90, r=90, t=70, b=40))
    d1, d2 = st.columns([1, 1])
    T.show(fig, d1)
    with d2:
        T.show(T.gauge(overall[pid], f"Wellness Score · {pid}", 210))
        st.markdown(f"Band: {T.badge(we.band(overall[pid]))} &nbsp; "
                    f"<span style='color:{T.MUTED}'>CI {o_lo[pid]:.1f} – {o_hi[pid]:.1f}</span>",
                    unsafe_allow_html=True)
    st.markdown("#### Why this score?")
    st.markdown(we.explain_property(results, F, fw, overall, o_lo, o_hi, pid, profile, level))

# ------------------------------------------------------------------ TAB: score a new property
with tabs[3]:
    T.section("Score a new property",
              "Find out where a property that is NOT in your dataset would stand")
    with st.expander("What is this tab for?", expanded=True):
        st.markdown(f"""
The model was **built (fitted) on your reference dataset** in the Data tab: that is where the medians, outlier
fences, scaling ranges and all the weights come from. This tab uses that finished model to score a property you are
considering, for example a flat you are about to buy, **without changing the model**.

**What happens when you submit values**
1. You enter the property's raw values (or upload a file with several properties).
2. They go through the **same preprocessing** as the reference data: a blank becomes the reference median, extreme values are clipped to the same fences, values are scaled with the same min / max, and "lower is better" parameters are flipped. A value outside the reference range is capped at 0 or 100.
3. The **same weights** produce sub-factor scores, factor scores and the overall score (using the overall weighting chosen in the sidebar: **{profile}**).
4. The **confidence interval** comes from the same bootstrap weight sets.

**How to read the result:** a score of 70 means the property sits at 70 on the 0-100 scale defined by the reference
set. With default (data-based) scaling that is *relative to those reference properties*; set benchmark Scale min / max
in the Data tab for absolute scoring.
""")
    mode = st.radio("How do you want to provide the property?",
                    ["Type the values for one property", "Upload a file with several properties"], horizontal=True)
    new_df = None
    if mode.startswith("Type"):
        st.caption("Every box starts at the reference **median**, a neutral value that neither helps nor hurts. "
                   "Change only the parameters you know; open each wellness factor below.")
        vals, changed, total = {}, 0, 0
        for c_code, r_ in results.items():
            ns, cs_ = r_.norm_stats, r_.clean_stats
            t = pd.DataFrame({
                "Parameter": [short(c) for c in r_.columns],
                "Sub-factor": [next(s for s, cl in r_.sub_map.items() if c in cl) for c in r_.columns],
                "Better when": [we.DIRECTION_LABEL[d] for d in ns["direction"]],
                "Reference min": ns["min"].values, "Reference median": cs_["median"].values,
                "Reference max": ns["max"].values, "Your value": cs_["median"].values})
            with st.expander(f"{r_.name}  [{c_code}]  ·  {len(r_.columns)} parameters"):
                ed = st.data_editor(
                    t, key=f"man_{c_code}", hide_index=True, width="stretch",
                    disabled=["Parameter", "Sub-factor", "Better when", "Reference min", "Reference median",
                              "Reference max"],
                    column_config={"Your value": st.column_config.NumberColumn("Your value", help="Raw value, same unit as the reference columns."),
                                   "Reference min": st.column_config.NumberColumn(format="%.2f"),
                                   "Reference median": st.column_config.NumberColumn(format="%.2f"),
                                   "Reference max": st.column_config.NumberColumn(format="%.2f")})
                vals.update(dict(zip(r_.columns, pd.to_numeric(ed["Your value"], errors="coerce"))))
                changed += int((~np.isclose(ed["Your value"].astype(float), t["Reference median"].astype(float), equal_nan=True)).sum())
                total += len(t)
        st.info(f"You have changed **{changed} of {total}** values from the neutral median.")
        new_df = pd.DataFrame([vals], index=["New property"])
    else:
        nf = st.file_uploader("New properties (same columns as the template)", type=["csv", "xlsx", "xls"], key="newfile")
        if nf is not None:
            new_df = with_index(read_upload(nf))
    if new_df is not None:
        new = we.score_new(results, new_df, fw, level)
        Fn, on = new["F"], new["overall"]
        weak = new["completeness"][new["completeness"] < 0.7]
        if len(weak):
            st.warning(f"{len(weak)} propert{'y has' if len(weak) == 1 else 'ies have'} fewer than 70% of the parameters filled in "
                       "(missing values are replaced by the reference median, which pulls scores towards the middle): "
                       + ", ".join(map(str, weak.index[:8])))
        oor = new["out_of_range"][new["out_of_range"] > 0]
        if len(oor):
            st.info("Some values lie outside the reference range and were capped at 0 or 100: "
                    + ", ".join(f"{i} ({int(v)})" for i, v in oor.head(8).items()))
        res_tbl = Fn.rename(columns=names).round(1)
        res_tbl["Wellness Score"] = on.round(1)
        res_tbl[f"CI low ({level:.0%})"] = new["lo"].round(1)
        res_tbl[f"CI high ({level:.0%})"] = new["hi"].round(1)
        res_tbl["Band"] = on.map(we.band)
        res_tbl["Parameters supplied %"] = (new["completeness"] * 100).round(0)
        st.dataframe(T.style_df(res_tbl), width="stretch")
        sel = st.selectbox("Property to explain", list(Fn.index), key="new_sel")
        g1, g2 = st.columns([1, 1])
        T.show(T.gauge(on[sel], f"Wellness Score · {sel}", 230), g1)
        with g2:
            st.markdown(f"Band: {T.badge(we.band(on[sel]))} &nbsp; "
                        f"<span style='color:{T.MUTED}'>CI {new['lo'][sel]:.1f} – {new['hi'][sel]:.1f}</span>",
                        unsafe_allow_html=True)
            cats = [T.wrap(names[c], 14) for c in Fn.columns]
            v = list(Fn.loc[sel].values)
            m = [results[c].score.mean() for c in Fn.columns]
            fig = go.Figure(go.Scatterpolar(r=v + v[:1], theta=cats + cats[:1], fill="toself", name=str(sel),
                                            line=dict(color=T.TEAL, width=2.5), fillcolor="rgba(15,118,110,0.22)"))
            fig.add_trace(go.Scatterpolar(r=m + m[:1], theta=cats + cats[:1], name="Reference mean",
                                          line=dict(dash="dash", color=T.CORAL, width=2)))
            fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100], gridcolor=T.LINE),
                                         angularaxis=dict(tickfont=dict(size=10))),
                              height=360, margin=dict(l=80, r=80, t=60, b=30))
            T.show(fig)
        st.markdown(we.explain_new_property(results, new, fw, sel, level))
        rows = [(names[c], s, v_) for c in results for s, v_ in new["subs"][c].loc[sel].items()]
        sub_df = pd.DataFrame(rows, columns=["Factor", "Sub-factor", "Score"]).sort_values("Score")
        with st.expander(f"All sub-factor scores for {sel}"):
            st.dataframe(sub_df.round(1), hide_index=True, width="stretch")
        st.download_button("Download results (CSV)", to_csv(res_tbl), "new_property_scores.csv")

# ------------------------------------------------------------------ TAB: explanation
with tabs[4]:
    T.section("How was the score derived?", "Plain-language walk-through of the calculation, with your numbers")
    scope = st.radio("Explain", ["Selected factor (from the Factor pipeline tab)", "Overall score for a property"],
                     horizontal=True)
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
                    st.markdown(we.llm_explain(text, tok, groq_model, q))
                except Exception as e:
                    st.error(f"LLM call failed: {e}\n\nCheck the API key, your Groq rate limits, "
                             "or try another model name (e.g. llama-3.1-8b-instant).")

# ------------------------------------------------------------------ TAB: method
FLOW = r'''
digraph G {
  rankdir=TB; nodesep=0.35; ranksep=0.40; bgcolor="transparent"; compound=true;
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#0F766E",
        fillcolor="#E6F3F1", fontcolor="#134E4A", margin="0.22,0.10"];
  edge [color="#6B7780", arrowsize=0.7, fontname="Helvetica", fontsize=10, fontcolor="#6B7780"];
  subgraph cluster_f {
    label="Repeated separately for each of the 9 wellness factors"; fontname="Helvetica"; fontsize=12;
    fontcolor="#134E4A"; style="rounded,dashed"; color="#0F766E";
    raw  [label="1  Raw parameters\none column per parameter"];
    pre  [label="2  Data preprocessing\nmissing value → column median\nouliers → clipped to Tukey fences\nrescale 0–100, flip 'lower is better'", fillcolor="#FBF0D2", color="#C58B16"];
    cor  [label="3  Correlation matrix\ndo parameters move together?"];
    fa   [label="4-5  Factor analysis\nhidden factors → loadings → communality h\nloading weight = h / Σh"];
    ent  [label="6  Entropy weight\nhow much a parameter differs\nbetween properties"];
    comb [label="Combined weight\nloading^α × entropy^(1−α), normalised"];
    boot [label="7  Bootstrap validation\nrepeat 3-6 on B resampled datasets\nmean = parameter weight · spread = stability", fillcolor="#FBE3DA", color="#D9684A"];
    pw   [label="8  Parameter weights\n→ effective weight (sums to 100%)"];
    sub  [label="9  Sub-factor scores\nweighted mean of its parameters"];
    ev   [label="9b  Explained-variance weights\nPCA on the sub-factor scores"];
    fs   [label="10  Factor score (0–100)\n+ confidence interval", fillcolor="#CCE8E3"];
    raw -> pre -> cor -> fa -> comb;
    pre -> ent -> comb;
    comb -> boot -> pw -> sub -> ev -> fs;
  }
  nine [label="9 factor scores"];
  wts  [label="Factor weights\nEqual  ·  Data-driven  ·  Custom", fillcolor="#FBF0D2", color="#C58B16"];
  ov   [label="Overall Wellness Score\nΣ factor weight × factor score", fillcolor="#134E4A", fontcolor="white", color="#134E4A"];
  ci   [label="Confidence interval\npercentiles of the bootstrap scores", fillcolor="#CCE8E3"];
  fs -> nine -> ov; wts -> ov; ov -> ci;
}
'''

with tabs[5]:
    T.section("Method", "Every step, why it is done, the formula, and a small example you can check by hand")
    st.markdown("### The big picture")
    st.markdown("""
Each of the 9 wellness factors (Emotional, Physical, Social, Occupational, **Environmental**, Intellectual, Spiritual,
Financial, Statutory) contains **sub-factors**, and each sub-factor contains **parameters** (raw measurements).
The score is built from the bottom up:
**parameters → sub-factor scores → factor scores → overall Wellness Score**. The hard part is deciding *how much
each parameter and sub-factor should count*; those weights are derived from the data rather than guessed.
""")
    try:
        st.graphviz_chart(FLOW)
    except Exception:
        st.code(FLOW, language="dot")

    st.markdown("### Key ideas in plain language")
    with st.expander("Why not just average the parameters?"):
        st.markdown("""
A simple average treats every parameter as equally informative. But a parameter that is almost the same in every
property tells you nothing, and two parameters that are near-copies of each other count the same idea twice. Data-driven
weights reduce both problems: **factor analysis** rewards parameters that represent their group's common theme, and
**entropy** rewards parameters that actually separate one property from another.
""")
    with st.expander("What is factor analysis?"):
        st.markdown("""
It looks for a few **hidden themes** behind many columns. If AQI, pollutants and ventilation always rise and fall
together, they are probably three views of one theme (air quality). The **loading** says how tightly a column is tied
to a theme; the **communality h** (sum of squared loadings) is the share of that column's variation explained by the
themes. A column with high h is a good representative of its group, so it gets more weight. **KMO** and **Bartlett's
test** check the data are suited to this; **eigenvalues** (Kaiser rule: keep > 1) decide how many themes exist.
""")
    with st.expander("What is entropy weighting?"):
        st.markdown("""
Entropy (from information theory) measures *how evenly spread* a parameter's total is across the properties.
If every property scores almost the same on a parameter, its entropy is close to 1 (maximum), and it cannot rank
anything, so its weight is near 0. If scores differ a lot, entropy is lower and the weight is higher.
It is purely about **variation**, not about importance to a buyer.
""")
    with st.expander("What is the bootstrap, and why is it used?"):
        st.markdown("""
Weights computed from one dataset are partly luck of the sample. The **bootstrap** re-draws the properties *with
replacement* (some appear twice, some not at all) many times and recomputes the weights each time. The **average**
is a steadier final weight, and the **spread** shows how much the weight depends on the sample. The same bootstrap
weight sets produce the **confidence interval** of every score.
""")
    with st.expander("What is explained-variance weighting (used for sub-factors and the data-driven overall option)?"):
        st.markdown("""
PCA finds the directions along which the scores vary most. Each direction has an eigenvalue λ and a share of
variance EVR = λ ÷ Σλ. Components with λ > 1 are kept, and each item's weight is the sum over those components of
EVR × (its squared loading on that component), normalised to 100%. Items that carry more of the shared pattern
get more weight.
""")

    st.markdown("### Stage-by-stage")
    st.markdown("""
**Stage 2 · Data preprocessing** (in this order)  
*A. Missing values* → median of the column.  *B. Outliers* → clipped to Q1 − 1.5·IQR and Q3 + 1.5·IQR.  
*C. Scaling* → `s = 100·(x − min)/(max − min)`, kept within 0-100 (min/max = your benchmark or the data's range).  
*D. Direction* → for "lower is better", `score = 100 − s`. After this, **100 always means best**.
""")
    st.markdown("""
**Stages 3-6 · Weights from the data**  
Pearson correlation matrix → number of hidden factors by the Kaiser rule (capped at 4 and at one-third of the
parameter count) → factor analysis (varimax if more than one factor) → communality `h_j = Σ_k λ_jk²` →
loading weight `h_j / Σh` and entropy weight (formula in the pipeline) → combined weight proportional to
`(loading weight)^α × (entropy weight)^(1−α)`.
""")
    st.markdown("""
**Stage 7 · Bootstrap**  
Draw n properties with replacement, redo stages 3-6 (and the sub-factor weights), repeat B times. Parameter weight =
mean over the B runs (renormalised). Interval = percentiles of the B weights. CV = std ÷ mean.

**Stages 8-10 · Scores**  
Sub-factor score `S_s = Σ w_j x_j / Σ w_j` (inside the sub-factor) → sub-factor weights `W_s` by explained variance →
factor score `Σ W_s S_s`. Equivalent form: factor score `= Σ effective_j · x_j`, where `effective_j = W_s · w_j / Σ_{k∈s} w_k`.
These effective weights add up to 100%.

**Overall Wellness Score** = Σ factor weight × factor score, with weights from one of three options:
- **Equal weights:** every factor 1/9.
- **Data-driven (explained variance):** PCA on the 9 factor scores, same formula as for sub-factors.
- **Custom:** your sliders, rescaled to add up to 100%.

**Confidence interval** (factor and overall): for each of the B bootstrap weight sets, score every property; the interval is the
percentile range of those B scores. It always contains the point score.
""")

    st.markdown("### Worked example you can check by hand")
    st.caption("A tiny 'Environmental' factor: 8 properties, 4 parameters, 2 sub-factors. All numbers below are computed "
               "live by the same functions that process your real data.")
    toy = cached_toy(float(alpha))
    cols = list(toy["raw"].columns)
    cst = toy["pre"]["clean_stats"]
    st.markdown("**Step 1 · Raw data** (P3 has no AQI reading; P7's AQI of 410 looks like a sensor error)")
    st.dataframe(toy["raw"], width="stretch")
    st.markdown("**Step 2A · Missing values.** "
                f"AQI median of the 7 observed readings = **{cst.loc['AQI value', 'median']:.0f}**, so P3's AQI becomes "
                f"{cst.loc['AQI value', 'median']:.0f}.")
    st.markdown("**Step 2B · Outliers.** For AQI: Q1 = "
                f"{cst.loc['AQI value', 'q1']:.2f}, Q3 = {cst.loc['AQI value', 'q3']:.2f}, IQR = "
                f"{cst.loc['AQI value', 'q3'] - cst.loc['AQI value', 'q1']:.2f}; upper fence = Q3 + 1.5·IQR = "
                f"**{cst.loc['AQI value', 'hi']:.2f}**. P7's 410 exceeds it and is clipped to {cst.loc['AQI value', 'hi']:.2f}.")
    st.dataframe(cst[["median", "q1", "q3", "lo", "hi", "missing", "outliers_clipped"]]
                 .rename(columns={"median": "Median", "q1": "Q1", "q3": "Q3", "lo": "Lower fence", "hi": "Upper fence",
                                  "missing": "Missing", "outliers_clipped": "Clipped"}).round(2), width="stretch")
    cl = toy["pre"]["cleaned"]
    sc_ = toy["pre"]["scaled"]
    x_ = cl.loc["P1", "AQI value"]
    mn_, mx_ = cl["AQI value"].min(), cl["AQI value"].max()
    st.markdown(f"**Step 2C-D · Scale and flip.** AQI of P1 = {x_:.0f}; cleaned AQI ranges {mn_:.0f} → {mx_:.2f}. "
                f"Scaled = 100 × ({x_:.0f} − {mn_:.0f}) ÷ ({mx_:.2f} − {mn_:.0f}) = {sc_.loc['P1', 'AQI value']:.1f}. "
                f"AQI is *lower is better*, so score = 100 − {sc_.loc['P1', 'AQI value']:.1f} = "
                f"**{toy['pre']['Xn'].loc['P1', 'AQI value']:.1f}**. (Noise is also flipped; green cover is not.)")
    st.dataframe(toy["pre"]["Xn"].round(1), width="stretch")
    st.markdown("**Step 3-5 · Correlation, factor analysis, communality.** "
                f"Eigenvalues of the correlation matrix: {', '.join(f'{v:.2f}' for v in toy['eig'])}; only the first is above 1, "
                f"so **{toy['k']} hidden factor** is kept. Loadings (sign is irrelevant) and communality h = loading²:")
    ldf = toy["loadings"].copy()
    ldf["Communality h"] = toy["weights"]["Communality h"]
    ldf["Loading weight = h / Σh"] = toy["weights"]["Loading weight"]
    st.dataframe(ldf.round(3), width="stretch")
    st.markdown("**Step 6 · Entropy weights.** Each column of scores is turned into shares *p* that add to 1, "
                "then `e = −Σ p ln p / ln 8` and `d = 1 − e`:")
    st.dataframe(toy["shares"].round(4), width="stretch")
    ent = toy["weights"][["Entropy e", "Divergence d = 1 − e", "Entropy weight"]]
    st.dataframe(ent.round(4), width="stretch")
    a_ = toy["alpha"]
    wl, we_ = toy["weights"]["Loading weight"], toy["weights"]["Entropy weight"]
    prod = (wl ** a_) * (we_ ** (1 - a_))
    st.markdown(f"**Combined weight** with α = {a_:.2f}: multiply `loading^α × entropy^(1−α)`, then divide by the total.")
    st.dataframe(pd.DataFrame({"Loading weight": wl, "Entropy weight": we_, "loading^α × entropy^(1−α)": prod,
                               "Combined (normalised)": prod / prod.sum()}).round(4), width="stretch")
    st.markdown(f"**Step 7 · Bootstrap.** The data were resampled {toy['B']} times; each resample gives a slightly different "
                "weight set (first six shown). With only 8 properties the weights move a lot, which is exactly what the "
                "bootstrap is designed to reveal. The final parameter weight is the mean over all resamples.")
    st.dataframe(toy["boot"].round(3), width="stretch")
    tw = toy["weights"][["Bootstrap mean (parameter weight)", "Effective weight"]].copy()
    tw["Sub-factor"] = [next(s for s, v in toy["groups"].items() if c in v) for c in tw.index]
    st.markdown("**Step 8 · Parameter weights → effective weights.** Effective = sub-factor weight × (parameter weight ÷ "
                "total parameter weight of its sub-factor). They add up to 100%.")
    st.dataframe((tw.assign(**{"Bootstrap mean (parameter weight)": tw["Bootstrap mean (parameter weight)"] * 100,
                               "Effective weight": tw["Effective weight"] * 100})
                  .rename(columns={"Bootstrap mean (parameter weight)": "Parameter weight %",
                                   "Effective weight": "Effective weight %"})).round(2), width="stretch")
    xn = toy["pre"]["Xn"]
    pw_ = toy["weights"]["Bootstrap mean (parameter weight)"]
    a1, a2 = toy["groups"]["Air Quality"]
    s1 = (pw_[a1] * xn.loc["P1", a1] + pw_[a2] * xn.loc["P1", a2]) / (pw_[a1] + pw_[a2])
    st.markdown(f"**Step 9 · Sub-factor score** for P1, Air Quality = ({pw_[a1]:.3f}×{xn.loc['P1', a1]:.1f} + "
                f"{pw_[a2]:.3f}×{xn.loc['P1', a2]:.1f}) ÷ ({pw_[a1]:.3f}+{pw_[a2]:.3f}) = **{s1:.1f}**. "
                f"Sub-factor weights by explained variance: "
                f"{', '.join(f'{k} {v * 100:.0f}%' for k, v in toy['sub_weights'].items())}. "
                "*With only two sub-factors the explained-variance weights are always 50/50 (a 2×2 correlation matrix has "
                "eigenvectors (0.71, ±0.71)); real factors have 3-8 sub-factors, so their weights differ.*")
    st.dataframe(toy["sub_scores"].round(1), width="stretch")
    st.markdown(f"**Step 10 · Factor score** for P1 = {toy['sub_weights'].iloc[0]:.2f}×{toy['sub_scores'].loc['P1'].iloc[0]:.1f} + "
                f"{toy['sub_weights'].iloc[1]:.2f}×{toy['sub_scores'].loc['P1'].iloc[1]:.1f} = "
                f"**{toy['score']['P1']:.1f}**.")
    st.dataframe(toy["score"].rename("Factor score").round(1).to_frame().T, width="stretch")

    st.markdown("### Known limitations (read before trusting a number)")
    st.markdown("""
- **Min-max scaling is relative** to the reference set unless you set benchmark Scale min / max: a "70" means "better
  than most of these properties", not an absolute standard.
- The **confidence interval reflects weighting uncertainty only**, not measurement error in sensors, surveys or GIS inputs.
- **Data-driven weights describe what varies and co-moves in the data, not what matters to a buyer.** Use *Custom*
  factor weights, or expert weights (e.g. AHP), to express preferences.
- **Factor analysis needs enough rows** (about 5-10 per parameter) and correlated parameters; check the flags.
- **Overlap between factors:** some parameters measure similar things in different factors (air quality in Physical
  and Environmental, noise in Emotional / Environmental / Spiritual), so the overall score can count one idea more
  than once. Consider removing duplicates from the hierarchy if that matters.
- **Median filling** pulls incomplete properties towards the middle; heavily incomplete rows should be treated with caution.
- **Synthetic data** are random with built-in structure; their scores only test the pipeline.
""")
