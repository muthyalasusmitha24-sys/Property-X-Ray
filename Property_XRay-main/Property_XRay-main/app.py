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
       "A transparent, statistically validated score for how well a property supports everyday living: "
       "from raw parameters to one score with a confidence range. Every step is explained and every number can be traced.",
       ["Data preprocessing", "Correlation", "Factor analysis", "Entropy", "Bootstrap", "Weights",
        "Sub-factor scores", "Factor scores", "Wellness Score + range"])

DIRECTION_LABEL = {"higher": "Higher is better", "lower": "Lower is better"}
DIRECTION_INV = {v: k for k, v in DIRECTION_LABEL.items()}


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


def fshort(name):
    return name.replace(" Wellness", "")


PROFILE_OPTIONS = list(we.BUYER_PROFILES) + ["Data-driven (explained variance)", "Custom"]

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    T.sidebar_brand()
    st.subheader("Settings")
    src = st.radio("Data source", ["Synthetic sample", "Upload file"],
                   help="Synthetic sample = random demo properties so you can try the app. Upload file = your own properties.")
    if src == "Synthetic sample":
        n_rows = st.number_input("Rows (properties)", 30, 1000, 100, 10)
        seed_data = st.number_input("Data seed", 0, 9999, 42, help="Change it to generate a different random sample.")
    alpha = st.slider("Weight blend α  (Entropy ⟷ Factor analysis)", 0.0, 1.0, 0.5, 0.05,
                      help="Each parameter's weight is built from two methods. α = 0 uses only the Entropy weight, "
                           "α = 1 uses only the Factor-analysis (loading) weight, 0.5 mixes them equally. "
                           "See the Method tab, step 6.")
    st.caption("0 = Entropy only · 0.5 = equal mix (default) · 1 = Factor analysis only")
    B = st.slider("Bootstrap resamples", 20, 400, 100, 20,
                  help="How many times the weights are re-calculated on random resamples of the properties. "
                       "More = more stable weights and ranges, but slower (cached after the first run).")
    level = st.selectbox("Confidence level", [0.90, 0.95, 0.99], index=1, format_func=lambda v: f"{v:.0%}",
                         help="Width of the score range shown next to each score. 95% is the usual choice.")
    st.divider()
    st.subheader("How should the 9 factors be weighted?")
    profile = st.selectbox("Overall weighting", PROFILE_OPTIONS,
                           help="Decides how much each of the 9 wellness factors counts in the final Wellness Score. "
                                "Open 'What do the weighting options mean?' in the Overall score tab for the full table.")
    st.caption(we.PROFILE_INFO[profile])
    if profile in we.BUYER_PROFILES:
        st.caption(we.profile_headline(profile))
    custom_w = {}
    if profile == "Custom":
        for code_, f_ in we.HIERARCHY.items():
            custom_w[code_] = st.slider(f_["name"], 0.0, 3.0, 1.0, 0.1)
    st.divider()
    st.subheader("AI explanation (Groq)")
    if not (get_token() and not st.session_state.get("groq_key_input")):
        st.text_input("Groq API key (kept only in this session)", type="password", key="groq_key_input",
                      help="Better: put GROQ_API_KEY in Streamlit Secrets or an environment variable.")
    llm_model = st.text_input("Model", "llama-3.3-70b-versatile")

tabs = st.tabs(["Data & settings", "Factor pipeline", "Overall score", "Score a new property", "Explanation", "Method"])

# ------------------------------------------------------------------ TAB 0: data
cfg_base = we.default_config()
with tabs[0]:
    if src == "Synthetic sample":
        df = cached_sample(int(n_rows), int(seed_data))
        st.success(f"Generated {len(df)} synthetic properties × {df.shape[1] - 1} parameter columns "
                   "(a few missing values and outliers were injected on purpose so you can see the data preprocessing at work).")
    else:
        up = st.file_uploader("Upload CSV / Excel (one row per property, one column per parameter)",
                              type=["csv", "xlsx", "xls"])
        tmpl = we.generate_sample_data(3, 1, messy=False).iloc[:0]
        st.download_button("⬇ Download blank template (column names)", to_csv(tmpl.set_index("Property_ID")),
                           "wellness_template.csv")
        if up is None:
            st.info("Upload a file to continue. Column names must match the template "
                    "(`CODE: Parameter`, e.g. `ENV: AQI value`). Missing columns are skipped; "
                    "a factor needs at least 3 columns.")
            st.stop()
        df = read_upload(up)
    if "Property_ID" in df.columns and df["Property_ID"].is_unique:
        df = df.set_index(df["Property_ID"].astype(str)).drop(columns="Property_ID")
    else:
        df = df.drop(columns=[c for c in ["Property_ID"] if c in df.columns])
        df.index = [f"P{i + 1:03d}" for i in range(len(df))]
    T.section("Data preview", "First 20 rows of the reference dataset (one row per property)")
    st.dataframe(df.head(20), width="stretch")

    T.section("Step 1 · Check that your columns were recognised",
              "Every column name looks like  CODE: Parameter  (for example  ENV: AQI value).")
    st.markdown(
        "The **code before the colon** tells the app which wellness factor a parameter belongs to: "
        "**EMO** = Emotional, **PHY** = Physical, **SOC** = Social, **OCC** = Occupational, "
        "**ENV** = Environmental, **INT** = Intellectual, **SPI** = Spiritual, **FIN** = Financial, **STA** = Statutory. "
        "A factor needs at least 3 matching columns to be scored.")
    match = (cfg_base.assign(found=cfg_base["Column"].isin(df.columns))
             .groupby(["Code", "Factor"]).agg(found=("found", "sum"), expected=("found", "size")).reset_index())
    match["Status"] = np.where(match["found"] >= 3, "✓ scored", "✗ skipped (needs ≥ 3 columns)")
    match.columns = ["Code (file prefix)", "Wellness factor", "Columns found", "Columns expected", "Status"]
    st.dataframe(match, hide_index=True, width="stretch")

    T.section("Step 2 · Parameter settings",
              "Tell the app, for each parameter, whether a higher value is good or bad, and (optionally) what range to expect.")
    with st.expander("How to read this table", expanded=True):
        st.markdown("""
Each row is one measurable **parameter** (for example *AQI value*). It belongs to a **sub-factor** (*Air Quality*)
which belongs to a **wellness factor** (*Environmental Wellness*). You can change three things:

| Column | What it means | Example |
|---|---|---|
| **Better when** | Is a **higher** raw number good or bad? If a higher number is *worse* (pollution, noise, crime, price, commute time) choose **Lower is better**; the app then flips the score so that **100 always means best**. | *AQI value*: a higher AQI means dirtier air, so **Lower is better**. *% Green cover*: more is good, so **Higher is better**. |
| **Lowest expected value** | Optional. The smallest value you consider realistic. Leave blank to use the smallest value found in the data. | AQI: **0** |
| **Highest expected value** | Optional. The largest value you consider realistic. Leave blank to use the largest value found in the data. | AQI: **500** |

**Why set the expected range?** The 0–100 scale is stretched between these two values. With a fixed range
(AQI 0–500) the same AQI always gets the same score, even for a new dataset. With a blank range the scale
depends on the properties in your file: a score of 70 then means *"better than most of these properties"*,
not an absolute standard.
""")
    view = cfg_base.copy()
    view["Direction"] = view["Direction"].map(DIRECTION_LABEL)
    edited = st.data_editor(
        view, key="cfg_editor_v3", hide_index=True, width="stretch", height=380,
        column_order=["Factor", "Subfactor", "Parameter", "Direction", "Min", "Max"],
        disabled=["Code", "Factor", "Subfactor", "Parameter", "Column"],
        column_config={
            "Factor": st.column_config.TextColumn("Wellness factor"),
            "Subfactor": st.column_config.TextColumn("Sub-factor"),
            "Parameter": st.column_config.TextColumn("Parameter"),
            "Direction": st.column_config.SelectboxColumn(
                "Better when", options=list(DIRECTION_LABEL.values()), required=True,
                help="Higher is better: more is good (e.g. % green cover). Lower is better: more is bad (e.g. AQI, noise, price)."),
            "Min": st.column_config.NumberColumn("Lowest expected value", help="Optional. Blank = smallest value in the data."),
            "Max": st.column_config.NumberColumn("Highest expected value", help="Optional. Blank = largest value in the data."),
        })
    cfg = cfg_base.copy()
    cfg["Direction"] = edited["Direction"].map(DIRECTION_INV).fillna(cfg_base["Direction"]).values
    cfg["Min"] = edited["Min"].values
    cfg["Max"] = edited["Max"].values
    bad = cfg[cfg["Min"].notna() & cfg["Max"].notna() & (cfg["Min"] >= cfg["Max"])]
    if len(bad):
        st.warning("Lowest expected value must be smaller than the highest expected value for: "
                   + ", ".join(bad["Column"].head(5)) + (" …" if len(bad) > 5 else ""))
    if match["Columns found"].sum() == 0:
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
    pw = we.BUYER_PROFILES[profile]
F, fw, overall, o_lo, o_hi = we.combine_overall(results, pw, level)
names = {c: r.name for c, r in results.items()}

# ------------------------------------------------------------------ TAB 1: pipeline
with tabs[1]:
    default_code = st.session_state.get("sel_factor", "ENV" if "ENV" in results else list(results)[0])
    code = st.selectbox("Wellness factor to inspect", list(results), format_func=lambda c: f"{names[c]}  ({c})",
                        index=list(results).index(default_code) if default_code in results else 0, key="sel_factor")
    r = results[code]
    lo, hi = r.ci(level)
    kmo = r.diagnostics["kmo"]
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Mean factor score", f"{r.score.mean():.1f}")
    m2.metric("Parameters / sub-factors", f"{len(r.columns)} / {len(r.sub_map)}")
    m3.metric("KMO (data suitability)", f"{kmo:.2f}",
              help=f"KMO is {we.kmo_label(kmo)}. It checks whether the parameters share enough variation for factor analysis: above 0.6 is acceptable, below 0.5 is not suitable.")
    m4.metric("Hidden themes kept", r.n_factors)
    m5.metric(f"Avg {level:.0%} range width", f"{(hi - lo).mean():.1f}")
    if r.flags:
        with st.expander(f"{len(r.flags)} possible problem(s) detected", expanded=True):
            for s_, msg in r.flags:
                T.flag(s_, msg)
    st.caption("Read this tab top to bottom: Stage 1 prepares the numbers, Stage 2 decides how much each parameter counts, "
               "Stage 3 turns them into scores. Every expander explains what happens and why.")

    # ============ STAGE 1
    T.stage(1, "Data preprocessing", "Make the raw numbers clean and comparable before any statistics are done")
    cs = r.clean_stats
    n_low = int((r.norm_stats["direction"] == "lower").sum())
    st.markdown(
        "Raw data cannot be compared directly: one parameter is in decibels, another in percent, another in rupees; "
        "some cells are blank and a few values are extreme. Preprocessing therefore does two jobs, in this order: "
        "**(a) cleaning**: fill blanks and tame extreme values, and **(b) scaling**: convert every parameter to a score from 0 to 100 "
        "where 100 is always best. All later stages work only with these 0–100 scores.")
    with st.expander("1.1 · Raw data (as received)", expanded=False):
        st.dataframe(r.raw.rename(columns=short).head(25), width="stretch")
        st.caption("First 25 properties. Blank cells are missing values.")
    with st.expander("1.2 · Cleaning: missing values and outliers", expanded=True):
        st.markdown(f"""
**Missing values → median.** A blank cell is replaced by the **median** of its parameter (the middle value when all
properties are sorted). The median is used instead of the average because one extreme value cannot move it.

**Outliers → capped at the IQR limits.** For each parameter we find the *lower quartile* **Q1** (25% of properties are below it),
the *upper quartile* **Q3** (75% are below it) and **IQR = Q3 − Q1**. Any value below **Q1 − 1.5 × IQR** or above **Q3 + 1.5 × IQR**
is pulled back to that limit (this is the standard *Tukey / IQR rule*; pulling values back instead of deleting them is called *winsorising*).
The limits are computed on the original column, then blanks are filled, then values are capped.

**In this factor:** {int(cs['missing'].sum())} blank cell(s) were filled and {int(cs['outliers_clipped'].sum())} value(s) were capped.
""")
        ct = pd.DataFrame({
            "Missing cells": cs["missing"].astype(int), "Missing %": cs["missing_pct"],
            "Median (fill value)": cs["median"], "Lower limit (Q1 − 1.5×IQR)": cs["lo"],
            "Upper limit (Q3 + 1.5×IQR)": cs["hi"], "Values capped": cs["outliers_clipped"].astype(int)})
        ct.index = [short(c) for c in ct.index]
        st.dataframe(T.style_df(ct, 2), width="stretch")
    with st.expander("1.3 · Scaling every parameter to 0–100", expanded=True):
        st.markdown(f"""
After cleaning, each parameter gets a score from 0 to 100:

> **score = (value − min) ÷ (max − min) × 100**

* **min / max** are your *expected values* from the Data tab if you set them, otherwise the smallest / largest cleaned value in the data.
* Values outside min–max are held at 0 or 100.
* If **lower is better** (noise, AQI, price…), the score is **flipped: 100 − score**, so the best value always scores 100.
* If min = max (the parameter never changes) every property gets 50.

**In this factor:** {n_low} of {len(r.columns)} parameters are *lower is better* and were flipped.
""")
        ns = r.norm_stats
        nt = pd.DataFrame({
            "Better when": ns["direction"].map(DIRECTION_LABEL), "Min used": ns["min"], "Max used": ns["max"],
            "Source of min / max": ns["bounds"].map({"benchmark": "Your expected values", "data": "Data"}),
            "Score at min → at max": np.where(ns["direction"] == "lower", "100 → 0", "0 → 100")})
        nt.index = [short(c) for c in nt.index]
        st.dataframe(T.style_df(nt, 2), width="stretch")
        st.markdown("**Resulting 0–100 scores** (first 25 properties):")
        st.dataframe(T.style_df(r.Xn.rename(columns=short).head(25), 1), width="stretch")
    with st.expander("1.4 · Follow one value through every preprocessing step", expanded=True):
        d_row, d_col = we.find_interesting(r)
        c1_, c2_ = st.columns(2)
        pid_t = c1_.selectbox("Property", list(r.raw.index), index=list(r.raw.index).index(d_row), key=f"tr_p_{code}")
        par_t = c2_.selectbox("Parameter", r.columns, index=list(r.columns).index(d_col), format_func=short, key=f"tr_c_{code}")
        st.dataframe(we.parameter_trace(r, pid_t, par_t), hide_index=True, width="stretch")
        st.caption("The default selection shows a blank or capped value when one exists. Result of step 5 = the score used in all later stages.")

    # ============ STAGE 2
    T.stage(2, "Finding the weights", "Decide how much each parameter should count, and check that the answer is stable")
    st.markdown(
        "Counting every parameter equally would be arbitrary: some parameters repeat the same information, others hardly vary. "
        "Instead the **data suggests the weights**. Two well-known methods look at the data from different angles "
        "(**Factor analysis**: which parameters move together; **Entropy**: which parameters differ most between properties). "
        "Their results are blended, then **bootstrapped** (re-calculated on many resamples) to make sure the weights are stable.")
    groups = [len(v) for v in r.sub_map.values()]
    with st.expander("2.1 · Correlation matrix: which parameters move together?", expanded=True):
        st.markdown("Each square shows the **correlation** between two parameters over all properties: "
                    "**+1** (teal) = they rise and fall together, **0** = unrelated, **−1** (orange) = one rises when the other falls. "
                    "Thick black lines separate the sub-factors, so squares on the diagonal blocks are parameters of the same sub-factor.")
        cm = r.corr.copy()
        cm.index = cm.columns = [short(c) for c in cm.columns]
        T.show(T.heatmap(cm, group_sizes=groups, cbar_title="correlation"))
    with st.expander("2.2 · Factor analysis: finding the hidden themes", expanded=True):
        st.markdown(f"""
**What is factor analysis?** Imagine the {len(r.columns)} parameters of this factor as {len(r.columns)} questions in a survey. Many of them are really
asking about the *same underlying theme*: for example several water-related parameters all reflect "water safety".
Factor analysis studies the correlation matrix above and finds a small number of **hidden themes** that explain why
parameters move together. In the statistics literature these are called *common factors*; here we call them **themes**
so they are not confused with the 9 wellness factors.

* **Number of themes** = how many eigenvalues of the correlation matrix are **greater than 1** (the *Kaiser rule*; at least 1, at most 4 and at most one per 3 parameters). The scree chart below shows the eigenvalues; each teal bar above the dashed line is one theme kept.
* **Varimax rotation** is applied (when there is more than one theme) so each parameter links mainly to one theme, which makes the themes easier to read.
* **KMO** (0–1) checks whether the parameters share enough variation for factor analysis to make sense: **above 0.6 is acceptable**, below 0.5 is not suitable.
* **Bartlett's test** asks whether the correlations differ from zero overall: a **p-value below 0.05** is what we want.
""")
        ev = r.eigenvalues
        xs = list(range(1, len(ev) + 1))
        fig = go.Figure(go.Bar(x=xs, y=ev, marker_color=[T.TEAL if v > 1 else "#B9C9C6" for v in ev], name="Eigenvalue"))
        fig.add_trace(go.Scatter(x=xs, y=ev, mode="lines", line=dict(color=T.TEAL_DARK, width=1.5), showlegend=False))
        fig.add_hline(y=1, line_dash="dash", line_color=T.CORAL, annotation_text="Kaiser line (eigenvalue = 1)",
                      annotation_position="top right", annotation_font_color=T.CORAL)
        fig.update_xaxes(title="Component number", dtick=1)
        fig.update_yaxes(title="Eigenvalue")
        fig.update_layout(height=340, showlegend=False, bargap=0.3)
        T.show(fig)
        bp = r.diagnostics["bartlett_p"]
        st.info(f"**KMO = {kmo:.3f}** ({we.kmo_label(kmo)})  ·  **Bartlett χ² = {r.diagnostics['bartlett_chi2']:.1f}, p = {bp:.2g}** "
                f"({'significant ✓' if bp < 0.05 else 'NOT significant ✗'})  ·  **{r.n_factors} theme(s) kept**")
    with st.expander("2.3 · Factor loadings: how strongly each parameter belongs to a theme", expanded=True):
        st.markdown("A **loading** (−1 to +1) says how strongly a parameter is tied to a theme. "
                    "The **communality** *h* of a parameter is the sum of its squared loadings: the share of its variation that the themes explain "
                    "(0 = not at all, 1 = completely). The **loading weight** of a parameter is its communality divided by the total: "
                    "**loading weight = h ÷ Σh**. Parameters that fit the shared patterns well get a larger loading weight.")
        Lm = r.loadings.copy()
        Lm.index = [short(c) for c in Lm.index]
        T.show(T.heatmap(Lm, group_sizes=None, text=True, cbar_title="loading"))
        lt = pd.DataFrame({"Communality h": r.weights["communality"].values,
                           "Loading weight %": r.weights["loading_weight"].values * 100}, index=[short(c) for c in r.columns])
        st.dataframe(T.style_df(lt, 3), width="stretch")
    with st.expander("2.4 · Two weights compared: Factor analysis vs Entropy", expanded=True):
        st.markdown("""
* **Factor-analysis (loading) weight**: rewards parameters that fit the *shared pattern* of the data.
* **Entropy weight**: rewards parameters whose values *differ most between properties* (a parameter that is almost the same everywhere tells us little).

The **diamond** is the **blend** of the two (step 6 in the Method tab). Where the dots are far apart the two methods disagree; the blend lands in between.
""")
        w = r.weights.sort_values("combined_weight", ascending=False)
        T.show(T.dumbbell(w["parameter"], w["loading_weight"] * 100, w["entropy_weight"] * 100, w["combined_weight"] * 100,
                          "Factor analysis (loading) weight", "Entropy weight", "Blend",
                          "Weight (% of this wellness factor)"))
    with st.expander("2.5 · Bootstrap: are the weights stable?", expanded=True):
        st.markdown(f"""
The properties are **resampled at random (with replacement)** and the weights are re-calculated from scratch. This is repeated **{len(r.boot_w)} times**.
The **dot** is the average weight over all resamples (this becomes the **final weight**) and the **bar** is the {level:.0%} range of the resampled weights.
A short bar means the weight hardly changes whichever properties are drawn, so it is **stable**.
A **orange dot** marks an *unstable* weight (its spread is more than 60% of the weight itself).
""")
        w = r.weights.sort_values("final_weight")
        a_ = (1 - level) / 2 * 100
        if len(r.boot_w):
            blo = pd.Series(np.percentile(r.boot_w, a_, axis=0), index=r.columns)[w.index].values * 100
            bhi = pd.Series(np.percentile(r.boot_w, 100 - a_, axis=0), index=r.columns)[w.index].values * 100
        else:
            blo = bhi = w["final_weight"].values * 100
        fv = w["final_weight"].values * 100
        colors = [T.CORAL if (c_ > 0.6) else T.TEAL for c_ in w["cv"].fillna(0).values]
        fig = go.Figure(go.Scatter(
            x=fv, y=w["parameter"], mode="markers", marker=dict(color=colors, size=10, line=dict(color="white", width=1.5)),
            error_x=dict(type="data", symmetric=False, array=np.maximum(bhi - fv, 0), arrayminus=np.maximum(fv - blo, 0),
                         color="#9AA5A8", thickness=2, width=4),
            hovertemplate="%{y}<br>final weight %{x:.2f}%<extra></extra>"))
        fig.update_yaxes(type="category", categoryorder="array", categoryarray=list(w["parameter"]), dtick=1, tickfont=dict(size=12))
        fig.update_xaxes(title=f"Weight % of this factor (average, with {level:.0%} range)", rangemode="tozero")
        fig.update_layout(height=max(400, 30 * len(w) + 130), showlegend=False)
        T.show(fig)
    with st.expander("2.6 · Final parameter weights and how one weight is built", expanded=True):
        st.markdown("**Final weight** = the bootstrap average of the blended weight. All final weights in a factor add up to 100%.")
        t = r.weights.copy()
        t.index = [short(c) for c in t.index]
        tab = pd.DataFrame({
            "Sub-factor": t["subfactor"], "Better when": t["direction"].map(DIRECTION_LABEL),
            "Loading weight %": t["loading_weight"] * 100, "Entropy weight %": t["entropy_weight"] * 100,
            "Blend %": t["combined_weight"] * 100, "Spread (std) %-pts": t["bootstrap_std"] * 100,
            "95% low %": t["ci_lo"] * 100, "95% high %": t["ci_hi"] * 100, "FINAL weight %": t["final_weight"] * 100})
        st.dataframe(T.style_df(tab.sort_values("FINAL weight %", ascending=False), 2), width="stretch")
        par_w = st.selectbox("Show the calculation for parameter", r.columns, format_func=short, key=f"wt_{code}")
        st.dataframe(we.weight_trace(r, par_w, float(alpha)), hide_index=True, width="stretch")

    # ============ STAGE 3
    T.stage(3, "Building the scores", "Combine parameter scores into sub-factor scores and then into one factor score")
    with st.expander("3.1 · Sub-factor scores and sub-factor weights", expanded=True):
        st.markdown("""
* **Sub-factor score** = *weighted average* of the 0–100 scores of its parameters, using the final parameter weights (rescaled to add up to 100% inside the sub-factor).
* **Sub-factor weight** (how much each sub-factor counts in the factor) comes from **PCA explained variance**: PCA is run on the sub-factor scores, components with eigenvalue > 1 are kept, and a sub-factor's weight is proportional to how much of the kept variance it carries.
""")
        c1_, c2_ = st.columns(2)
        ssl = r.sub_scores.melt(var_name="Sub-factor", value_name="Score")
        fig = px.box(ssl, x="Score", y="Sub-factor", color_discrete_sequence=[T.TEAL])
        fig.update_yaxes(type="category", dtick=1, automargin=True)
        fig.update_layout(height=max(360, 52 * len(r.sub_map) + 100), title="Spread of sub-factor scores (all properties)")
        T.show(fig, c1_)
        sw = (r.sub_weights * 100).sort_values()
        fig = T.style_bars(px.bar(x=sw.values, y=sw.index, orientation="h", labels={"x": "Sub-factor weight %", "y": ""},
                                  color_discrete_sequence=[T.TEAL]))
        fig.update_yaxes(type="category", dtick=1, automargin=True)
        fig.update_layout(height=max(360, 52 * len(r.sub_map) + 100), title="Sub-factor weights (PCA explained variance)")
        T.show(fig, c2_)
        d1_, d2_ = st.columns(2)
        pid_s = d1_.selectbox("Property", list(r.sub_scores.index), key=f"sf_p_{code}")
        sub_s = d2_.selectbox("Sub-factor", list(r.sub_map), key=f"sf_s_{code}")
        st.dataframe(T.style_df(we.subfactor_trace(r, pid_s, sub_s), 2), hide_index=True, width="stretch")
        st.caption("Last row: the sub-factor score = sum of (score × weight inside the sub-factor).")
    with st.expander("3.2 · Factor score and its confidence range", expanded=True):
        st.markdown("**Factor score = Σ (sub-factor weight × sub-factor score)**, a number from 0 to 100. "
                    f"The **{level:.0%} range** comes from re-scoring every property with each of the {len(r.boot_w)} bootstrap weight sets and taking the middle {level:.0%} of the results.")
        tbl = pd.DataFrame({"Factor score": r.score, "Range low": lo, "Range high": hi}).round(2)
        tbl["Band"] = tbl["Factor score"].map(we.band)
        c1_, c2_ = st.columns([1, 1])
        fig = T.add_band_zones(T.style_hist(px.histogram(tbl, x="Factor score", nbins=20, height=340)))
        fig.update_xaxes(range=[0, 100], title="Factor score")
        fig.update_yaxes(title="Number of properties")
        T.show(fig, c1_)
        c2_.dataframe(T.style_df(tbl.sort_values("Factor score", ascending=False), 2), width="stretch", height=340)
        pid_f = st.selectbox("Property", list(r.score.index), key=f"fs_p_{code}")
        st.dataframe(T.style_df(we.factor_trace(r, pid_f), 2), hide_index=True, width="stretch")

# ------------------------------------------------------------------ TAB 2: overall
with tabs[2]:
    T.section("Overall Wellness Score", f"Weighting: {profile}  ·  {level:.0%} confidence range")
    out = F.rename(columns=names).round(1)
    out["Wellness Score"] = overall.round(1)
    out[f"Range low ({level:.0%})"] = o_lo.round(1)
    out[f"Range high ({level:.0%})"] = o_hi.round(1)
    out["Band"] = overall.map(we.band)
    out = out.sort_values("Wellness Score", ascending=False)
    g, k = st.columns([1.1, 2])
    T.show(T.gauge(overall.mean(), "Mean Wellness Score", 250), g)
    with k:
        k1, k2, k3 = st.columns(3)
        k1.metric("Mean score", f"{overall.mean():.1f}")
        k2.metric("Best / worst", f"{overall.max():.1f} / {overall.min():.1f}")
        k3.metric("Avg range width", f"{(o_hi - o_lo).mean():.1f}")
        st.markdown(f"Typical band: {T.badge(we.band(overall.mean()))}", unsafe_allow_html=True)
        st.caption("Bands: below 35 Poor · 35–50 Below average · 50–65 Moderate · 65–80 Good · 80 and above Excellent.")

    with st.expander("What do the weighting options mean?", expanded=False):
        st.markdown("The overall score is **Σ (factor weight × factor score)**. The option you pick in the sidebar decides the 9 factor weights:")
        for nm in PROFILE_OPTIONS:
            extra = f"  \n*{we.profile_headline(nm)}*" if nm in we.BUYER_PROFILES else ""
            st.markdown(f"* **{nm}**: {we.PROFILE_INFO[nm]}{extra}")
        pt = we.profile_table()
        pt.index = [fshort(i) for i in pt.index]
        pt.columns = [c.replace(" (equal weights)", "") for c in pt.columns]
        T.show(T.heatmap(pt, text=True, zmin=0, zmax=float(pt.values.max()), colorscale=T.SEQUENTIAL,
                         cbar_title="weight %", fmt=".1f"))
        st.caption("Each column adds up to 100%. Darker = that factor counts more for that buyer.")

    left, right = st.columns([1, 1])
    fwp = (fw * 100).rename(index=lambda c: fshort(names[c])).sort_values()
    fig = T.style_bars(px.bar(x=fwp.values, y=fwp.index, orientation="h", color_discrete_sequence=[T.TEAL],
                              labels={"x": f"Factor weight % ({profile})", "y": ""}))
    fig.update_yaxes(type="category", dtick=1, automargin=True)
    fig.update_layout(height=360, title="Factor weights in use")
    T.show(fig, left)
    fig = T.add_band_zones(T.style_hist(px.histogram(overall, nbins=20, labels={"value": "Wellness Score"})))
    fig.update_xaxes(range=[0, 100], title="Wellness Score")
    fig.update_yaxes(title="Number of properties")
    fig.update_layout(height=360, title="Distribution of Wellness Scores")
    T.show(fig, right)

    show_n = st.radio("Properties to show in the range chart", ["Top 25", "Bottom 25"], horizontal=True)
    sel = out.head(25) if show_n == "Top 25" else out.tail(25).iloc[::-1]
    fig = T.range_plot(sel.index, sel["Wellness Score"], sel[f"Range low ({level:.0%})"], sel[f"Range high ({level:.0%})"],
                       sel["Band"], level)
    fig.update_layout(title=f"{show_n} properties: score with {level:.0%} range (dot colour = band)")
    T.show(fig)
    st.dataframe(T.style_df(out), width="stretch", height=400)
    st.download_button("⬇ Download all scores (CSV)", to_csv(out), "wellness_scores.csv")

    st.divider()
    T.section("Single property drill-down", "See exactly how one property's score is built")
    pid = st.selectbox("Property", list(F.index))
    cats = [fshort(names[c]) for c in F.columns]
    vals = list(F.loc[pid].values)
    fig = go.Figure(go.Scatterpolar(r=vals + vals[:1], theta=cats + cats[:1], fill="toself", name=str(pid),
                                    line=dict(color=T.TEAL, width=2.5), fillcolor="rgba(15,118,110,0.22)"))
    fig.add_trace(go.Scatterpolar(r=list(F.mean().values) + [F.mean().values[0]], theta=cats + cats[:1],
                                  name="Dataset mean", line=dict(dash="dash", color=T.CORAL, width=2)))
    fig.update_layout(polar=dict(bgcolor="rgba(255,255,255,0.6)",
                                 radialaxis=dict(range=[0, 100], gridcolor=T.LINE, linecolor=T.LINE, tickfont=dict(size=10)),
                                 angularaxis=dict(gridcolor=T.LINE, tickfont=dict(size=12))),
                      height=440, margin=dict(l=70, r=70, t=40, b=40))
    d1, d2 = st.columns([1, 1])
    T.show(fig, d1)
    with d2:
        T.show(T.gauge(overall[pid], f"Wellness Score · {pid}", 220))
        st.markdown(f"Band: {T.badge(we.band(overall[pid]))} &nbsp; "
                    f"<span style='color:{T.MUTED}'>{level:.0%} range {o_lo[pid]:.1f} – {o_hi[pid]:.1f}</span>",
                    unsafe_allow_html=True)
        st.caption(f"Scores {'above' if overall[pid] >= overall.mean() else 'below'} the dataset mean of {overall.mean():.1f}.")
    st.markdown("#### How the score builds up")
    st.caption("Each bar = factor score × factor weight. The bars add up to the Wellness Score.")
    contrib = (F.loc[pid] * fw).rename(index=lambda c: fshort(names[c]))
    T.show(T.waterfall(contrib, overall[pid]))
    with st.expander("Show the calculation as a table", expanded=False):
        st.dataframe(T.style_df(we.overall_trace(F, fw, names, pid), 2), hide_index=True, width="stretch")
        st.caption("Small differences in the last digit are rounding.")
    st.markdown("#### Why this score?")
    st.markdown(we.explain_property(results, F, fw, overall, o_lo, o_hi, pid, profile, level))

# ------------------------------------------------------------------ TAB 3: new property
with tabs[3]:
    T.section("Score a new property", "See how a property that is NOT in your dataset would score")
    with st.expander("What does this tab do?", expanded=True):
        st.markdown(f"""
The dataset you loaded in the **Data & settings** tab is the **reference set**. From it the app *learned* everything it needs:
the median and outlier limits of every parameter (preprocessing), the min / max for the 0–100 scale, and all the weights.

This tab lets you **enter the values of one new property** (a house you are thinking of buying, for example) and scores it with that
same, already-learned model, **without re-fitting anything**. So the new property is judged on exactly the same yardstick as the {len(df)} reference properties.

**How to use it**
1. Choose **Type values** (one property) or **Upload a file** (several properties, same column names as the template).
2. Open a factor, then change the values you know. Every box starts at the **typical (median) value** of the reference set, and a blank box is also treated as typical.
3. Read the result: the **Wellness Score**, its **band**, the **{level:.0%} range**, and how it **ranks against the reference properties**.
""")
    mode = st.radio("How do you want to provide the property?", ["Type values for one property", "Upload a file with new properties"],
                    horizontal=True)
    new_df = None
    if mode.startswith("Type"):
        pname = st.text_input("Name of the property", "New property")
        vals = {}
        for c_, r_ in results.items():
            with st.expander(f"{r_.name} ({c_}) · {len(r_.columns)} parameters"):
                base = pd.DataFrame({
                    "Column": r_.columns,
                    "Sub-factor": [next(s_ for s_, cs_ in r_.sub_map.items() if col in cs_) for col in r_.columns],
                    "Parameter": [short(col) for col in r_.columns],
                    "Value": [float(r_.clean_stats.loc[col, "median"]) for col in r_.columns],
                    "Better when": r_.norm_stats["direction"].map(DIRECTION_LABEL).values,
                    "Range in reference data": [f"{r_.cleaned[col].min():,.2f} – {r_.cleaned[col].max():,.2f}" for col in r_.columns]})
                ed = st.data_editor(
                    base, key=f"manual_{c_}", hide_index=True, width="stretch",
                    column_order=["Sub-factor", "Parameter", "Value", "Better when", "Range in reference data"],
                    disabled=["Column", "Sub-factor", "Parameter", "Better when", "Range in reference data"],
                    column_config={"Value": st.column_config.NumberColumn("Value for this property", help="Change to the real value. Blank = typical value.")})
                vals.update(dict(zip(ed["Column"], ed["Value"])))
        new_df = pd.DataFrame([vals], index=[pname.strip() or "New property"])
    else:
        nf = st.file_uploader("New properties (same columns as the template)", type=["csv", "xlsx", "xls"], key="newfile")
        if nf is not None:
            new_df = read_upload(nf)
            if "Property_ID" in new_df.columns:
                new_df = new_df.set_index(new_df["Property_ID"].astype(str)).drop(columns="Property_ID")
    if new_df is not None:
        Fn, on, nlo, nhi, subs = we.score_new(results, new_df, fw, level)
        T.section("Result", "Scored with the model learned from the reference set")
        pick = st.selectbox("Property to explain", list(Fn.index)) if len(Fn) > 1 else Fn.index[0]
        pct = float((overall < on[pick]).mean() * 100)
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Wellness Score", f"{on[pick]:.1f}")
        r2.metric("Band", we.band(on[pick]))
        r3.metric(f"{level:.0%} range", f"{nlo[pick]:.1f} – {nhi[pick]:.1f}")
        r4.metric("Rank vs reference", f"Better than {pct:.0f}%", help=f"Share of the {len(overall)} reference properties with a lower Wellness Score.")
        g1, g2 = st.columns([1, 1.4])
        T.show(T.gauge(on[pick], f"Wellness Score · {pick}", 240), g1)
        contrib_n = (Fn.loc[pick] * fw).rename(index=lambda c: fshort(names[c]))
        fig = T.waterfall(contrib_n, on[pick])
        fig.update_layout(title="How the score builds up (factor score × factor weight)")
        T.show(fig, g2)
        res_tbl = Fn.rename(columns=names).round(1)
        res_tbl["Wellness Score"] = on.round(1)
        res_tbl[f"Range low ({level:.0%})"] = nlo.round(1)
        res_tbl[f"Range high ({level:.0%})"] = nhi.round(1)
        res_tbl["Band"] = on.map(we.band)
        st.dataframe(T.style_df(res_tbl), width="stretch")
        rows = [(names[c], s_, v) for c in results for s_, v in subs[c].loc[pick].items()]
        sub_df = pd.DataFrame(rows, columns=["Factor", "Sub-factor", "Score"]).sort_values("Score")
        with st.expander(f"Sub-factor scores for {pick} (weakest first)"):
            st.dataframe(T.style_df(sub_df), hide_index=True, width="stretch")
        st.download_button("⬇ Download (CSV)", to_csv(res_tbl), "new_property_scores.csv")

# ------------------------------------------------------------------ TAB 4: explanation
with tabs[4]:
    T.section("How was the score derived?", "Plain-language walk-through of the calculation, with your numbers")
    scope = st.radio("What should be explained?", ["One wellness factor (preprocessing, weights, scores)", "The overall score of one property"],
                     horizontal=True)
    if scope.startswith("One"):
        ex_code = st.selectbox("Wellness factor", list(results), format_func=lambda c: f"{names[c]}  ({c})",
                               index=list(results).index(st.session_state["sel_factor"]) if st.session_state.get("sel_factor") in results else 0,
                               key="expl_factor")
        text = we.explain_factor(results[ex_code], level, float(alpha))
    else:
        pid2 = st.selectbox("Property", list(F.index), key="expl_pid")
        text = we.explain_property(results, F, fw, overall, o_lo, o_hi, pid2, profile, level)
        flags = [f"{results[c].name}: {m}" for c in results for _, m in results[c].flags]
        if flags:
            text += "\n\n**Data-quality flags across factors:**\n" + "\n".join(f"- {f}" for f in flags[:15])
    st.markdown(text)
    st.divider()
    st.markdown("#### AI narrative + critique (Groq)")
    st.caption("Sends the explanation above to a language model, which rewrites it in plain words and points out possible weaknesses.")
    q = st.text_input("Optional question for the model", "")
    if st.button("Generate AI explanation", type="primary"):
        tok = get_token()
        if not tok:
            st.warning("Add your Groq API key in the sidebar, in Streamlit Secrets (GROQ_API_KEY), or as an environment variable.")
        else:
            with st.spinner("Asking the model…"):
                try:
                    st.markdown(we.llm_explain(text, tok, llm_model, q))
                except Exception as e:
                    st.error(f"LLM call failed: {e}\n\nCheck the API key, your Groq rate limits, "
                             "or try another model name (e.g. llama-3.1-8b-instant).")

# ------------------------------------------------------------------ TAB 5: method
FLOW = r"""
digraph G {
  rankdir=TB; bgcolor="transparent";
  graph [fontname="Helvetica", nodesep=0.35, ranksep=0.42, pad=0.15];
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, color="#0F766E",
        fillcolor="#EAF4F2", fontcolor="#134E4A", margin="0.18,0.10"];
  edge [color="#6B7780", arrowsize=0.8];

  subgraph cluster_1 { label="STAGE 1 · Data preprocessing"; labeljust=l; style="rounded"; color="#9DBFB9"; fontcolor="#134E4A"; fontname="Helvetica-Bold";
    raw   [label="Raw parameters\n(one column per parameter)"];
    clean [label="Cleaning\nblank → median\noutlier → capped at IQR limits"];
    scale [label="Scale to 0–100\n(value − min) ÷ (max − min) × 100\nflip if lower is better"];
    raw -> clean -> scale;
  }
  subgraph cluster_2 { label="STAGE 2 · Finding the weights"; labeljust=l; style="rounded"; color="#9DBFB9"; fontcolor="#134E4A"; fontname="Helvetica-Bold";
    corr  [label="Correlation matrix"];
    fa    [label="Factor analysis\nhidden themes + loadings"];
    lw    [label="Loading weight\nh ÷ Σh"];
    ent   [label="Entropy weight\nhow much a parameter varies"];
    blend [label="Blend\nloading^α × entropy^(1−α)\nrescaled to 100%"];
    boot  [label="Bootstrap\nrepeat on B resamples\nFINAL weight = average"];
    corr -> fa -> lw -> blend;
    ent -> blend;
    blend -> boot;
  }
  subgraph cluster_3 { label="STAGE 3 · Building the scores"; labeljust=l; style="rounded"; color="#9DBFB9"; fontcolor="#134E4A"; fontname="Helvetica-Bold";
    sub   [label="Sub-factor score\nweighted average of its parameters"];
    subw  [label="Sub-factor weights\nPCA explained variance"];
    fscore [label="Factor score (0–100)\nΣ sub-factor weight × score"];
    sub -> fscore; subw -> fscore;
  }
  subgraph cluster_4 { label="STAGE 4 · Across the 9 factors"; labeljust=l; style="rounded"; color="#9DBFB9"; fontcolor="#134E4A"; fontname="Helvetica-Bold";
    fw    [label="Factor weights\nbuyer profile / equal / custom / data-driven"];
    total [label="WELLNESS SCORE (0–100)\nΣ factor weight × factor score", fillcolor="#0F766E", fontcolor="white"];
    ci    [label="Confidence range\npercentiles of the B bootstrap scores"];
    bnd   [label="Band\nPoor · Below average · Moderate · Good · Excellent"];
    fw -> total; total -> ci -> bnd;
  }
  scale -> corr; scale -> ent;
  boot -> sub; boot -> subw;
  fscore -> total;
}
"""

with tabs[5]:
    T.section("How the Wellness Score is calculated", "Every step, with the formula, the reason and a worked example")
    st.markdown("""
The score is built in **four stages**. Stages 1–3 run **separately inside each of the 9 wellness factors**
(Emotional, Physical, Social, Occupational, Environmental, Intellectual, Spiritual, Financial, Statutory);
stage 4 combines the 9 factor scores into the final **Wellness Score**.
""")
    st.graphviz_chart(FLOW, width="stretch")
    with st.expander("Words used in this app (glossary)", expanded=False):
        st.markdown("""
| Word | Meaning |
|---|---|
| **Parameter** | One measurable fact about a property, e.g. *AQI value*. |
| **Sub-factor** | A group of related parameters, e.g. *Air Quality*. |
| **Wellness factor** | One of the 9 areas of wellness, e.g. *Environmental Wellness*. |
| **Preprocessing** | Cleaning the raw data and converting it to a common 0–100 scale. |
| **Median** | The middle value of a sorted list; not distorted by extreme values. |
| **IQR** | Inter-quartile range = Q3 − Q1: the spread of the middle half of the data. |
| **Correlation** | −1 … +1: how closely two parameters rise and fall together. |
| **Factor analysis** | Statistical technique that finds a few *hidden themes* behind many correlated parameters. |
| **Theme (common factor)** | One hidden pattern found by factor analysis (not the same as a wellness factor). |
| **Loading** | How strongly a parameter is tied to a theme (−1 … +1). |
| **Communality** | Share of a parameter's variation explained by all themes (sum of squared loadings). |
| **KMO / Bartlett** | Two tests that check the data suits factor analysis. |
| **Entropy weight** | Weight based on how much a parameter varies between properties. |
| **Bootstrap** | Re-doing the calculation on many random resamples to see how stable it is. |
| **CV** | Coefficient of variation = spread ÷ average; high CV = unstable weight. |
| **PCA / explained variance** | Technique that measures how much of the total variation each item carries; used for sub-factor and factor weights. |
| **Confidence range** | Interval in which the score is expected to lie given the uncertainty in the weights. |
""")

    st.markdown("### Follow one real property through the calculation")
    st.caption("The examples below use your current data. Pick a factor and a property; everything updates.")
    mc1, mc2, mc3 = st.columns(3)
    m_code = mc1.selectbox("Wellness factor", list(results), format_func=lambda c: f"{names[c]} ({c})",
                           index=list(results).index("ENV") if "ENV" in results else 0, key="m_code")
    mr = results[m_code]
    di_row, di_col = we.find_interesting(mr)
    m_pid = mc2.selectbox("Property", list(mr.raw.index), index=list(mr.raw.index).index(di_row), key=f"m_pid_{m_code}")
    m_par = mc3.selectbox("Parameter", mr.columns, index=list(mr.columns).index(di_col), format_func=short, key=f"m_par_{m_code}")
    m_sub = next(s_ for s_, cs_ in mr.sub_map.items() if m_par in cs_)
    ptrace = we.parameter_trace(mr, m_pid, m_par)

    T.stage(1, "Data preprocessing", "Cleaning and scaling")
    with st.expander("Step 1 · Clean the data", expanded=True):
        st.markdown("""
**Why:** real data has blanks and a few absurd values. Left alone, a single extreme value would stretch the scale and make every other property look the same.

**How:**
1. Non-numbers become blanks.
2. **Fill blanks** with the **median** of that parameter.
3. **Cap outliers** at the **IQR limits**: lower = Q1 − 1.5 × IQR, upper = Q3 + 1.5 × IQR (Q1 and Q3 are the 25th and 75th percentiles, IQR = Q3 − Q1).
   Values beyond a limit are set equal to the limit. If IQR = 0 the limits are the column's min and max, so nothing is capped.
""")
        st.markdown(f"**Example with your data:** {short(m_par)} of property **{m_pid}**")
        st.dataframe(ptrace.iloc[:3], hide_index=True, width="stretch")
    with st.expander("Step 2 · Scale to 0–100 (and flip if lower is better)", expanded=True):
        st.markdown("""
**Why:** parameters have different units (dB, %, rupees…). A common 0–100 scale makes them comparable and lets 100 mean *best* for every parameter.

**How:** `score = (value − min) ÷ (max − min) × 100`, clipped to 0–100. *min* and *max* are your expected values (Data tab) or the smallest and largest cleaned values in the data.
If **lower is better**, `score = 100 − score`. If min = max, the score is 50.

**Numeric illustration:** AQI = 162, with expected range 0–500, lower is better →
(162 − 0) ÷ (500 − 0) × 100 = **32.4** → flipped: 100 − 32.4 = **67.6**.
""")
        st.markdown(f"**Example with your data:** {short(m_par)} of property **{m_pid}**")
        st.dataframe(ptrace.iloc[3:], hide_index=True, width="stretch")

    T.stage(2, "Finding the weights", "Correlation → factor analysis → entropy → blend → bootstrap")
    with st.expander("Step 3 · Correlation matrix", expanded=False):
        sub_cols = mr.sub_map[m_sub]
        a_c, b_c = (sub_cols[0], sub_cols[1]) if len(sub_cols) > 1 else (sub_cols[0], sub_cols[0])
        st.markdown(f"""
**Why:** to learn which parameters carry the same information.

**How:** Pearson correlation *r* between every pair of 0–100 parameter scores over all properties: +1 = move together perfectly, 0 = unrelated, −1 = opposite.

**Example with your data:** correlation between *{short(a_c)}* and *{short(b_c)}* = **{mr.corr.loc[a_c, b_c]:.2f}**.
""")
    with st.expander("Step 4 · Factor analysis → loading weights", expanded=True):
        load_txt = ", ".join(f"{v:.2f}" for v in mr.loadings.loc[m_par].values)
        st.markdown(f"""
**Why:** several parameters often measure the same underlying theme. Factor analysis finds those themes, so a parameter that fits the shared pattern can be recognised and weighted accordingly.

**How:**
1. The correlation matrix is analysed. The number of themes **k** = number of eigenvalues above 1 (Kaiser rule), at least 1 and at most min(4, parameters ÷ 3). Here **k = {mr.n_factors}**.
2. Each parameter gets a **loading λ** on each theme (maximum-likelihood factor analysis with varimax rotation).
3. **Communality** `h = λ₁² + λ₂² + …` (the share of the parameter explained by the themes).
4. **Loading weight** `= h ÷ Σh`, so all loading weights in the factor add up to 100%.

**Example with your data:** *{short(m_par)}* has loadings {load_txt}
→ h = {mr.weights.loc[m_par, 'communality']:.3f} → loading weight = {mr.weights.loc[m_par, 'loading_weight'] * 100:.2f}% of {mr.name}.
(KMO = {mr.diagnostics['kmo']:.2f} – {we.kmo_label(mr.diagnostics['kmo'])}.)
""")
    with st.expander("Step 5 · Entropy weights (with a complete small example)", expanded=True):
        st.markdown("""
**Why:** a parameter that is almost the same for every property cannot tell properties apart, so it deserves little weight. One that varies a lot carries more information.

**How:** for a parameter *j* over *n* properties
1. `p(i,j) = x(i,j) ÷ Σ x(·,j)` (each property's share of the column total),
2. `e(j) = −(1 ÷ ln n) × Σ p ln p` (entropy; 1 = all properties identical, lower = more varied),
3. `d(j) = 1 − e(j)` (information carried), 4. `entropy weight(j) = d(j) ÷ Σ d`.
""")
        toy = np.array([[35, 10], [45, 90], [55, 20], [65, 80]], float)
        P = toy / toy.sum(axis=0)
        e = -(P * np.log(P)).sum(axis=0) / np.log(len(toy))
        d_ = 1 - e
        wts = d_ / d_.sum()
        assert np.allclose(wts, we.entropy_weights(toy), atol=1e-6)
        st.markdown("**Worked example:** 4 properties, 2 parameters (both already 0–100 scores).")
        c1_, c2_ = st.columns(2)
        c1_.dataframe(pd.DataFrame(toy, index=[f"Property {i + 1}" for i in range(4)], columns=["Parameter A", "Parameter B"]),
                      width="stretch")
        c2_.dataframe(pd.DataFrame(P, index=[f"Property {i + 1}" for i in range(4)],
                                   columns=["share p (A)", "share p (B)"]).round(4), width="stretch")
        st.dataframe(pd.DataFrame({"Entropy e": e, "Information d = 1 − e": d_, "Entropy weight": wts},
                                  index=["Parameter A (35, 45, 55, 65: similar)", "Parameter B (10, 90, 20, 80: very different)"]).round(4),
                     width="stretch")
        st.markdown(f"Parameter B varies far more between properties, so it earns **{wts[1] * 100:.1f}%** of the weight and parameter A only **{wts[0] * 100:.1f}%**.")
    with st.expander("Step 6 · Blend the two weights (with a worked example)", expanded=True):
        st.markdown(f"""
**Why:** the two methods look at different things and can disagree. Blending them uses both.

**How:** `blend(j) = loading(j)^α × entropy(j)^(1−α)`, then all blends are **rescaled to add up to 100%**. α is the sidebar slider (now **α = {alpha:g}**; 0.5 = geometric mean).
""")
        wl = np.array([0.50, 0.30, 0.20])
        wen = np.array([0.10, 0.30, 0.60])
        raw_b = wl ** float(alpha) * wen ** (1 - float(alpha))
        st.markdown(f"**Numeric illustration** (3 parameters, α = {alpha:g}):")
        norm_col = f"Final blend (÷ {raw_b.sum():.4f})"
        st.dataframe(pd.DataFrame({"Loading weight": wl, "Entropy weight": wen, "Blend before rescaling": raw_b,
                                   norm_col: raw_b / raw_b.sum()},
                                  index=["Parameter 1", "Parameter 2", "Parameter 3"]).round(4), width="stretch")
        st.markdown(f"**Example with your data:** *{short(m_par)}*")
        st.dataframe(we.weight_trace(mr, m_par, float(alpha)).iloc[:5], hide_index=True, width="stretch")
    with st.expander("Step 7 · Bootstrap → final weights", expanded=True):
        wrow = mr.weights.loc[m_par]
        st.markdown(f"""
**Why:** a weight is only trustworthy if it does not change much when the data changes a little.

**How:**
1. Draw **n properties at random with replacement** from the n properties (some appear twice, some not at all).
2. Re-run steps 3–6 (and the sub-factor weights of step 9) on that resample.
3. Repeat **B = {len(mr.boot_w)}** times (sidebar slider).
4. **Final weight = average** of the B weights (rescaled to 100%). **Spread** (standard deviation) and the 2.5%–97.5% range show how stable it is; **CV = spread ÷ weight**, and a CV above 0.6 raises an "unstable weight" flag.

**Example with your data:** *{short(m_par)}*: final weight **{wrow['final_weight'] * 100:.2f}%**, spread {wrow['bootstrap_std'] * 100:.2f} points,
95% range {wrow['ci_lo'] * 100:.2f}% – {wrow['ci_hi'] * 100:.2f}%, CV = {wrow['cv']:.2f}.
""")

    T.stage(3, "Building the scores", "Parameters → sub-factors → factor")
    with st.expander("Step 8 · Sub-factor score (weighted average)", expanded=True):
        st.markdown(f"""
**How:** `sub-factor score = Σ (parameter score × weight) ÷ Σ weight`, summed over the parameters of the sub-factor (the weights are rescaled to 100% inside the sub-factor).

**Example with your data:** sub-factor **{m_sub}** of property **{m_pid}**
""")
        st.dataframe(T.style_df(we.subfactor_trace(mr, m_pid, m_sub), 2), hide_index=True, width="stretch")
    with st.expander("Step 9 · Sub-factor weights (PCA explained variance)", expanded=False):
        sw_ = (mr.sub_weights * 100).sort_values(ascending=False)
        sw_txt = "; ".join(f"{s_name}: {v:.1f}%" for s_name, v in sw_.items())
        st.markdown(f"""
**Why:** sub-factors do not all contribute equally to the factor.

**How:** standardise the sub-factor scores, compute their correlation matrix and its eigenvalues λₖ and eigenvectors. Keep the components with λ > 1.
Each component's **explained-variance ratio** is `EVRₖ = λₖ ÷ Σλ`. The weight of sub-factor *s* is proportional to `Σ over kept components of EVRₖ × (eigenvector element)²`, rescaled to 100%.
A sub-factor that carries more of the common variation gets more weight.

**Example with your data ({mr.name}):** {sw_txt}.
""")
    with st.expander("Step 10 · Factor score", expanded=True):
        st.markdown(f"""
**How:** `factor score = Σ (sub-factor weight × sub-factor score)`, a number from 0 to 100.

**Example with your data:** {mr.name}, property **{m_pid}**
""")
        st.dataframe(T.style_df(we.factor_trace(mr, m_pid), 2), hide_index=True, width="stretch")

    T.stage(4, "Across the 9 factors", "Combine, add the confidence range, name the band")
    with st.expander("Step 11 · Combine the 9 factor scores into the Wellness Score", expanded=True):
        st.markdown(f"""
**How:** `Wellness Score = Σ (factor weight × factor score)`. Factor weights come from the option chosen in the sidebar (now **{profile}**):
*Balanced* (all equal), a *buyer profile*, *Custom* sliders, or *Data-driven* (PCA explained variance across the factor scores, as in step 9).
See "What do the weighting options mean?" in the Overall score tab.

**Example with your data:** property **{m_pid}**
""")
        st.dataframe(T.style_df(we.overall_trace(F, fw, names, m_pid), 2), hide_index=True, width="stretch")
    with st.expander("Step 12 · Confidence range and band", expanded=True):
        st.markdown(f"""
**Confidence range:** every one of the B bootstrap weight sets is applied to **every** property, giving B possible scores for each property.
For a {level:.0%} range the **{(1 - level) / 2 * 100:.1f}th and {100 - (1 - level) / 2 * 100:.1f}th percentiles** of those B scores are the range limits (widened if necessary so the score itself is inside).

**Band:**

| Score | Band |
|---|---|
| below 35 | Poor |
| 35 – 50 | Below average |
| 50 – 65 | Moderate |
| 65 – 80 | Good |
| 80 and above | Excellent |

**Example with your data:** property **{m_pid}** scores **{overall[m_pid]:.1f}** (band **{we.band(overall[m_pid])}**),
{level:.0%} range **{o_lo[m_pid]:.1f} – {o_hi[m_pid]:.1f}**.
""")

    T.section("Automatic checks the app runs")
    st.markdown("""
* **Enough properties:** at least about 5 properties per parameter.
* **KMO below 0.5**, or **Bartlett's test p above 0.05:** the data do not suit factor analysis.
* **More than 10% missing** values in a parameter, or **many outliers** capped.
* **Zero variance:** a parameter that never changes (weight 0).
* **Near-duplicates:** two parameters with correlation above 0.9 (information counted twice).
* **Weight dominance:** one parameter carries more than 15% of a factor.
* **Unstable weights:** bootstrap CV above 0.6.
* **Reversed direction:** a parameter negatively correlated with the others in its sub-factor (check *Better when*).
* **Wide range:** average 95% range wider than 15 points.
""")
    T.section("Limits to be upfront about")
    st.markdown("""
* **Scores are relative** to the reference properties unless you set expected min / max values: a "70" means "better than most of these properties", not an absolute standard.
* The **confidence range reflects weighting uncertainty only**, not measurement error in the input data.
* **Data-driven weights show what varies in the data, not what matters to a buyer.** Use a buyer profile or expert (AHP) weights for preferences.
* Factor analysis needs enough rows (about 5–10 per parameter) and correlated parameters. Check the quality flags.
* Some parameters in different factors describe similar things (for example noise or air quality appear in more than one factor). They are scored in each factor where they are listed, so those themes count in more than one place.
* The built-in synthetic data are random with built-in structure. Scores from them only test the pipeline.
""")
