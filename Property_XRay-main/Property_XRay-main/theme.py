"""
theme.py  --  visual layer for Property X-Ray (colours, CSS, Plotly template, small UI helpers).
Pure presentation: nothing here touches the scoring logic.
"""
import html

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ------------------------------------------------------------------ palette
TEAL, TEAL_DARK, TEAL_SOFT = "#0F766E", "#134E4A", "#CCE8E3"
CORAL, GOLD, INK, MUTED = "#D9684A", "#D9A441", "#1F2A2E", "#6B7780"
BG, CARD, LINE = "#F7F5F0", "#FFFFFF", "#E6E1D6"
COLORWAY = [TEAL, CORAL, GOLD, "#3B82A0", "#7A8F3C", "#8E5A9B", "#5B6770"]
DIVERGING = [[0, "#C2553A"], [0.25, "#EBB9A6"], [0.5, "#F7F5F0"], [0.75, "#8FCBC2"], [1, TEAL]]
SEQUENTIAL = [[0, "#F7F5F0"], [1, TEAL]]

BAND_COLORS = {"Excellent": "#15803D", "Good": "#65A30D", "Moderate": "#C58B16",
               "Below average": "#D9684A", "Poor": "#B91C1C"}
BAND_BG = {"Excellent": "#DCF3E3", "Good": "#EAF4D3", "Moderate": "#FBF0D2",
           "Below average": "#FBE3DA", "Poor": "#F8D9D9"}
BAND_EDGES = [(0, 35, "Poor"), (35, 50, "Below average"), (50, 65, "Moderate"),
              (65, 80, "Good"), (80, 100, "Excellent")]

# ------------------------------------------------------------------ plotly template
pio.templates["xray"] = go.layout.Template(layout=dict(
    font=dict(family="Inter, system-ui, sans-serif", color=INK, size=13),
    colorway=COLORWAY,
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    title=dict(font=dict(family="Fraunces, Georgia, serif", size=18, color=TEAL_DARK), x=0.01),
    margin=dict(l=10, r=10, t=55, b=10),
    xaxis=dict(gridcolor="#EAE6DB", zeroline=False, linecolor=LINE, ticks="outside", tickcolor=LINE, automargin=True),
    yaxis=dict(gridcolor="#EAE6DB", zeroline=False, linecolor=LINE, automargin=True),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text=""),
    hoverlabel=dict(bgcolor="white", bordercolor=LINE, font=dict(family="Inter", color=INK)),
    colorscale=dict(diverging=DIVERGING, sequential=SEQUENTIAL),
))
pio.templates.default = "xray"


def show(fig, container=st, **kw):
    """Render a Plotly figure with our own theme (theme=None stops Streamlit overriding it)."""
    fig.update_layout(template="xray")
    container.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False}, **kw)


def style_hist(fig, color=TEAL):
    fig.update_traces(marker_color=color, marker_line_color="white", marker_line_width=1.2, opacity=0.95)
    fig.update_layout(bargap=0.04, showlegend=False)
    return fig


def style_bars(fig):
    fig.update_traces(marker_line_width=0)
    fig.update_layout(bargap=0.28)
    return fig


def add_band_zones(fig, axis="x"):
    """Soft coloured background zones for the five score bands (0-100)."""
    for lo, hi, name in BAND_EDGES:
        kw = dict(fillcolor=BAND_COLORS[name], opacity=0.07, line_width=0, layer="below")
        if axis == "x":
            fig.add_vrect(x0=lo, x1=hi, **kw)
        else:
            fig.add_hrect(y0=lo, y1=hi, **kw)
    return fig


def gauge(value, title, height=250):
    from_band = next((n for lo, hi, n in BAND_EDGES if value < hi), "Excellent")
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=float(value), number=dict(font=dict(family="Fraunces, serif", size=44, color=INK),
                                                              valueformat=".1f"),
        title=dict(text=title, font=dict(size=13, color=MUTED)),
        gauge=dict(axis=dict(range=[0, 100], tickcolor=LINE, tickfont=dict(size=10, color=MUTED)),
                   bar=dict(color=BAND_COLORS[from_band], thickness=0.28), bgcolor="white", borderwidth=0,
                   steps=[dict(range=[lo, hi], color=BAND_BG[n]) for lo, hi, n in BAND_EDGES])))
    fig.update_layout(height=height, margin=dict(l=25, r=25, t=45, b=5))
    return fig


# ------------------------------------------------------------------ label-safe chart helpers
def heatmap(mat, group_sizes=None, text=None, zmin=-1.0, zmax=1.0, colorscale=None, cbar_title="", x_title=None, fmt=".2f"):
    """Heat-map with EVERY label shown (no skipping / overlapping), values printed when the grid is small,
    and optional separator lines between groups (e.g. sub-factors)."""
    n_r, n_c = mat.shape
    show_text = (n_r <= 14 and n_c <= 14) if text is None else text
    fig = go.Figure(go.Heatmap(
        z=mat.values, x=list(mat.columns), y=list(mat.index), colorscale=colorscale or DIVERGING,
        zmin=zmin, zmax=zmax, xgap=1, ygap=1,
        colorbar=dict(thickness=12, len=0.85, title=dict(text=cbar_title, side="right")),
        texttemplate=f"%{{z:{fmt}}}" if show_text else None, textfont=dict(size=11),
        hovertemplate="%{y}<br>%{x}<br>value: %{z:.2f}<extra></extra>"))
    fig.update_xaxes(type="category", dtick=1, tickangle=-45, side="bottom", tickfont=dict(size=11),
                     showgrid=False, ticks="", automargin=True, title=x_title)
    fig.update_yaxes(type="category", dtick=1, autorange="reversed", tickfont=dict(size=11),
                     showgrid=False, ticks="", automargin=True)
    if group_sizes:
        pos = 0
        for size in group_sizes[:-1]:
            pos += size
            fig.add_vline(x=pos - 0.5, line_width=2, line_color=INK, opacity=0.55)
            fig.add_hline(y=pos - 0.5, line_width=2, line_color=INK, opacity=0.55)
    fig.update_layout(height=max(380, 27 * n_r + 230), margin=dict(l=10, r=10, t=20, b=10))
    return fig


def dumbbell(labels, a, b, mid, a_name, b_name, mid_name, xtitle):
    """One row per label: two dots joined by a line (a vs b) and a diamond for the blend."""
    labels = list(labels)
    xs, ys = [], []
    for lab, x1, x2 in zip(labels, a, b):
        xs += [x1, x2, None]
        ys += [lab, lab, None]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#C9D3D1", width=3), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=list(a), y=labels, mode="markers", name=a_name,
                             marker=dict(color=TEAL, size=11, line=dict(color="white", width=1.5))))
    fig.add_trace(go.Scatter(x=list(b), y=labels, mode="markers", name=b_name,
                             marker=dict(color=CORAL, size=11, line=dict(color="white", width=1.5))))
    fig.add_trace(go.Scatter(x=list(mid), y=labels, mode="markers", name=mid_name,
                             marker=dict(color=TEAL_DARK, size=12, symbol="diamond", line=dict(color="white", width=1.5))))
    fig.update_yaxes(type="category", categoryorder="array", categoryarray=labels, autorange="reversed", dtick=1,
                     tickfont=dict(size=12))
    fig.update_xaxes(title=xtitle, rangemode="tozero")
    fig.update_layout(height=max(380, 30 * len(labels) + 140))
    return fig


def range_plot(names, score, lo, hi, bands, level):
    """Horizontal dot-and-whisker chart: one row per property, coloured by band, with background band zones."""
    names = list(names)
    score, lo, hi = list(score), list(lo), list(hi)
    fig = go.Figure(go.Scatter(
        x=score, y=names, mode="markers",
        marker=dict(size=11, color=[BAND_COLORS[b] for b in bands], line=dict(color="white", width=1.5)),
        error_x=dict(type="data", symmetric=False, color="#8D999C", thickness=2, width=5,
                     array=[h - s_ for h, s_ in zip(hi, score)], arrayminus=[s_ - l for s_, l in zip(score, lo)]),
        customdata=list(zip(lo, hi, bands)),
        hovertemplate="%{y}<br>score %{x:.1f}<br>" + f"{level:.0%}" + " range %{customdata[0]:.1f} – %{customdata[1]:.1f}<br>%{customdata[2]}<extra></extra>"))
    add_band_zones(fig, "x")
    fig.update_yaxes(type="category", categoryorder="array", categoryarray=names, autorange="reversed", dtick=1,
                     tickfont=dict(size=12))
    fig.update_xaxes(range=[max(0, min(lo) - 6), min(100, max(hi) + 6)], title="Wellness Score (0–100)")
    fig.update_layout(height=max(360, 26 * len(names) + 120), showlegend=False)
    return fig


def waterfall(contrib, total, total_label="Wellness Score", xtitle="Points contributed"):
    """Horizontal waterfall: each bar = score x weight; bars add up to the total."""
    labels = list(contrib.index) + [total_label]
    vals = [float(v) for v in contrib.values] + [float(total)]
    fig = go.Figure(go.Waterfall(
        orientation="h", measure=["relative"] * len(contrib) + ["total"], y=labels, x=vals,
        text=[f"{v:.1f}" for v in vals], textposition="outside", cliponaxis=False,
        connector=dict(line=dict(color=MUTED, width=1, dash="dot")),
        increasing=dict(marker=dict(color=TEAL)), totals=dict(marker=dict(color=TEAL_DARK))))
    fig.update_yaxes(type="category", categoryorder="array", categoryarray=labels, autorange="reversed", dtick=1,
                     tickfont=dict(size=12))
    fig.update_xaxes(title=xtitle, range=[0, max(vals) * 1.12])
    fig.update_layout(height=max(320, 40 * len(labels) + 90), showlegend=False)
    return fig


def stage(num, title, sub):
    """Big numbered banner that separates the pipeline stages."""
    st.markdown(f"<div class='xr-stage'><span>{num}</span><div><b>{html.escape(title)}</b>"
                f"<small>{html.escape(sub)}</small></div></div>", unsafe_allow_html=True)


# ------------------------------------------------------------------ dataframe styling
def style_df(df, precision=1):
    """Round numbers and colour-code a 'Band' column if present."""
    sty = df.style.format(precision=precision, na_rep="")
    if "Band" in df.columns:
        fn = sty.map if hasattr(sty, "map") else sty.applymap
        sty = fn(lambda v: f"background-color:{BAND_BG.get(v, 'transparent')};"
                           f"color:{BAND_COLORS.get(v, INK)};font-weight:600;", subset=["Band"])
    return sty


# ------------------------------------------------------------------ html helpers
def badge(band, size="md"):
    pad = "4px 14px" if size == "md" else "2px 10px"
    return (f"<span style='display:inline-block;padding:{pad};border-radius:999px;font-weight:600;"
            f"font-size:{'0.95rem' if size == 'md' else '0.8rem'};background:{BAND_BG[band]};"
            f"color:{BAND_COLORS[band]};border:1px solid {BAND_COLORS[band]}33'>{band}</span>")


def section(title, sub=None):
    s = f"<div class='xr-section'><h3>{html.escape(title)}</h3>"
    if sub:
        s += f"<p>{html.escape(sub)}</p>"
    st.markdown(s + "</div>", unsafe_allow_html=True)


def hero(title, subtitle, steps):
    chips = "".join(f"<span class='xr-chip'>{html.escape(s)}</span>" for s in steps)
    st.markdown(f"""
<div class='xr-hero'>
  <div class='xr-eyebrow'>PROPERTY&nbsp;X-RAY</div>
  <h1>{html.escape(title)}</h1>
  <p>{html.escape(subtitle)}</p>
  <div class='xr-chips'>{chips}</div>
</div>""", unsafe_allow_html=True)


def flag(severity, msg):
    col = "#B91C1C" if severity == "high" else "#C58B16"
    bg = "#FBEAEA" if severity == "high" else "#FCF4DE"
    label = "HIGH" if severity == "high" else "NOTE"
    st.markdown(f"<div class='xr-flag' style='border-left-color:{col};background:{bg}'>"
                f"<b style='color:{col}'>{label}</b> &nbsp;{html.escape(msg)}</div>", unsafe_allow_html=True)


def sidebar_brand():
    st.markdown("<div class='xr-brand'><span>🏠</span><div><b>Property X-Ray</b>"
                "<small>Wellness Score engine</small></div></div>", unsafe_allow_html=True)


# ------------------------------------------------------------------ CSS
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"], .stApp {{ font-family: 'Inter', system-ui, sans-serif; }}
.stApp {{ background: {BG}; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 4rem; max-width: 1280px; }}
header[data-testid="stHeader"] {{ background: transparent; }}
#MainMenu, footer {{ visibility: hidden; }}

h1, h2, h3, h4 {{ font-family: 'Fraunces', Georgia, serif !important; color: {TEAL_DARK}; letter-spacing: -0.01em; }}

/* hero */
.xr-hero {{ background: radial-gradient(1200px 300px at 90% -20%, #1E9C8F55, transparent),
            linear-gradient(135deg, {TEAL_DARK} 0%, {TEAL} 100%);
            border-radius: 22px; padding: 34px 40px 28px; margin-bottom: 22px; color: white;
            box-shadow: 0 10px 30px -12px {TEAL_DARK}99; }}
.xr-hero h1 {{ color: white !important; font-size: 2.5rem; margin: 6px 0 6px; padding: 0; line-height: 1.1; }}
.xr-hero p {{ color: #D5EEEA; font-size: 1.02rem; margin: 0 0 16px; max-width: 760px; }}
.xr-eyebrow {{ letter-spacing: .22em; font-size: .72rem; font-weight: 600; color: #9FDDD3; }}
.xr-chips {{ display: flex; flex-wrap: wrap; gap: 8px; }}
.xr-chip {{ background: #ffffff1c; border: 1px solid #ffffff33; padding: 3px 12px; border-radius: 999px;
            font-size: .78rem; color: #EAFBF8; backdrop-filter: blur(4px); }}

/* section heads */
.xr-section {{ margin: 6px 0 10px; }}
.xr-section h3 {{ margin: 0; font-size: 1.45rem; }}
.xr-section p {{ margin: 2px 0 0; color: {MUTED}; font-size: .92rem; }}

/* tabs as pills */
div[data-baseweb="tab-list"] {{ gap: 6px; background: #EFEBE1; padding: 5px; border-radius: 14px; width: fit-content; }}
button[data-baseweb="tab"] {{ height: 40px; border-radius: 10px; padding: 0 18px; font-weight: 500; color: {MUTED}; background: transparent; }}
button[data-baseweb="tab"][aria-selected="true"] {{ background: white; color: {TEAL_DARK}; box-shadow: 0 1px 4px #0000001a; }}
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] {{ display: none; }}

/* metric cards */
[data-testid="stMetric"] {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 16px; padding: 16px 18px;
                            box-shadow: 0 1px 2px #0000000a; border-top: 3px solid {TEAL}; }}
[data-testid="stMetricLabel"] p {{ color: {MUTED}; font-size: .78rem; text-transform: uppercase; letter-spacing: .06em; font-weight: 600; }}
[data-testid="stMetricValue"] {{ font-family: 'Fraunces', serif; color: {INK}; font-size: 1.9rem; }}

/* expanders as cards */
[data-testid="stExpander"] {{ background: {CARD}; border: 1px solid {LINE} !important; border-radius: 16px; overflow: hidden;
                              box-shadow: 0 1px 2px #0000000a; margin-bottom: 10px; }}
[data-testid="stExpander"] summary {{ font-weight: 600; color: {TEAL_DARK}; padding: 12px 16px; }}
[data-testid="stExpander"] summary:hover {{ background: #F3F8F7; }}

/* sidebar */
[data-testid="stSidebar"] {{ background: linear-gradient(180deg, #FFFFFF 0%, #F1EEE5 100%); border-right: 1px solid {LINE}; }}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{ font-size: 1.05rem; }}
.xr-brand {{ display: flex; align-items: center; gap: 12px; padding: 4px 0 14px; }}
.xr-brand span {{ font-size: 1.7rem; background: {TEAL}; border-radius: 12px; padding: 6px 9px; }}
.xr-brand b {{ display: block; font-family: 'Fraunces', serif; font-size: 1.15rem; color: {TEAL_DARK}; line-height: 1.1; }}
.xr-brand small {{ color: {MUTED}; }}

/* buttons */
.stButton > button, .stDownloadButton > button {{ border-radius: 12px; font-weight: 600; padding: .5rem 1.1rem; transition: all .15s; }}
.stButton > button[kind="primary"] {{ background: linear-gradient(135deg, {TEAL}, {TEAL_DARK}); border: none; color: white;
                                      box-shadow: 0 6px 14px -6px {TEAL_DARK}; }}
.stButton > button[kind="primary"]:hover {{ transform: translateY(-1px); box-shadow: 0 10px 18px -8px {TEAL_DARK}; }}
.stDownloadButton > button {{ background: white; border: 1px solid {TEAL}; color: {TEAL_DARK}; }}
.stDownloadButton > button:hover {{ background: {TEAL_SOFT}; border-color: {TEAL}; color: {TEAL_DARK}; }}

/* tables, inputs, alerts */
[data-testid="stDataFrame"] {{ border: 1px solid {LINE}; border-radius: 14px; overflow: hidden; }}
[data-baseweb="select"] > div, [data-baseweb="input"] > div {{ border-radius: 10px; }}
[data-testid="stAlert"] {{ border-radius: 14px; border: none; }}
hr {{ border-color: {LINE}; margin: 1.6rem 0; }}

/* flags */
.xr-flag {{ border-left: 4px solid; padding: 9px 14px; border-radius: 8px; margin: 6px 0; font-size: .92rem; }}

/* pipeline stage banner */
.xr-stage {{ display: flex; align-items: center; gap: 16px; background: {CARD}; border: 1px solid {LINE}; border-radius: 16px;
             padding: 14px 20px; margin: 22px 0 12px; box-shadow: 0 1px 2px #0000000a; }}
.xr-stage span {{ background: {TEAL}; color: white; font-family: 'Fraunces', serif; font-size: 1.5rem; font-weight: 600;
                  width: 46px; height: 46px; border-radius: 50%; display: flex; align-items: center; justify-content: center; flex: none; }}
.xr-stage b {{ display: block; font-family: 'Fraunces', serif; font-size: 1.3rem; color: {TEAL_DARK}; }}
.xr-stage small {{ color: {MUTED}; font-size: .92rem; }}

/* explanation / method prose */
.xr-prose {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 16px; padding: 18px 24px; }}
</style>
"""


def inject():
    st.markdown(CSS, unsafe_allow_html=True)
