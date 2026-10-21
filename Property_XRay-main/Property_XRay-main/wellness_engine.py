"""
wellness_engine.py  --  Property X-Ray : Wellness Score engine
=================================================================
Pipeline (applied independently to each of the 9 wellness factors):

    1  Raw parameters
    2  Data preprocessing   (missing values -> outliers -> direction + 0-100 scaling)
    3  Correlation matrix
    4  Factor analysis      (how many hidden factors? KMO / Bartlett checks)
    5  Factor loadings      (-> communality = how "central" a parameter is)
    6  Entropy weights      (how much a parameter separates one property from another)
    7  Bootstrap validation (repeat 3-6 on resampled properties)
    8  Parameter weights    (bootstrap mean)  ->  effective weight in the factor score
    9  Sub-factor scores    (weighted mean) and explained-variance sub-factor weights
    10 Factor score + confidence interval

Then the 9 factor scores are combined into the overall Wellness Score
(equal / custom / data-driven weights) with a confidence interval.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import FactorAnalysis

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# 1. HIERARCHY  (taken from the Wellness Score diagram)
# --------------------------------------------------------------------------
HIERARCHY: dict[str, dict] = {
    "EMO": {"name": "Emotional Wellness", "subs": {
        "Stress Reduction Environment": ["Green views", "Noise levels", "Natural light"],
        "Aesthetic Pleasure": ["Visual appeal", "Architecture quality", "Interior design"],
        "Sense of Safety & Security": ["Security features", "Safe surroundings", "Gated / access control"]}},
    "PHY": {"name": "Physical Wellness", "subs": {
        "Physical Activity Support": ["Proximity to parks", "Sports facilities", "Gyms / fitness centers"],
        "Walkability": ["Walk score", "Sidewalk quality", "Pedestrian friendliness"],
        "Safety Features": ["Lighting quality", "Traffic safety", "Crossings & signs"],
        "Environmental Quality": ["Air quality index", "Water quality", "Clean surroundings"],
        "Accessibility": ["Universal design", "Ramps & lifts", "Senior/disabled access"],
        "Healthcare Access": ["Distance to hospitals", "Clinics availability", "Emergency access"],
        "Transportation": ["Public transport access", "Frequency & reliability", "Connectivity"],
        "Ergonomics": ["Space planning", "Ventilation", "Comfort design"]}},
    "SOC": {"name": "Social Wellness", "subs": {
        "Community Engagement": ["Social events", "Community organizations", "Volunteer opportunities"],
        "Neighbourhood Safety": ["Crime rate", "Perception of safety", "Emergency services access"],
        "Social Connectivity": ["Public spaces", "Transportation links", "Digital connectivity"],
        "Inclusivity & Diversity": ["Demographic diversity", "Cultural activities"],
        "Community Support Networks": ["Support groups", "Local resources", "Mutual help"],
        "Work-Life Balance": ["Proximity to workplaces", "Leisure facilities"]}},
    "OCC": {"name": "Occupational Wellness", "subs": {
        "Work-Life Balance": ["Flexible workspace", "Work hours balance", "Personal time"],
        "Commute Efficiency": ["Commute time", "Traffic conditions", "Transport options"],
        "Coworking Availability": ["Coworking spaces", "Shared offices", "Business centers"],
        "Employment Density": ["Job opportunities", "Employment rate", "Business ecosystem"],
        "Job Opportunities": ["Career growth", "Industry presence", "Skill demand"]}},
    "ENV": {"name": "Environmental Wellness", "subs": {
        "Air Quality": ["AQI value", "Pollutant levels", "Ventilation quality"],
        "Water Quality": ["Drinking water quality", "Water source", "Contamination levels"],
        "Waste Management": ["Waste segregation", "Collection systems", "Recycling facilities"],
        "Noise Pollution": ["Noise levels (dB)", "Traffic noise", "Industrial noise"],
        "Green Cover": ["% Green cover", "Trees / vegetation", "Open spaces"],
        "Energy Efficiency": ["Energy star rating", "Renewable energy", "Efficiency systems"]}},
    "INT": {"name": "Intellectual Wellness", "subs": {
        "Educational Resources": ["Schools proximity", "Colleges proximity", "Quality of education"],
        "Library Access": ["Public libraries", "Digital resources", "Reading spaces"],
        "Skill Development": ["Training centers", "Workshops", "Online learning access"],
        "Cultural & Creative Engagement": ["Cultural events", "Museums / galleries", "Creative spaces"],
        "Innovation & Learning Ecosystem": ["Hackathons / forums", "Community learning", "Knowledge networks"]}},
    "SPI": {"name": "Spiritual Wellness", "subs": {
        "Places of Worship": ["Temples / Churches", "Mosques / Gurudwaras", "Distance & access"],
        "Meditation Spaces": ["Quiet zones", "Meditation centers", "Yoga facilities"],
        "Tranquility & Peace": ["Calm environment", "Low noise", "Stress-free zones"],
        "Cultural Heritage": ["Heritage sites", "Historical places", "Traditions & art"],
        "Mind-Body Balance": ["Wellness centers", "Yoga / fitness", "Holistic activities"]}},
    "FIN": {"name": "Financial Wellness", "subs": {
        "Financial Performance Metrics": ["Cap Rate", "Rent to Cost Ratio", "ROI"],
        "Investment Viability": ["NPV", "IRR", "Break-even"],
        "Cost & Affordability": ["Property price", "Maintenance cost", "Utility cost"],
        "Cash Flow Stability": ["Rental income", "Occupancy rate", "Debt coverage"],
        "Market & Growth Potential": ["Market trend", "Appreciation", "Future demand"]}},
    "STA": {"name": "Statutory Wellness", "subs": {
        "Legal Compliance": ["RERA registration", "Building approvals", "Land use compliance"],
        "Title & Ownership Documents": ["Title deed", "Encumbrance certificate", "Sale agreement"],
        "Building & Safety Certifications": ["Occupancy certificate", "Fire safety NOC", "Structural safety"],
        "Tax & Financial Documents": ["Property tax receipt", "Utility bills", "Financial statements"],
        "Government Verifications": ["Aadhaar / PAN", "KYC documents", "Other clearances"]}},
}

# Parameters where a HIGHER raw value is WORSE (direction = "lower").
# Everything else is "higher is better".  Editable in the app.
LOWER_IS_BETTER = {
    "EMO": {"Noise levels"},
    "PHY": {"Air quality index", "Distance to hospitals"},
    "SOC": {"Crime rate"},
    "OCC": {"Commute time", "Traffic conditions"},
    "ENV": {"AQI value", "Pollutant levels", "Contamination levels",
            "Noise levels (dB)", "Traffic noise", "Industrial noise"},
    "FIN": {"Property price", "Maintenance cost", "Utility cost", "Break-even"},
}

# Realistic units for the synthetic data generator (default = 0-100 score)
SYN_RANGES = {
    "EMO: Noise levels": (35, 90),
    "PHY: Air quality index": (15, 300), "PHY: Distance to hospitals": (0.3, 15),
    "SOC: Crime rate": (0.5, 40),
    "OCC: Commute time": (5, 90), "OCC: Traffic conditions": (10, 100),
    "ENV: AQI value": (15, 300), "ENV: Pollutant levels": (5, 150),
    "ENV: Contamination levels": (0, 10), "ENV: Noise levels (dB)": (35, 90),
    "ENV: Traffic noise": (30, 85), "ENV: Industrial noise": (30, 85),
    "ENV: % Green cover": (2, 60), "ENV: Energy star rating": (1, 5),
    "FIN: Cap Rate": (1, 8), "FIN: Rent to Cost Ratio": (0.2, 1.2), "FIN: ROI": (2, 18),
    "FIN: NPV": (-5, 40), "FIN: IRR": (3, 20), "FIN: Break-even": (3, 25),
    "FIN: Property price": (4000, 25000), "FIN: Maintenance cost": (1, 12),
    "FIN: Utility cost": (1, 8), "FIN: Rental income": (10, 120),
    "FIN: Occupancy rate": (50, 100), "FIN: Debt coverage": (0.8, 3),
    "FIN: Appreciation": (1, 15),
}

# The three overall-weighting methods offered in the app
OVERALL_METHODS = ["Equal weights", "Data-driven (explained variance)", "Custom"]

STAGES = ["Raw data", "Data preprocessing", "Correlation", "Factor analysis", "Entropy",
          "Bootstrap", "Parameter weights", "Sub-factor scores", "Factor scores", "Wellness Score + CI"]

DIRECTION_LABEL = {"higher": "Higher value", "lower": "Lower value"}


def col_name(code: str, label: str) -> str:
    return f"{code}: {label}"


def short_name(col: str) -> str:
    return col.split(": ", 1)[1] if ": " in col else col


def default_config() -> pd.DataFrame:
    """One row per parameter: factor, sub-factor, direction, optional benchmark min / max."""
    rows = []
    for code, f in HIERARCHY.items():
        lower = LOWER_IS_BETTER.get(code, set())
        for sub, labels in f["subs"].items():
            for lab in labels:
                rows.append(dict(Code=code, Factor=f["name"], Subfactor=sub, Parameter=lab,
                                 Column=col_name(code, lab),
                                 Direction="lower" if lab in lower else "higher",
                                 Min=np.nan, Max=np.nan))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 2. SYNTHETIC DATA  (n rows x all parameter columns, with a latent structure)
# --------------------------------------------------------------------------
def generate_sample_data(n: int = 100, seed: int = 42, messy: bool = True) -> pd.DataFrame:
    """Synthetic properties. Correlated within sub-factor/factor so factor analysis has
    real structure to find. Optionally injects missing values and outliers."""
    rng = np.random.default_rng(seed)
    g = rng.normal(size=n)                                   # overall locality quality
    data = {"Property_ID": [f"P{i + 1:03d}" for i in range(n)]}
    for code, f in HIERARCHY.items():
        fl = rng.normal(size=n)
        for sub, labels in f["subs"].items():
            sl = rng.normal(size=n)
            for lab in labels:
                c = col_name(code, lab)
                z = 0.35 * g + 0.45 * fl + 0.65 * sl + 0.55 * rng.normal(size=n)
                z = (z - z.mean()) / z.std()
                q = stats.norm.cdf(z)                        # 0..1 quality
                lo, hi = SYN_RANGES.get(c, (0, 100))
                lower = lab in LOWER_IS_BETTER.get(code, set())
                raw = lo + (hi - lo) * ((1 - q) if lower else q)
                data[c] = np.round(raw, 2)
    df = pd.DataFrame(data)
    if messy:
        cols = [c for c in df.columns if c != "Property_ID"]
        for c in cols:
            miss = rng.random(n) < 0.015
            df.loc[miss, c] = np.nan
            out = rng.random(n) < 0.007
            lo, hi = SYN_RANGES.get(c, (0, 100))
            df.loc[out, c] = hi + 2 * (hi - lo)
    return df


# --------------------------------------------------------------------------
# 3. STATISTICAL HELPERS
# --------------------------------------------------------------------------
def kmo_bartlett(R: np.ndarray, n: int) -> dict:
    p = R.shape[0]
    try:
        inv = np.linalg.pinv(R)
        d = np.sqrt(np.outer(np.diag(inv), np.diag(inv)))
        P = -inv / d
        np.fill_diagonal(P, 0)
        R0 = R.copy()
        np.fill_diagonal(R0, 0)
        kmo = float((R0 ** 2).sum() / ((R0 ** 2).sum() + (P ** 2).sum()))
        sign, logdet = np.linalg.slogdet(R)
        chi2 = float(-(n - 1 - (2 * p + 5) / 6) * logdet) if sign > 0 else np.nan
        dfree = p * (p - 1) / 2
        pval = float(stats.chi2.sf(chi2, dfree)) if np.isfinite(chi2) else np.nan
    except Exception:
        kmo, chi2, pval = np.nan, np.nan, np.nan
    return dict(kmo=kmo, bartlett_chi2=chi2, bartlett_p=pval)


def kmo_label(k: float) -> str:
    """Kaiser's verbal labels for the KMO statistic."""
    if k is None or not np.isfinite(k):
        return "n/a"
    for lim, lab in [(0.9, "marvellous"), (0.8, "meritorious"), (0.7, "middling"),
                     (0.6, "mediocre"), (0.5, "miserable")]:
        if k >= lim:
            return lab
    return "unacceptable"


def entropy_detail(X: np.ndarray):
    """Shannon-entropy objective weights.  X: n x p, non-negative (0-100 scores).
    Returns (P, e, d, w):  P = column shares, e = entropy, d = 1 - e (divergence), w = d / sum(d)."""
    n, p = X.shape
    Xs = X + 1e-9
    P = Xs / Xs.sum(axis=0, keepdims=True)
    e = -(P * np.log(P)).sum(axis=0) / np.log(n)
    d = 1 - e
    s = d.sum()
    w = d / s if s > 1e-12 else np.full(p, 1 / p)
    return P, e, d, w


def entropy_weights(X: np.ndarray) -> np.ndarray:
    return entropy_detail(X)[3]


def _ev_detail(S: np.ndarray) -> dict:
    """Explained-variance weights: PCA on the standardised score matrix, keep Kaiser
    components (eigenvalue > 1); weight_j  ∝  Σ_k EVR_k * v_jk²   (then normalised to 1)."""
    m = S.shape[1]
    if m == 1:
        return dict(w=np.ones(1), evals=np.ones(1), k=1, evr=np.ones(1), V=np.ones((1, 1)))
    sd = S.std(axis=0, ddof=1)
    act = sd > 1e-9
    w = np.zeros(m)
    if act.sum() < 2:
        w[:] = 1 / m
        return dict(w=w, evals=np.ones(m), k=1, evr=np.full(m, 1 / m), V=np.eye(m))
    Z = (S[:, act] - S[:, act].mean(0)) / sd[act]
    R = np.corrcoef(Z, rowvar=False)
    evals, evecs = np.linalg.eigh(R)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    k = max(1, int((evals > 1).sum()))
    evr = evals / evals.sum()
    wt = (evecs[:, :k] ** 2 * evr[:k]).sum(axis=1)
    w[act] = wt / wt.sum()
    return dict(w=w, evals=evals, k=k, evr=evr, V=evecs)


def _ev_weights(S: np.ndarray) -> np.ndarray:
    return _ev_detail(S)["w"]


def _derive(X: np.ndarray, alpha: float, kmax: int) -> dict:
    """Steps: correlation -> factor analysis -> loadings -> entropy -> combined weights."""
    n, p = X.shape
    sd = X.std(axis=0, ddof=1)
    active = sd > 1e-9
    pa = int(active.sum())
    if pa < 2:                                   # nothing to factor-analyse: equal weights
        w = active.astype(float)
        w = w / w.sum() if w.sum() > 0 else np.full(p, 1 / p)
        return dict(w=w, w_load=w.copy(), w_ent=w.copy(), L=np.zeros((p, 1)), k=1,
                    eig=np.ones(max(pa, 1)), R=np.eye(max(pa, 1)), active=active,
                    comm=np.zeros(p), var_explained=np.nan)
    Xa = X[:, active]
    Z = (Xa - Xa.mean(0)) / sd[active]
    R = np.corrcoef(Z, rowvar=False)
    eig = np.sort(np.linalg.eigvalsh(R))[::-1]
    k = int(np.clip((eig > 1).sum(), 1, max(1, min(kmax, pa // 3))))
    fa = FactorAnalysis(n_components=k, rotation="varimax" if k > 1 else None,
                        random_state=0, max_iter=500)
    fa.fit(Z)
    L = fa.components_.T                                    # pa x k  (loadings)
    h = np.clip((L ** 2).sum(axis=1), 1e-6, None)            # communality (sign-invariant)
    w_load = np.zeros(p)
    w_load[active] = h / h.sum()
    w_ent = np.zeros(p)
    w_ent[active] = entropy_weights(Xa)
    comb = np.zeros(p)
    comb[active] = (w_load[active] ** alpha) * (w_ent[active] ** (1 - alpha))
    comb = comb / comb.sum()
    Lfull = np.zeros((p, k))
    Lfull[active] = L
    comm = np.zeros(p)
    comm[active] = h
    return dict(w=comb, w_load=w_load, w_ent=w_ent, L=Lfull, k=k, eig=eig, R=R,
                active=active, comm=comm, var_explained=float(h.sum() / pa))


def _sub_scores(X: np.ndarray, w: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    cols = []
    for g in groups:
        wg = w[g]
        tot = wg.sum()
        cols.append((X[:, g] * wg).sum(1) / tot if tot > 1e-12 else X[:, g].mean(1))
    return np.column_stack(cols)


def effective_weights(w: np.ndarray, sub_w: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    """Weight each parameter really carries in the factor score:
    effective_j = W_s * w_j / Σ_{k in s} w_k   (sums to 1; factor score = Σ effective_j * x_j)."""
    eff = np.zeros(len(w))
    for g, s in zip(groups, sub_w):
        tot = w[g].sum()
        eff[g] = s * (w[g] / tot if tot > 1e-12 else 1 / len(g))
    return eff


def band(score: float) -> str:
    if score >= 80:
        return "Excellent"
    if score >= 65:
        return "Good"
    if score >= 50:
        return "Moderate"
    if score >= 35:
        return "Below average"
    return "Poor"


def _ci(boot: np.ndarray, level: float):
    a = (1 - level) / 2 * 100
    return np.percentile(boot, a, axis=0), np.percentile(boot, 100 - a, axis=0)


# --------------------------------------------------------------------------
# 4. DATA PREPROCESSING  (one function used by fitting AND by the worked example)
# --------------------------------------------------------------------------
def preprocess(raw: pd.DataFrame, direction: pd.Series, bmin: pd.Series, bmax: pd.Series) -> dict:
    """
    Stage 2 - Data preprocessing, in this exact order:
      A. Missing values  -> replaced by the column MEDIAN (robust to outliers)
      B. Outliers        -> winsorised (clipped) to Tukey fences  Q1-1.5*IQR  and  Q3+1.5*IQR
                            (fences are computed on the observed values, before imputation;
                             if IQR = 0 the fences are the column min / max, i.e. no clipping)
      C. Scaling         -> min-max to 0-100:   100 * (x - min) / (max - min)
                            min / max = benchmark values if the user gave them, else data min / max
      D. Direction       -> if "lower is better":  score = 100 - score, so 100 is ALWAYS the best
    """
    cols = list(raw.columns)
    q1, q3 = raw.quantile(.25), raw.quantile(.75)
    iqr = q3 - q1
    fence_lo = pd.Series(np.where(iqr > 0, q1 - 1.5 * iqr, raw.min()), index=cols)
    fence_hi = pd.Series(np.where(iqr > 0, q3 + 1.5 * iqr, raw.max()), index=cols)
    median = raw.median()
    below, above = (raw < fence_lo).sum(), (raw > fence_hi).sum()
    clean_stats = pd.DataFrame({
        "median": median, "q1": q1, "q3": q3, "lo": fence_lo, "hi": fence_hi,
        "raw_min": raw.min(), "raw_max": raw.max(),
        "missing": raw.isna().sum(), "missing_pct": raw.isna().mean() * 100,
        "clipped_low": below, "clipped_high": above, "outliers_clipped": below + above}, index=cols)

    filled = raw.fillna(median)
    cleaned = pd.DataFrame(np.clip(filled.values, fence_lo.values, fence_hi.values),
                           columns=cols, index=raw.index)

    bmin = bmin.reindex(cols).astype(float)
    bmax = bmax.reindex(cols).astype(float)
    nmin = bmin.fillna(cleaned.min())
    nmax = bmax.fillna(cleaned.max())
    span = (nmax - nmin).where((nmax - nmin) > 0)               # NaN if range is zero / invalid
    scaled = ((cleaned - nmin) / span * 100).fillna(50).clip(0, 100)
    Xn = scaled.copy()
    lower_cols = [c for c in cols if direction[c] == "lower"]
    Xn[lower_cols] = 100 - Xn[lower_cols]
    norm_stats = pd.DataFrame({
        "min": nmin, "max": nmax, "direction": direction.reindex(cols),
        "bounds": np.where(bmin.notna() | bmax.notna(), "benchmark", "data")}, index=cols)
    return dict(clean_stats=clean_stats, cleaned=cleaned, scaled=scaled, Xn=Xn, norm_stats=norm_stats)


# --------------------------------------------------------------------------
# 5. FACTOR RESULT
# --------------------------------------------------------------------------
@dataclass
class FactorResult:
    code: str
    name: str
    columns: list
    sub_map: dict                    # sub-factor -> [column names]
    clean_stats: pd.DataFrame
    norm_stats: pd.DataFrame
    raw: pd.DataFrame
    cleaned: pd.DataFrame
    scaled: pd.DataFrame             # 0-100 BEFORE the direction flip
    Xn: pd.DataFrame                 # 0-100 AFTER the direction flip (100 = best)
    corr: pd.DataFrame
    eigenvalues: np.ndarray
    n_factors: int
    loadings: pd.DataFrame
    weights: pd.DataFrame
    sub_scores: pd.DataFrame
    sub_weights: pd.Series
    score: pd.Series
    boot_scores: np.ndarray          # B x n   (full-data scores under each bootstrap weight set)
    boot_w: np.ndarray               # B x p
    boot_subw: np.ndarray            # B x m
    diagnostics: dict
    flags: list = field(default_factory=list)
    id_col: pd.Series | None = None

    # ----- scoring of NEW rows -----
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the SAME preprocessing (medians, fences, min/max, direction) learned on the reference data."""
        num = df[self.columns].apply(pd.to_numeric, errors="coerce")
        num = num.fillna(self.clean_stats["median"].reindex(self.columns))
        arr = np.clip(num.values, self.clean_stats["lo"].reindex(self.columns).values,
                      self.clean_stats["hi"].reindex(self.columns).values)
        lo = self.norm_stats["min"].reindex(self.columns).values
        hi = self.norm_stats["max"].reindex(self.columns).values
        span = np.where(hi - lo > 0, hi - lo, np.nan)
        x = np.clip(np.nan_to_num((arr - lo) / span * 100, nan=50.0), 0, 100)
        lower = (self.norm_stats["direction"].reindex(self.columns) == "lower").values
        x[:, lower] = 100 - x[:, lower]
        return pd.DataFrame(x, columns=self.columns, index=df.index)

    def out_of_range(self, df: pd.DataFrame) -> pd.Series:
        """How many supplied values fall outside the reference scaling range (they get capped at 0 / 100)."""
        num = df[self.columns].apply(pd.to_numeric, errors="coerce")
        lo = self.norm_stats["min"].reindex(self.columns)
        hi = self.norm_stats["max"].reindex(self.columns)
        return ((num < lo) | (num > hi)).sum(axis=1)

    def apply(self, Xn: pd.DataFrame) -> dict:
        """Score already-normalised rows with the fitted weights."""
        groups = [np.array([self.columns.index(c) for c in cols]) for cols in self.sub_map.values()]
        X = Xn[self.columns].values
        w = self.weights["final_weight"].values
        S = _sub_scores(X, w, groups)
        score = S @ self.sub_weights.values
        boot = np.zeros((len(self.boot_w), len(X)))
        for b in range(len(self.boot_w)):
            boot[b] = _sub_scores(X, self.boot_w[b], groups) @ self.boot_subw[b]
        return dict(sub_scores=pd.DataFrame(S, columns=list(self.sub_map), index=Xn.index),
                    score=pd.Series(score, index=Xn.index), boot=boot)

    def ci(self, level: float = 0.95):
        if len(self.boot_scores) == 0:
            return self.score.copy(), self.score.copy()
        lo, hi = _ci(self.boot_scores, level)
        return (pd.Series(np.minimum(lo, self.score), index=self.score.index),
                pd.Series(np.maximum(hi, self.score), index=self.score.index))


# --------------------------------------------------------------------------
# 6. FIT ONE FACTOR
# --------------------------------------------------------------------------
def fit_factor(df: pd.DataFrame, code: str, config: pd.DataFrame, boot_idx: np.ndarray,
               alpha: float = 0.5, kmax: int = 4) -> FactorResult | None:
    f = HIERARCHY[code]
    cfg = config[(config["Code"] == code) & (config["Column"].isin(df.columns))]
    if len(cfg) < 3:
        return None
    columns = list(cfg["Column"])
    sub_map: dict[str, list] = {}
    for sub in f["subs"]:
        cols = list(cfg.loc[cfg["Subfactor"] == sub, "Column"])
        if cols:
            sub_map[sub] = cols
    n = len(df)
    raw = df[columns].apply(pd.to_numeric, errors="coerce")

    # ---- Stage 2: DATA PREPROCESSING
    cfg_i = cfg.set_index("Column")
    pre = preprocess(raw, cfg_i["Direction"].reindex(columns),
                     cfg_i["Min"].reindex(columns), cfg_i["Max"].reindex(columns))
    clean_stats, cleaned, scaled, Xn, norm_stats = (pre["clean_stats"], pre["cleaned"], pre["scaled"],
                                                    pre["Xn"], pre["norm_stats"])
    X = Xn.values
    p = X.shape[1]
    groups = [np.array([columns.index(c) for c in cols]) for cols in sub_map.values()]

    # ---- Stages 3-6: correlation, FA, loadings, entropy (base fit on the full data)
    base = _derive(X, alpha, kmax)
    corr = pd.DataFrame(np.corrcoef(X, rowvar=False), index=columns, columns=columns).fillna(0)
    diag = kmo_bartlett(base["R"], n)
    diag["var_explained"] = base["var_explained"]

    # ---- Stage 7: BOOTSTRAP VALIDATION
    B = len(boot_idx)
    boot_w = np.zeros((B, p))
    boot_subw = np.zeros((B, len(groups)))
    boot_scores = np.zeros((B, n))
    base_sw = _ev_weights(_sub_scores(X, base["w"], groups))
    for b in range(B):
        try:
            Xb = X[boot_idx[b]]
            d = _derive(Xb, alpha, kmax)
            sw = _ev_weights(_sub_scores(Xb, d["w"], groups))
            boot_w[b], boot_subw[b] = d["w"], sw
        except Exception:
            boot_w[b], boot_subw[b] = base["w"], base_sw
        boot_scores[b] = _sub_scores(X, boot_w[b], groups) @ boot_subw[b]
    if B > 0:
        w_final = boot_w.mean(0)
        w_final /= w_final.sum()
        wlo, whi = np.percentile(boot_w, 2.5, axis=0), np.percentile(boot_w, 97.5, axis=0)
        wstd = boot_w.std(0)
    else:
        w_final, wlo, whi, wstd = base["w"], base["w"], base["w"], np.zeros(p)

    # ---- Stages 8-10: parameter weights -> sub-factor scores -> EV weights -> factor score
    S = _sub_scores(X, w_final, groups)
    ev = _ev_detail(S)
    sw_final = ev["w"]
    score = S @ sw_final
    eff = effective_weights(w_final, sw_final, groups)

    sub_of = {c: s for s, cs in sub_map.items() for c in cs}
    within = np.zeros(p)
    sub_w_per_param = np.zeros(p)
    for g, s in zip(groups, sw_final):
        tot = w_final[g].sum()
        within[g] = w_final[g] / tot if tot > 1e-12 else 1 / len(g)
        sub_w_per_param[g] = s
    weights = pd.DataFrame({
        "parameter": [short_name(c) for c in columns],
        "subfactor": [sub_of[c] for c in columns],
        "direction": norm_stats["direction"].values,
        "communality": base["comm"], "loading_weight": base["w_load"],
        "entropy_weight": base["w_ent"], "combined_weight": base["w"],
        "bootstrap_mean": w_final, "bootstrap_std": wstd,
        "ci_lo": wlo, "ci_hi": whi,
        "cv": np.where(w_final > 0, wstd / np.maximum(w_final, 1e-12), np.nan),
        "final_weight": w_final,
        "within_sub_weight": within, "subfactor_weight": sub_w_per_param,
        "effective_weight": eff}, index=columns)
    loadings = pd.DataFrame(base["L"], index=columns,
                            columns=[f"Factor {i + 1}" for i in range(base["k"])])
    result = FactorResult(
        code=code, name=f["name"], columns=columns, sub_map=sub_map,
        clean_stats=clean_stats, norm_stats=norm_stats, raw=raw, cleaned=cleaned, scaled=scaled, Xn=Xn,
        corr=corr, eigenvalues=base["eig"], n_factors=base["k"], loadings=loadings,
        weights=weights, sub_scores=pd.DataFrame(S, columns=list(sub_map), index=df.index),
        sub_weights=pd.Series(sw_final, index=list(sub_map)),
        score=pd.Series(score, index=df.index), boot_scores=boot_scores,
        boot_w=boot_w, boot_subw=boot_subw, diagnostics=diag)
    result.flags = diagnose(result, n)
    return result


def diagnose(r: FactorResult, n: int) -> list:
    """Automatic flaw detector -> list of (severity, message)."""
    flags, p = [], len(r.columns)
    if n < 5 * p:
        flags.append(("high", f"Sample size is small for factor analysis (n={n}, p={p}, n/p={n / p:.1f}; "
                              "rule of thumb ≥5-10). Loadings may be unstable."))
    k = r.diagnostics.get("kmo")
    if k is not None and np.isfinite(k):
        if k < 0.5:
            flags.append(("high", f"KMO = {k:.2f} (<0.50): data are NOT suitable for factor analysis."))
        elif k < 0.6:
            flags.append(("medium", f"KMO = {k:.2f}: only 'miserable/mediocre' sampling adequacy."))
    bp = r.diagnostics.get("bartlett_p")
    if bp is not None and np.isfinite(bp) and bp > 0.05:
        flags.append(("high", f"Bartlett's test p = {bp:.3f}: correlation matrix is not different from identity."))
    cs = r.clean_stats
    for c in cs.index[cs["missing_pct"] > 10]:
        flags.append(("medium", f"'{short_name(c)}' has {cs.loc[c, 'missing_pct']:.0f}% missing values (median-imputed)."))
    for c in cs.index[cs["outliers_clipped"] > max(3, 0.05 * n)]:
        flags.append(("medium", f"'{short_name(c)}' had {int(cs.loc[c, 'outliers_clipped'])} outliers clipped (>5% of rows)."))
    zero = [short_name(c) for c in r.columns if r.Xn[c].std() < 1e-9]
    if zero:
        flags.append(("high", f"Zero-variance parameters (carry no information, weight = 0): {', '.join(zero)}"))
    iu = np.triu_indices(p, 1)
    cm = r.corr.values
    hi = [(r.columns[i], r.columns[j], cm[i, j]) for i, j in zip(*iu) if abs(cm[i, j]) > 0.9]
    for a, b, v in hi[:5]:
        flags.append(("medium", f"Near-duplicate parameters (r={v:.2f}): '{short_name(a)}' and '{short_name(b)}' – double counting."))
    eff = r.weights["effective_weight"]
    limit = max(0.15, 2.5 / p)
    if eff.max() > limit:
        flags.append(("medium", f"Weight dominance: '{short_name(eff.idxmax())}' carries {eff.max() * 100:.1f}% "
                                "of the factor score."))
    unstable = r.weights.index[(r.weights["cv"] > 0.6) & (r.weights["final_weight"] > 0)]
    if len(unstable):
        flags.append(("medium", f"{len(unstable)} parameter weights are unstable under bootstrap (CV>0.6)."))
    for sub, cols in r.sub_map.items():
        if len(cols) > 1:
            sc = r.corr.loc[cols, cols].values
            for i, c in enumerate(cols):
                m = (sc[i].sum() - 1) / (len(cols) - 1)
                if m < -0.1:
                    flags.append(("medium", f"'{short_name(c)}' correlates NEGATIVELY with its sub-factor peers "
                                            f"(mean r={m:.2f}) – check its direction (higher/lower is better)."))
    lo, hi_ = r.ci(0.95)
    width = float((hi_ - lo).mean())
    if width > 15:
        flags.append(("medium", f"Average 95% CI width is {width:.1f} points – scores are imprecise."))
    return flags


# --------------------------------------------------------------------------
# 7. RUN ALL FACTORS + OVERALL SCORE
# --------------------------------------------------------------------------
def run_all(df: pd.DataFrame, config: pd.DataFrame, alpha: float = 0.5, B: int = 150,
            seed: int = 7, kmax: int = 4) -> dict:
    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, len(df), size=(B, len(df)))
    out = {}
    for code in HIERARCHY:
        r = fit_factor(df, code, config, boot_idx, alpha, kmax)
        if r is not None:
            out[code] = r
    return out


def resolve_factor_weights(codes: list, factor_weights: dict | None):
    """Return a normalised weight vector. Falls back to equal weights if all weights are zero."""
    w = np.array([factor_weights.get(c, 0.0) for c in codes], dtype=float)
    if not np.isfinite(w).all() or w.sum() <= 0:
        w = np.ones(len(codes))
    return w / w.sum()


def combine_overall(results: dict, factor_weights: dict | None, level: float = 0.95):
    """factor_weights=None -> data-driven explained-variance weights across factors."""
    codes = list(results)
    F = pd.DataFrame({c: results[c].score for c in codes})
    w = _ev_weights(F.values) if factor_weights is None else resolve_factor_weights(codes, factor_weights)
    overall = pd.Series(F.values @ w, index=F.index, name="Wellness Score")
    boots = np.stack([results[c].boot_scores for c in codes], axis=2)   # B x n x m
    ob = boots @ w
    if len(ob):
        lo, hi = _ci(ob, level)
        lo, hi = np.minimum(lo, overall.values), np.maximum(hi, overall.values)
    else:
        lo = hi = overall.values
    return F, pd.Series(w, index=codes, name="weight"), overall, pd.Series(lo, index=F.index), pd.Series(hi, index=F.index)


# --------------------------------------------------------------------------
# 8. TRACING ONE PROPERTY  (worked example on real data)
# --------------------------------------------------------------------------
def trace_property(r: FactorResult, row) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Follow one property through every stage of one factor. Returns (parameter table, sub-factor table)."""
    rows = []
    for c in r.columns:
        raw_v, clean_v = r.raw.loc[row, c], r.cleaned.loc[row, c]
        if pd.isna(raw_v):
            status = "missing → median"
        elif not np.isclose(raw_v, clean_v):
            status = "outlier → clipped"
        else:
            status = "unchanged"
        w = r.weights.loc[c]
        rows.append({"Parameter": short_name(c), "Sub-factor": w["subfactor"],
                     "Raw value": raw_v, "Preprocessing": status, "After cleaning": clean_v,
                     "Scale min": r.norm_stats.loc[c, "min"], "Scale max": r.norm_stats.loc[c, "max"],
                     "Better when": DIRECTION_LABEL[r.norm_stats.loc[c, "direction"]],
                     "Score 0-100": r.Xn.loc[row, c],
                     "Weight inside sub-factor": w["within_sub_weight"],
                     "Effective weight": w["effective_weight"],
                     "Contribution to factor score": w["effective_weight"] * r.Xn.loc[row, c]})
    ptab = pd.DataFrame(rows)
    stab = pd.DataFrame({"Sub-factor": list(r.sub_map),
                         "Sub-factor score": [r.sub_scores.loc[row, s] for s in r.sub_map],
                         "Sub-factor weight": [r.sub_weights[s] for s in r.sub_map]})
    stab["Contribution to factor score"] = stab["Sub-factor score"] * stab["Sub-factor weight"]
    return ptab, stab


# --------------------------------------------------------------------------
# 9. EXPLANATIONS
# --------------------------------------------------------------------------
def explain_factor(r: FactorResult, level: float = 0.95) -> str:
    n, p = r.Xn.shape
    cs, W = r.clean_stats, r.weights
    lo, hi = r.ci(level)
    d = r.diagnostics
    top = W.sort_values("effective_weight", ascending=False)
    t1 = W.sort_values("loading_weight", ascending=False).iloc[0]
    t2 = W.sort_values("entropy_weight", ascending=False).iloc[0]
    star = top.iloc[0]
    n_bench = int((r.norm_stats["bounds"] == "benchmark").sum())
    n_inv = int((r.norm_stats["direction"] == "lower").sum())
    miss_c = cs["missing"].idxmax()
    clip_c = cs["outliers_clipped"].idxmax()
    iu = np.triu_indices(p, 1)
    cm = r.corr.values
    mean_abs = float(np.abs(cm[iu]).mean()) if p > 1 else float("nan")
    j = np.argmax(np.abs(cm[iu])) if p > 1 else 0
    pair = (r.columns[iu[0][j]], r.columns[iu[1][j]], cm[iu[0][j], iu[1][j]])

    L = [f"### {r.name}: how the score was derived",
         f"**Step 1 · Raw data.** {n} properties × {p} parameters, grouped in {len(r.sub_map)} sub-factors "
         f"({', '.join(r.sub_map)}).",

         "**Step 2 · Data preprocessing** (makes parameters comparable and safe to model):\n"
         f"- *Missing values:* {int(cs['missing'].sum())} empty cells ({cs['missing'].sum() / (n * p) * 100:.1f}% of all cells) "
         f"were filled with the column **median**. Most affected: '{short_name(miss_c)}' ({int(cs.loc[miss_c, 'missing'])}).\n"
         f"- *Outliers:* {int(cs['outliers_clipped'].sum())} extreme values were clipped to the **Tukey fences** "
         f"(Q1 − 1.5×IQR, Q3 + 1.5×IQR). Most affected: '{short_name(clip_c)}' ({int(cs.loc[clip_c, 'outliers_clipped'])}).\n"
         f"- *Scaling:* each parameter was rescaled to 0-100 with min-max "
         f"(100 × (x − min) ÷ (max − min)); {n_bench} parameter(s) use your benchmark min/max, "
         f"{p - n_bench} use the data's own min/max.\n"
         f"- *Direction:* {n_inv} 'lower is better' parameter(s) were flipped (score = 100 − scaled value), so "
         "**100 always means best**.",

         f"**Step 3 · Correlation.** Average absolute correlation between parameters = {mean_abs:.2f}. "
         f"Strongest pair: '{short_name(pair[0])}' and '{short_name(pair[1])}' (r = {pair[2]:.2f}).",

         f"**Step 4 · Factor analysis.** KMO = {d['kmo']:.2f} ({kmo_label(d['kmo'])}), "
         f"Bartlett p = {d['bartlett_p']:.3g}. Kaiser rule (eigenvalue > 1) kept **{r.n_factors} hidden factor(s)**"
         f"{', varimax-rotated' if r.n_factors > 1 else ''}, which together explain about "
         f"**{d['var_explained'] * 100:.0f}%** of the parameters' variance.",

         "**Steps 5-6 · Two ways of judging importance.**\n"
         "- *Loading (communality) weight* – how strongly a parameter is tied to the hidden factors. "
         f"Highest: '{t1['parameter']}' ({t1['loading_weight'] * 100:.1f}%).\n"
         "- *Entropy weight* – how much a parameter differs from property to property (a parameter that is "
         f"the same everywhere cannot rank anything). Highest: '{t2['parameter']}' ({t2['entropy_weight'] * 100:.1f}%).",

         f"**Step 7 · Bootstrap.** Steps 3-6 were repeated on {len(r.boot_w)} random resamples of the properties; "
         "the parameter weight is the average over all resamples, and the spread tells you how stable it is.",

         "**Step 8 · Parameter weights.** Top parameters by effective weight: " + "; ".join(
             f"{row['parameter']} {row['effective_weight'] * 100:.1f}%" for _, row in top.head(5).iterrows()),

         "**Step 9 · Sub-factor scores.** Each sub-factor score is the weighted mean of its parameters. "
         "Sub-factor weights (explained variance): " + "; ".join(
             f"{s} {v * 100:.0f}%" for s, v in r.sub_weights.sort_values(ascending=False).items()),

         f"**Step 10 · Factor score.** Mean {r.score.mean():.1f} (min {r.score.min():.1f}, max {r.score.max():.1f}); "
         f"average {level * 100:.0f}% confidence-interval width {(hi - lo).mean():.1f} points.",

         "**Worked weight example** for the most influential parameter "
         f"**{star['parameter']}** (sub-factor *{star['subfactor']}*):\n"
         f"- communality h = {star['communality']:.3f} → loading weight = {star['loading_weight'] * 100:.2f}%\n"
         f"- entropy weight = {star['entropy_weight'] * 100:.2f}%\n"
         f"- combined = loading^α × entropy^(1−α), normalised → {star['combined_weight'] * 100:.2f}%\n"
         f"- bootstrap mean = **{star['final_weight'] * 100:.2f}%** (95% interval {star['ci_lo'] * 100:.2f}% – {star['ci_hi'] * 100:.2f}%)\n"
         f"- share inside its sub-factor = {star['within_sub_weight'] * 100:.1f}%, sub-factor weight = "
         f"{star['subfactor_weight'] * 100:.1f}% → **effective weight in the factor score = "
         f"{star['effective_weight'] * 100:.2f}%**"]
    if r.flags:
        L.append("**Possible flaws detected:**\n" + "\n".join(f"- ({s}) {m}" for s, m in r.flags))
    else:
        L.append("**No automatic data-quality flags raised.**")
    return "\n\n".join(L)


def _drags(results: dict, norm: dict, fw: pd.Series, row, top: int = 5):
    """Parameters that cost the overall score the most:  factor weight × effective weight × shortfall."""
    out = []
    for c, r in results.items():
        x = norm[c].loc[row, r.columns].values
        gap = fw[c] * r.weights["effective_weight"].values * (100 - x)
        out += [(col, g, v) for col, g, v in zip(r.columns, gap, x)]
    out.sort(key=lambda t: -t[1])
    return out[:top]


def explain_property(results: dict, F: pd.DataFrame, fw: pd.Series, overall: pd.Series,
                     lo: pd.Series, hi: pd.Series, row, profile: str, level: float = 0.95) -> str:
    L = [f"### Property {row}: Wellness Score **{overall[row]:.1f}** "
         f"({band(overall[row])}), {level * 100:.0f}% CI [{lo[row]:.1f}, {hi[row]:.1f}]",
         f"Overall weighting: **{profile}**."]
    L.append("**Factor scores (score, weight in the overall score):**\n" + "\n".join(
        f"- {results[c].name}: {F.loc[row, c]:.1f} (weight {fw[c] * 100:.1f}%)" for c in F.columns))
    subs = []
    for c, r in results.items():
        for s in r.sub_map:
            subs.append((r.name, s, r.sub_scores.loc[row, s]))
    subs.sort(key=lambda t: t[2])
    L.append("**Weakest sub-factors:** " + "; ".join(f"{a} / {b} {v:.0f}" for a, b, v in subs[:3]))
    L.append("**Strongest sub-factors:** " + "; ".join(f"{a} / {b} {v:.0f}" for a, b, v in subs[-3:][::-1]))
    norm = {c: r.Xn for c, r in results.items()}
    L.append("**Biggest drags on the score (weight × shortfall from 100):** " + "; ".join(
        f"{short_name(c)} (score {v:.0f}/100)" for c, _, v in _drags(results, norm, fw, row)))
    return "\n\n".join(L)


def explain_new_property(results: dict, new: dict, fw: pd.Series, row, level: float = 0.95) -> str:
    """Plain-language explanation for a NEW property (output of score_new)."""
    F, overall, lo, hi = new["F"], new["overall"], new["lo"], new["hi"]
    ref_mean = {c: results[c].score.mean() for c in results}
    L = [f"### Why this property scores {overall[row]:.1f} ({band(overall[row])})",
         f"The {level * 100:.0f}% confidence interval is [{lo[row]:.1f}, {hi[row]:.1f}]. "
         f"You supplied **{new['completeness'][row] * 100:.0f}%** of the parameters"
         + (f"; {int(new['out_of_range'][row])} value(s) lie outside the reference range and were capped at 0 or 100."
            if new['out_of_range'][row] else "; every value lies inside the reference range.")
         + " Missing values are replaced by the reference median, exactly as during training."]
    L.append("**Factor scores vs the reference average:**\n" + "\n".join(
        f"- {results[c].name}: {F.loc[row, c]:.1f} ({F.loc[row, c] - ref_mean[c]:+.1f} vs average "
        f"{ref_mean[c]:.1f}); weight {fw[c] * 100:.1f}%" for c in F.columns))
    subs = []
    for c, r in results.items():
        for s in r.sub_map:
            subs.append((r.name, s, new["subs"][c].loc[row, s]))
    subs.sort(key=lambda t: t[2])
    L.append("**Weakest sub-factors:** " + "; ".join(f"{a} / {b} {v:.0f}" for a, b, v in subs[:3]))
    L.append("**Strongest sub-factors:** " + "; ".join(f"{a} / {b} {v:.0f}" for a, b, v in subs[-3:][::-1]))
    L.append("**Biggest drags on the score:** " + "; ".join(
        f"{short_name(c)} (score {v:.0f}/100)" for c, _, v in _drags(results, new["norm"], fw, row)))
    return "\n\n".join(L)


def llm_explain(context: str, token: str, model: str, question: str = "") -> str:
    """Ask a Groq-hosted LLM (OpenAI-compatible API) to narrate the derivation and critique it."""
    import requests
    system = ("You are a careful data-science reviewer for a property-wellness scoring system. "
              "Explain, in plain language for a property buyer, how the score was derived from the facts "
              "provided. Then give a short critique: possible methodological weaknesses and what to check. "
              "Use ONLY the numbers in the context; never invent values.")
    user = f"CONTEXT:\n{context}\n\nTASK: {question or 'Explain the derivation and highlight any flaws.'}"
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"model": model,
              "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
              "max_tokens": 1200, "temperature": 0.2},
        timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"Groq API error {resp.status_code}: {resp.text[:400]}")
    return resp.json()["choices"][0]["message"]["content"]


# --------------------------------------------------------------------------
# 10. SCORING NEW PROPERTIES
# --------------------------------------------------------------------------
def score_new(results: dict, df_new: pd.DataFrame, factor_weights: pd.Series, level: float = 0.95) -> dict:
    """Score properties that are NOT in the reference set, with the already-fitted model
    (medians, Tukey fences, scaling bounds, parameter weights, sub-factor weights, bootstrap weights).
    factor_weights: pd.Series indexed by factor code (as returned by combine_overall)."""
    codes = list(results)
    F, boots, subs, norm = {}, [], {}, {}
    oor = pd.Series(0, index=df_new.index)
    all_cols = [c for c in dict.fromkeys(c for r in results.values() for c in r.columns)]
    given = df_new.reindex(columns=all_cols).apply(pd.to_numeric, errors="coerce")
    completeness = given.notna().mean(axis=1)
    for c in codes:
        r = results[c]
        sub = df_new.reindex(columns=r.columns)
        Xn = r.transform(sub)
        a = r.apply(Xn)
        F[c], subs[c], norm[c] = a["score"], a["sub_scores"], Xn
        boots.append(a["boot"])
        oor = oor + r.out_of_range(sub)
    F = pd.DataFrame(F)
    w = factor_weights.reindex(codes).values
    overall = pd.Series(F.values @ w, index=F.index, name="Wellness Score")
    ob = np.stack(boots, axis=2) @ w
    if len(ob):
        lo, hi = _ci(ob, level)
        lo, hi = np.minimum(lo, overall.values), np.maximum(hi, overall.values)
    else:
        lo = hi = overall.values
    return dict(F=F, overall=overall, lo=pd.Series(lo, index=F.index), hi=pd.Series(hi, index=F.index),
                subs=subs, norm=norm, completeness=completeness, out_of_range=oor)


# --------------------------------------------------------------------------
# 11. TOY EXAMPLE  (small enough to follow by hand; every number comes from the real functions)
# --------------------------------------------------------------------------
TOY_RAW = pd.DataFrame({
    "AQI value": [62, 85, np.nan, 120, 48, 95, 410, 70],
    "Pollutant levels": [30, 44, 38, 70, 22, 52, 90, 35],
    "Noise levels (dB)": [52, 60, 58, 78, 48, 66, 72, 55],
    "% Green cover": [34, 22, 25, 8, 40, 15, 12, 30]},
    index=[f"P{i}" for i in range(1, 9)])
TOY_SUBS = {"Air Quality": ["AQI value", "Pollutant levels"], "Surroundings": ["Noise levels (dB)", "% Green cover"]}
TOY_DIR = {"AQI value": "lower", "Pollutant levels": "lower", "Noise levels (dB)": "lower", "% Green cover": "higher"}


def toy_example(alpha: float = 0.5, B: int = 20, seed: int = 3) -> dict:
    raw = TOY_RAW.copy()
    cols = list(raw.columns)
    nan = pd.Series(np.nan, index=cols)
    pre = preprocess(raw, pd.Series(TOY_DIR), nan, nan)
    X = pre["Xn"].values
    groups = [np.array([cols.index(c) for c in v]) for v in TOY_SUBS.values()]
    P, e, d, w_ent = entropy_detail(X)
    base = _derive(X, alpha, 4)
    idx = np.random.default_rng(seed).integers(0, len(X), size=(B, len(X)))
    bw = np.zeros((B, len(cols)))
    for b in range(B):
        try:
            bw[b] = _derive(X[idx[b]], alpha, 4)["w"]
        except Exception:
            bw[b] = base["w"]
    w_final = bw.mean(0)
    w_final /= w_final.sum()
    S = _sub_scores(X, w_final, groups)
    ev = _ev_detail(S)
    eff = effective_weights(w_final, ev["w"], groups)
    sub_names = list(TOY_SUBS)
    wt = pd.DataFrame({
        "Communality h": base["comm"], "Loading weight": base["w_load"],
        "Entropy e": e, "Divergence d = 1 − e": d, "Entropy weight": base["w_ent"],
        "Combined weight": base["w"], "Bootstrap mean (parameter weight)": w_final,
        "Effective weight": eff}, index=cols)
    return dict(
        raw=raw, pre=pre, alpha=alpha, B=B,
        shares=pd.DataFrame(P, index=raw.index, columns=cols),
        corr=pd.DataFrame(np.corrcoef(X, rowvar=False), index=cols, columns=cols),
        eig=base["eig"], k=base["k"],
        loadings=pd.DataFrame(base["L"], index=cols, columns=[f"Factor {i + 1}" for i in range(base["k"])]),
        weights=wt,
        boot=pd.DataFrame(bw[:6], columns=cols, index=[f"Resample {i + 1}" for i in range(min(6, B))]),
        sub_scores=pd.DataFrame(S, index=raw.index, columns=sub_names),
        sub_weights=pd.Series(ev["w"], index=sub_names),
        score=pd.Series(S @ ev["w"], index=raw.index), groups=TOY_SUBS)
