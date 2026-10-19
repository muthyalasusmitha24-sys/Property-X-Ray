"""
wellness_engine.py  --  Property X-Ray : Wellness Score engine
=================================================================
Pipeline (applied independently to each of the 9 wellness factors):
 
    Raw parameters -> Cleaning -> Normalisation -> Correlation matrix
    -> Factor analysis -> Factor loadings -> Entropy weighting
    -> Bootstrap validation -> Parameter weights -> Sub-factor scores
    -> Explained-variance weights -> Factor score
    -> Confidence interval
 
Then the 9 factor scores are combined into the overall Wellness Score
(equal / buyer-profile / data-driven weights) with a confidence interval.
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
 
BUYER_PROFILES = {
    "Balanced (equal weights)": dict(EMO=1, PHY=1, SOC=1, OCC=1, ENV=1, INT=1, SPI=1, FIN=1, STA=1),
    "Young family": dict(EMO=1.0, PHY=1.2, SOC=1.3, OCC=0.8, ENV=1.3, INT=1.4, SPI=0.7, FIN=1.0, STA=1.2),
    "Retiree": dict(EMO=1.3, PHY=1.5, SOC=1.2, OCC=0.3, ENV=1.3, INT=0.6, SPI=1.4, FIN=0.8, STA=1.0),
    "Working professional": dict(EMO=1.0, PHY=1.0, SOC=0.9, OCC=1.6, ENV=1.0, INT=0.8, SPI=0.6, FIN=1.2, STA=1.0),
    "Investor": dict(EMO=0.3, PHY=0.6, SOC=0.6, OCC=1.0, ENV=0.8, INT=0.5, SPI=0.3, FIN=2.0, STA=1.6),
}
 
 
def col_name(code: str, label: str) -> str:
    return f"{code}: {label}"
 
 
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
# 2. SYNTHETIC DATA  (100 rows x all parameter columns, with a latent structure)
# --------------------------------------------------------------------------
def generate_sample_data(n: int = 100, seed: int = 42, messy: bool = True) -> pd.DataFrame:
    """Synthetic properties. Correlated within sub-factor/factor so factor analysis has
    real structure to find. Optionally injects missing values and outliers."""
    rng = np.random.default_rng(seed)
    cfg = default_config()
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
 
 
def entropy_weights(X: np.ndarray) -> np.ndarray:
    """Shannon-entropy objective weights. X: n x p, non-negative (0-100 scores)."""
    n, p = X.shape
    Xs = X + 1e-9
    P = Xs / Xs.sum(axis=0, keepdims=True)
    e = -(P * np.log(P)).sum(axis=0) / np.log(n)
    d = 1 - e
    s = d.sum()
    return d / s if s > 1e-12 else np.full(p, 1 / p)
 
 
def _ev_weights(S: np.ndarray) -> np.ndarray:
    """Explained-variance weights: PCA on the (standardised) score matrix, keep Kaiser
    components (eigenvalue > 1); weight_j ∝ Σ_k EVR_k * v_jk^2."""
    m = S.shape[1]
    if m == 1:
        return np.ones(1)
    sd = S.std(axis=0, ddof=1)
    act = sd > 1e-9
    w = np.zeros(m)
    if act.sum() < 2:
        w[:] = 1 / m
        return w
    Z = (S[:, act] - S[:, act].mean(0)) / sd[act]
    R = np.corrcoef(Z, rowvar=False)
    evals, evecs = np.linalg.eigh(R)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    k = max(1, int((evals > 1).sum()))
    evr = evals / evals.sum()
    wt = (evecs[:, :k] ** 2 * evr[:k]).sum(axis=1)
    w[act] = wt / wt.sum()
    return w
 
 
def _derive(X: np.ndarray, alpha: float, kmax: int) -> dict:
    """Steps: correlation -> factor analysis -> loadings -> entropy -> combined weights."""
    n, p = X.shape
    sd = X.std(axis=0, ddof=1)
    active = sd > 1e-9
    Xa = X[:, active]
    pa = Xa.shape[1]
    Z = (Xa - Xa.mean(0)) / sd[active]
    R = np.corrcoef(Z, rowvar=False)
    eig = np.sort(np.linalg.eigvalsh(R))[::-1]
    k = int(np.clip((eig > 1).sum(), 1, max(1, min(kmax, pa // 3))))
    fa = FactorAnalysis(n_components=k, rotation="varimax" if k > 1 else None,
                        random_state=0, max_iter=500)
    fa.fit(Z)
    L = fa.components_.T                                    # pa x k
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
                active=active, comm=comm)
 
 
def _sub_scores(X: np.ndarray, w: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    cols = []
    for g in groups:
        wg = w[g]
        tot = wg.sum()
        cols.append((X[:, g] * wg).sum(1) / tot if tot > 1e-12 else X[:, g].mean(1))
    return np.column_stack(cols)
 
 
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
# 4. FACTOR RESULT
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
    Xn: pd.DataFrame
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
 
    # ----- scoring -----
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        num = df[self.columns].apply(pd.to_numeric, errors="coerce")
        med = self.clean_stats["median"].reindex(self.columns)
        num = num.fillna(med)
        arr = np.clip(num.values, self.clean_stats["lo"].reindex(self.columns).values,
                      self.clean_stats["hi"].reindex(self.columns).values)
        lo = self.norm_stats["min"].reindex(self.columns).values
        hi = self.norm_stats["max"].reindex(self.columns).values
        rng = np.where(hi - lo == 0, np.nan, hi - lo)
        x = np.nan_to_num((arr - lo) / rng * 100, nan=50.0)
        x = np.clip(x, 0, 100)
        lower = (self.norm_stats["direction"].reindex(self.columns) == "lower").values
        x[:, lower] = 100 - x[:, lower]
        return pd.DataFrame(x, columns=self.columns, index=df.index)
 
    def apply(self, Xn: pd.DataFrame) -> dict:
        """Score already-normalised rows with the fitted weights."""
        groups = [np.array([self.columns.index(c) for c in cols]) for cols in self.sub_map.values()]
        X = Xn[self.columns].values
        w = self.weights["final_weight"].values
        S = _sub_scores(X, w, groups)
        sw = self.sub_weights.values
        score = S @ sw
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
# 5. FIT ONE FACTOR
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
 
    # ---- Step 2: CLEANING (coerce numeric, Tukey-fence winsorising, median imputation)
    q1, q3 = raw.quantile(.25), raw.quantile(.75)
    iqr = q3 - q1
    fence_lo = np.where(iqr > 0, q1 - 1.5 * iqr, raw.min())
    fence_hi = np.where(iqr > 0, q3 + 1.5 * iqr, raw.max())
    clean_stats = pd.DataFrame({
        "median": raw.median(), "lo": fence_lo, "hi": fence_hi,
        "missing": raw.isna().sum(), "missing_pct": raw.isna().mean() * 100,
        "outliers_clipped": ((raw < fence_lo) | (raw > fence_hi)).sum()}, index=columns)
    cleaned = raw.fillna(clean_stats["median"])
    cleaned = pd.DataFrame(np.clip(cleaned.values, fence_lo, fence_hi), columns=columns, index=df.index)
 
    # ---- Step 3: NORMALISATION (min-max 0-100, direction aware, optional benchmark bounds)
    cfg_i = cfg.set_index("Column")
    mn = cfg_i["Min"].reindex(columns).astype(float)
    mx = cfg_i["Max"].reindex(columns).astype(float)
    nmin = mn.fillna(cleaned.min())
    nmax = mx.fillna(cleaned.max())
    norm_stats = pd.DataFrame({"min": nmin, "max": nmax,
                               "direction": cfg_i["Direction"].reindex(columns),
                               "bounds": np.where(mn.notna() | mx.notna(), "benchmark", "data")}, index=columns)
    rng_ = (nmax - nmin).replace(0, np.nan)
    Xn = ((cleaned - nmin) / rng_ * 100).fillna(50).clip(0, 100)
    lower_cols = [c for c in columns if norm_stats.loc[c, "direction"] == "lower"]
    Xn[lower_cols] = 100 - Xn[lower_cols]
 
    X = Xn.values
    p = X.shape[1]
    groups = [np.array([columns.index(c) for c in cols]) for cols in sub_map.values()]
 
    # ---- Steps 4-8: correlation, FA, loadings, entropy (base fit on full data)
    base = _derive(X, alpha, kmax)
    corr = pd.DataFrame(np.corrcoef(X, rowvar=False), index=columns, columns=columns).fillna(0)
    diag = kmo_bartlett(base["R"], n)
 
    # ---- Step 9: BOOTSTRAP VALIDATION
    B = len(boot_idx)
    boot_w = np.zeros((B, p))
    boot_subw = np.zeros((B, len(groups)))
    boot_scores = np.zeros((B, n))
    base_S = _sub_scores(X, base["w"], groups)
    base_sw = _ev_weights(base_S)
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
 
    # ---- Steps 10-13: parameter weights -> sub-factor scores -> EV weights -> factor score
    S = _sub_scores(X, w_final, groups)
    sw_final = _ev_weights(S)
    score = S @ sw_final
 
    short = [c.split(": ", 1)[1] for c in columns]
    weights = pd.DataFrame({
        "parameter": short,
        "subfactor": [next(s for s, cs in sub_map.items() if c in cs) for c in columns],
        "direction": norm_stats["direction"].values,
        "communality": base["comm"], "loading_weight": base["w_load"],
        "entropy_weight": base["w_ent"], "combined_weight": base["w"],
        "bootstrap_mean": w_final, "bootstrap_std": wstd,
        "ci_lo": wlo, "ci_hi": whi,
        "cv": np.where(w_final > 0, wstd / np.maximum(w_final, 1e-12), np.nan),
        "final_weight": w_final}, index=columns)
    loadings = pd.DataFrame(base["L"], index=columns,
                            columns=[f"Factor {i + 1}" for i in range(base["k"])])
    result = FactorResult(
        code=code, name=f["name"], columns=columns, sub_map=sub_map,
        clean_stats=clean_stats, norm_stats=norm_stats, raw=raw, cleaned=cleaned, Xn=Xn,
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
        flags.append(("medium", f"'{c}' has {cs.loc[c, 'missing_pct']:.0f}% missing values (median-imputed)."))
    for c in cs.index[cs["outliers_clipped"] > max(3, 0.05 * n)]:
        flags.append(("medium", f"'{c}' had {int(cs.loc[c, 'outliers_clipped'])} outliers clipped (>5% of rows)."))
    zero = [c for c in r.columns if r.Xn[c].std() < 1e-9]
    if zero:
        flags.append(("high", f"Zero-variance parameters (carry no information, weight = 0): {', '.join(zero)}"))
    iu = np.triu_indices(p, 1)
    cm = r.corr.values
    hi = [(r.columns[i], r.columns[j], cm[i, j]) for i, j in zip(*iu) if abs(cm[i, j]) > 0.9]
    for a, b, v in hi[:5]:
        flags.append(("medium", f"Near-duplicate parameters (r={v:.2f}): '{a}' and '{b}' – double counting."))
    w = r.weights["final_weight"]
    if w.max() > 0.15:
        flags.append(("medium", f"Weight dominance: '{w.idxmax()}' carries {w.max() * 100:.1f}% of the factor weight."))
    unstable = r.weights.index[(r.weights["cv"] > 0.6) & (r.weights["final_weight"] > 0)]
    if len(unstable):
        flags.append(("medium", f"{len(unstable)} parameter weights are unstable under bootstrap (CV>0.6)."))
    for sub, cols in r.sub_map.items():
        if len(cols) > 1:
            sc = r.corr.loc[cols, cols].values
            for i, c in enumerate(cols):
                m = (sc[i].sum() - 1) / (len(cols) - 1)
                if m < -0.1:
                    flags.append(("medium", f"'{c}' correlates NEGATIVELY with its sub-factor peers (mean r={m:.2f}) – "
                                            "check its direction (higher/lower is better)."))
    lo, hi_ = r.ci(0.95)
    width = float((hi_ - lo).mean())
    if width > 15:
        flags.append(("medium", f"Average 95% CI width is {width:.1f} points – scores are imprecise."))
    return flags
 
 
# --------------------------------------------------------------------------
# 6. RUN ALL FACTORS + OVERALL SCORE
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
 
 
def combine_overall(results: dict, factor_weights: dict | None, level: float = 0.95):
    """factor_weights=None -> data-driven explained-variance weights across factors."""
    codes = list(results)
    F = pd.DataFrame({c: results[c].score for c in codes})
    if factor_weights is None:
        w = _ev_weights(F.values)
    else:
        w = np.array([factor_weights.get(c, 0.0) for c in codes], dtype=float)
        w = w / w.sum()
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
# 7. EXPLANATIONS
# --------------------------------------------------------------------------
def explain_factor(r: FactorResult, level: float = 0.95) -> str:
    n, p = r.Xn.shape
    cs, w = r.clean_stats, r.weights.sort_values("final_weight", ascending=False)
    lo, hi = r.ci(level)
    var_k = r.eigenvalues[:r.n_factors].sum() / r.eigenvalues.sum() * 100
    L = [f"### {r.name} – how the score was derived",
         f"**1. Data.** {n} properties × {p} parameters in {len(r.sub_map)} sub-factors.",
         f"**2. Cleaning.** {int(cs['missing'].sum())} missing cells were filled with the column median; "
         f"{int(cs['outliers_clipped'].sum())} outliers were clipped to Tukey fences (1.5×IQR).",
         f"**3. Normalisation.** Every parameter was scaled to 0-100 (min-max, using benchmark bounds where "
         f"given). {int((r.norm_stats['direction'] == 'lower').sum())} 'lower is better' parameters were inverted.",
         f"**4-5. Correlation & factor analysis.** KMO = {r.diagnostics['kmo']:.2f}, "
         f"Bartlett p = {r.diagnostics['bartlett_p']:.3g}. {r.n_factors} common factor(s) retained "
         f"(Kaiser eigenvalue>1, varimax) explaining ≈{var_k:.0f}% of variance.",
         "**6-7. Loadings → entropy.** Parameter importance = communality (Σ loading²) blended with "
         "Shannon-entropy dispersion weights (geometric mean).",
         "**8-9. Bootstrap.** The whole weighting procedure was repeated on "
         f"{len(r.boot_w)} resamples; final weights are the bootstrap means.",
         "**Top parameter weights:** " + "; ".join(
             f"{i.split(': ', 1)[1]} {v * 100:.1f}%" for i, v in zip(w.index[:5], w['final_weight'][:5])),
         "**10-11. Sub-factor scores and explained-variance weights:** " + "; ".join(
             f"{s} {v * 100:.0f}%" for s, v in r.sub_weights.sort_values(ascending=False).items()),
         f"**12. Factor score.** Mean {r.score.mean():.1f} (min {r.score.min():.1f}, max {r.score.max():.1f}); "
         f"average {level * 100:.0f}% CI width {(hi - lo).mean():.1f} points."]
    if r.flags:
        L.append("**⚠ Possible flaws detected:**")
        L += [f"- ({s}) {m}" for s, m in r.flags]
    else:
        L.append("**No automatic data-quality flags raised.**")
    return "\n\n".join(L)
 
 
def explain_property(results: dict, F: pd.DataFrame, fw: pd.Series, overall: pd.Series,
                     lo: pd.Series, hi: pd.Series, row, profile: str, level: float = 0.95) -> str:
    L = [f"### Property {row}: Wellness Score **{overall[row]:.1f}** "
         f"({band(overall[row])}), {level * 100:.0f}% CI [{lo[row]:.1f}, {hi[row]:.1f}]",
         f"Weight profile: **{profile}**.", "**Factor scores (weight → score):**"]
    for c in F.columns:
        L.append(f"- {results[c].name}: {F.loc[row, c]:.1f} (weight {fw[c] * 100:.1f}%)")
    subs = []
    for c, r in results.items():
        for s in r.sub_map:
            subs.append((r.name, s, r.sub_scores.loc[row, s], r.sub_weights[s]))
    subs.sort(key=lambda t: t[2])
    L.append("**Weakest sub-factors:** " + "; ".join(f"{a} / {b} {v:.0f}" for a, b, v, _ in subs[:3]))
    L.append("**Strongest sub-factors:** " + "; ".join(f"{a} / {b} {v:.0f}" for a, b, v, _ in subs[-3:][::-1]))
    drags = []
    for c, r in results.items():
        gap = r.weights["final_weight"] * (100 - r.Xn.loc[row, r.columns].values) * fw[c]
        for col, g in zip(r.columns, gap):
            drags.append((col, g, r.Xn.loc[row, col]))
    drags.sort(key=lambda t: -t[1])
    L.append("**Biggest drags on the score (weight × shortfall):** " + "; ".join(
        f"{c} (normalised {v:.0f}/100)" for c, _, v in drags[:5]))
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
 
 
def score_new(results: dict, df_new: pd.DataFrame, factor_weights, level: float = 0.95):
    """Score NEW properties with the already-fitted model (cleaning stats, normalisation bounds,
    parameter weights, sub-factor weights, bootstrap weights).
    factor_weights: pd.Series indexed by factor code (as returned by combine_overall)."""
    codes = list(results)
    F, boots, subs = {}, [], {}
    for c in codes:
        r = results[c]
        Xn = r.transform(df_new.reindex(columns=r.columns))
        a = r.apply(Xn)
        F[c] = a["score"]
        boots.append(a["boot"])
        subs[c] = a["sub_scores"]
    F = pd.DataFrame(F)
    w = factor_weights.reindex(codes).values
    overall = pd.Series(F.values @ w, index=F.index, name="Wellness Score")
    ob = np.stack(boots, axis=2) @ w
    if len(ob):
        lo, hi = _ci(ob, level)
        lo, hi = np.minimum(lo, overall.values), np.maximum(hi, overall.values)
    else:
        lo = hi = overall.values
    return F, overall, pd.Series(lo, index=F.index), pd.Series(hi, index=F.index), subs
