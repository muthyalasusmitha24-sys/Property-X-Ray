
    lines = [
        f"### {r.name}: how the score was calculated",
        f"**Data used.** {n} properties and {p} parameters were available across {len(r.sub_map)} sub-factors.",
        (f"**1. Data preprocessing.** {int(cs['missing'].sum())} missing/non-numeric cells were median-imputed. "
         f"{int(cs['outliers_clipped'].sum())} observed values were clipped to the Tukey 1.5×IQR fences. "
         "The fences flag unusually distant values; clipping limits their influence but does not prove they were errors."),
        (f"**2. Normalisation.** Each parameter was mapped to a 0–100 quality scale. "
         "Configured benchmark bounds are used when supplied; otherwise cleaned training-data bounds are used. "
         "Parameters marked 'lower is better' are reversed so that a higher score always means a better assessed condition."),
        (f"**3. Correlation and factor analysis.** The correlation matrix checks which parameters move together. "
         f"KMO = {kmo_text}; Bartlett's test p-value = {bartlett_text}. These are suitability diagnostics, not proof "
         "that the resulting score is valid. The retained factor count uses a Kaiser eigenvalue>1 rule, capped by "
         f"the configured limit; {r.n_factors} factor(s) were retained. Selected eigenvalues account for about "
         f"{pca_screen_share:.1f}% of the total correlation-matrix eigenvalue sum (a PCA screening summary, not a claim "
         "that the common-factor model explains that exact percentage of observed variance)."),
        ("**4. Communality-based weight.** For parameter i, communality is the sum of its squared loadings across "
         "the retained common factors: hᵢ = Σₖ loadingᵢₖ². It indicates how much of that parameter's standardized "
         "variation is represented by the retained factors. Communalities are normalized to sum to 1."),
        ("**5. Entropy-based weight.** Within each parameter, normalized scores across properties are treated as a "
         "distribution. Entropy near 1 means values are relatively evenly spread; lower entropy means the distribution "
         "is more concentrated. The objective weight is proportional to 1−entropy and is normalized to sum to 1. "
         "This rewards variation in the dataset; it does not measure desirability or buyer importance."),
        ("**6. Combined parameter weight.** The two normalized weights are blended geometrically: combinedᵢ ∝ "
         "loading_weightᵢ^α × entropy_weightᵢ^(1−α), then normalized to sum to 1. With α=0.5, both sources contribute "
         "equally on the geometric scale. α closer to 1 emphasizes communality; α closer to 0 emphasizes entropy."),
        (f"**7. Bootstrap stability check.** Weight estimation was repeated on {len(r.boot_w)} resamples, each formed by "
         "sampling property rows with replacement. Final parameter weights are normalized means across resamples. "
         "Standard deviation, percentile interval, and coefficient of variation show sensitivity to the sampled properties."),
        ("**8. Sub-factor scores.** For each property, parameters within a sub-factor are combined using their final "
         "parameter weights, renormalized within that sub-factor. This produces a 0–100 score for each sub-factor."),
        ("**9. Sub-factor weights.** PCA is applied to standardized sub-factor scores. Components with eigenvalue>1 are "
         "retained; each sub-factor's contribution is calculated from retained component variance shares and squared "
         "eigenvector coefficients, then normalized. These are data-driven weights, not expert judgments."),
        (f"**10. Factor score.** The factor score is the weighted sum of its sub-factor scores. Mean score = "
         f"{r.score.mean():.1f}; observed range = {r.score.min():.1f}–{r.score.max():.1f}."),
        (f"**Uncertainty interval.** The displayed {level * 100:.0f}% interval is derived from bootstrap-based weight "
         f"variation; its average width is {(hi - lo).mean():.1f} score points. It does not include measurement error, "
         "uncertainty in input data, or uncertainty about the chosen model/design."),
        "**Highest final parameter weights:** " + "; ".join(
            f"{i.split(': ', 1)[1]} {v * 100:.1f}%" for i, v in zip(w.index[:5], w["final_weight"][:5])),
        "**Sub-factor weights:** " + "; ".join(
            f"{name} {value * 100:.1f}%" for name, value in r.sub_weights.sort_values(ascending=False).items()),
    ]
    if r.flags:
        lines.append("**Checks to review:**")
        lines.extend(f"- {severity.upper()}: {message}" for severity, message in r.flags)
    else:
        lines.append("**Automatic checks:** no configured warning was triggered. This does not establish that the method or inputs are correct.")
    return "\n\n".join(lines)


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
    w = factor_weights.reindex(codes).fillna(0).to_numpy(dtype=float)
    w = np.where(np.isfinite(w) & (w >= 0), w, 0.0)
    if w.sum() <= 1e-12:
        raise ValueError("The fitted factor weights must contain at least one positive value.")
    w = w / w.sum()
    overall = pd.Series(F.values @ w, index=F.index, name="Wellness Score")
    ob = np.stack(boots, axis=2) @ w
    if len(ob):
        lo, hi = _ci(ob, level)
        lo, hi = np.minimum(lo, overall.values), np.maximum(hi, overall.values)
    else:
        lo = hi = overall.values
    return F, overall, pd.Series(lo, index=F.index), pd.Series(hi, index=F.index), subs
