                             "or try another model name (e.g. llama-3.1-8b-instant).")
 
# ------------------------------------------------------------------ TAB 5: method
with tabs[5]:
    st.markdown("""
## Method: what happens at each stage

The pipeline is fitted separately for each wellness factor. The overall score combines the resulting factor scores.

### Flow of the calculation

```text
Property table (rows = properties, columns = raw parameters)
        ↓
Data preprocessing: numeric conversion → median imputation → Tukey-fence clipping
        ↓
Direction-aware normalisation to 0–100
        ↓
Pearson correlation matrix → KMO / Bartlett diagnostics
        ↓
Exploratory factor analysis → loading-based parameter weights
        ↓
Entropy-based parameter weights → geometric-mean blend
        ↓
Bootstrap refitting → mean final parameter weights + score intervals
        ↓
Weighted parameter scores within each sub-factor
        ↓
PCA-derived sub-factor weights → wellness-factor score (0–100)
        ↓
Factor weighting method → overall Wellness Score (0–100) + percentile interval
```

### 1. Data preprocessing

- **Numeric conversion:** values such as `"not available"` cannot be used numerically and become missing values.
- **Median imputation:** a missing value is replaced by the median of that column in the reference dataset. Median is less affected by extreme observations than the mean.
- **Outlier clipping (winsorising):** calculate Q1, Q3 and IQR = Q3−Q1. Values below Q1−1.5×IQR are replaced by that lower fence; values above Q3+1.5×IQR are replaced by the upper fence. The row is retained. A flagged value is not automatically wrong; check its source.
- **Important:** preprocessing statistics are learned from the reference dataset and reused for new properties. New properties do not redefine the medians or fences.

### 2. Normalisation and direction

For a parameter with lower bound `min` and upper bound `max`:

`mapped = 100 × (raw − min) / (max − min)`

The result is clipped to 0–100. A configured benchmark bound is used where supplied; otherwise, the cleaned reference-data minimum/maximum is used. For a parameter where **lower is better**, the final score is `100 − mapped`. After direction correction, higher always means more favourable according to the chosen rule. Direction choices are assumptions that must match the meaning and units of the real data.

### 3. Correlation and factor analysis

- **Pearson correlation (`r`):** measures the strength and direction of a linear relationship between two parameters. `r=+1` means they move together perfectly; `r=−1` means they move in opposite directions perfectly; `r≈0` means little linear relationship. Correlation does not prove causation.
- **KMO:** assesses whether the pattern of correlations is suitable for factor analysis; closer to 1 is better, and values below about 0.5 are generally poor.
- **Bartlett's test:** tests whether the correlation matrix differs from an identity matrix. A small p-value suggests enough correlations exist to consider factor analysis.
- **Eigenvalue / scree plot:** an eigenvalue describes how much variance a component of the correlation matrix accounts for. The Kaiser rule proposes retaining eigenvalues greater than 1. The code also applies a maximum-factor cap; if no eigenvalue passes the rule, it retains one factor to allow a score to be calculated. Treat this as a modelling choice, not proof that the retained factors are the uniquely correct answer.
- **Factor loading:** indicates how strongly a parameter is represented by a latent factor. The sign gives direction; magnitude gives strength. The heatmap shows these values, not final parameter weights.

### 4. How parameter weights are derived

**A. Loading-based weight.** For parameter `i`, square each loading and add them across retained factors:

`hᵢ = Σₖ loadingᵢₖ²`

Then normalise: `loading_weightᵢ = hᵢ / Σᵢ hᵢ`. Squaring means positive and negative loadings both contribute by magnitude. This weight reflects representation in the fitted factor structure.

**B. Entropy-based weight.** For each parameter, turn its non-negative scores across `n` properties into proportions:

`pᵢⱼ = scoreᵢⱼ / Σᵢ scoreᵢⱼ`

Then calculate normalised entropy:

`entropyⱼ = −Σᵢ pᵢⱼ ln(pᵢⱼ) / ln(n)`

Finally, `entropy_weightⱼ ∝ 1 − entropyⱼ`, normalised to sum to 1. A parameter with more variation across the properties tends to have a lower entropy and a higher objective weight. This means it distinguishes properties in this dataset; it does **not** mean it is inherently more important to every buyer.

**C. Combined weight.** The sidebar value α controls the blend:

`combinedᵢ ∝ loading_weightᵢ^α × entropy_weightᵢ^(1−α)`

The combined weights are normalised to sum to 1. α=0 means entropy only; α=1 means loading-based only; α=0.5 gives both equal exponents. These weights are data-dependent and can change if the reference dataset changes.

**D. Bootstrap.** Sample properties with replacement and refit the weighting steps repeatedly. The final parameter weight is the average across the successful resamples. The distribution of scores from these bootstrap weight sets gives a percentile interval. This interval reflects sensitivity to resampling the reference properties; it does not include every possible source of uncertainty.

### 5. From parameter scores to the overall score

1. **Sub-factor score:** take the weighted mean of the normalised parameters belonging to that sub-factor. The weights are renormalised within each sub-factor.
2. **Sub-factor weights:** apply PCA to standardised sub-factor scores. Retain components with eigenvalue > 1. A sub-factor's weight is proportional to the sum of (component explained-variance ratio × squared component loading), then weights are normalised.
3. **Wellness-factor score:** weighted sum of its sub-factor scores. The result stays on a 0–100 scale.
4. **Overall Wellness Score:** weighted sum of the available wellness-factor scores. The selected factor weights are normalised to sum to 100%.

### Overall factor-weighting choices

- **Balanced (equal weights):** each available factor contributes equally. This is transparent and does not claim one factor matters more than another.
- **Data-driven (explained variance):** PCA-based weights are derived from how factor scores vary together in the current reference dataset. It is not automatically more correct or more fair than equal weights.
- **Custom:** you choose relative factor importance. These weights express your priorities; they are not learned from data.

The demographic presets (for example, “retiree” or “investor”) are intentionally not offered here because their multipliers are subjective assumptions and may not fit an individual buyer. If you use custom weights, document why you chose them.

### Worked example (illustrative numbers only)

Suppose a parameter has raw value 40, with benchmark min=0 and max=100, and higher is better:

`mapped = 100 × (40−0)/(100−0) = 40`

If lower is better, reverse it: `100−40 = 60`.

Suppose a sub-factor contains two already-normalised parameter scores, 80 and 40, with weights 0.6 and 0.4:

`sub-factor score = (80×0.6 + 40×0.4)/(0.6+0.4) = 64`

Suppose three factor scores are 70, 60 and 80 with balanced weights 1/3 each:

`overall = 70/3 + 60/3 + 80/3 = 70`

These are toy numbers to explain arithmetic, not outputs from your uploaded data.

### Interpretation and limitations

- A score is a **constructed comparative index**, not a property valuation, health assessment, legal certification, or guarantee of future outcomes.
- Data-driven weights depend on the reference dataset and may change with the sample. Custom weights reflect preferences. Equal weights are a simple baseline.
- Missing data, incorrect units, wrong direction settings, small samples, weak KMO, non-significant Bartlett test, unstable bootstrap weights, or highly correlated duplicate inputs can undermine interpretation.
- If real data is used, verify parameter definitions, measurement units, direction and benchmark bounds before relying on the score.
""")
