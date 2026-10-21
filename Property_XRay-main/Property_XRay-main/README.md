Property X-Ray — Wellness Score
A Streamlit application for exploring a transparent 0–100 property-wellness index.
Files
- app.py — revised Streamlit interface, clearer configuration labels, weighting choices, method explanations, and chart readability improvements.
- wellness_engine.py — scoring engine.
- theme.py — visual theme and shared chart/UI helpers.
- requirements.txt — Python dependencies.
- METHODOLOGY.md — detailed pipeline, formulas, examples, assumptions, and interpretation guide.
Important factor-code convention
- EMO = Emotional Wellness.
- ENV = Environmental Wellness.
Do not use EMO to label Environmental Wellness.
Scoring pipeline
Raw property data → data preprocessing → direction-aware 0–100 normalization → correlation diagnostics → factor analysis → communality and entropy weights → geometric weight combination → bootstrap re-estimation → parameter-weighted sub-factor scores → PCA/eigenvalue-based sub-factor weights → factor scores → overall score.

Read METHODOLOGY.md before interpreting the weights. In particular, data-driven weights measure variation in the reference dataset; they do not determine what matters most to a buyer.
Run locally
Use Python 3.11 or 3.12 for broad compatibility.
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
The command requires app.py. It was not among the files supplied for this review, so the existing UI labels, configuration table, and overlapping charts cannot be verified or fixed in this bundle until that file is provided.
Input data
Use one row per property and parameter columns named CODE: Parameter, such as ENV: AQI value. An optional Property_ID can label rows. Numeric measurements should use consistent units. Categorical values need documented numeric encoding before scoring.
Known limitations
- If benchmark bounds are absent, normalization is relative to cleaned training-data minima and maxima.
- The reported uncertainty interval represents bootstrap weighting uncertainty, not measurement error or total model uncertainty.
- Factor-analysis and PCA-derived weights are data-dependent, not expert judgements.
- The default synthetic data are only for exercising the pipeline and are not evidence about real properties.
