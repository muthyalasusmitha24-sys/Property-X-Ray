Property X-Ray — Wellness Score
A Streamlit application that converts property measurements into sub-factor scores, wellness-factor scores, and an overall model-defined Wellness Score.
Interpretation note: this is a data-dependent scoring index, not a certified property rating, probability, or guarantee of quality. Statistical diagnostics help assess the data but do not by themselves validate the score.

Files
- app.py — Streamlit user interface, input configuration, charts, and explanations.
- wellness_engine.py — preprocessing, normalization, factor analysis, parameter weights, bootstrap stability, and scoring.
- theme.py — existing visual styling and chart helpers (unchanged by this patch).
- requirements.txt — existing dependencies (unchanged by this patch).
- METHODOLOGY.md — detailed calculations, formulas, worked examples, flowchart, and limitations.
Run locally
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
Deploy through GitHub and Streamlit Community Cloud
1. Replace only the changed files supplied with this patch (app.py, wellness_engine.py, and METHODOLOGY.md) in the existing repository. README.md is optional documentation.
2. Keep theme.py and requirements.txt from your existing repository; they are not changed by this patch.
3. Commit and push to the branch connected to Streamlit Community Cloud. Streamlit will redeploy from the repository.
4. If deployment fails, read the first traceback in Streamlit's app logs. Do not change Python versions or dependency versions blindly; the current requirements already specify the project's dependencies.
Data format
Upload CSV or Excel data with one property per row and exact template column names such as ENV: AQI value. Property_ID is optional. Environmental parameters use the ENV: prefix. Emotional wellness uses EMO:. Environmental parameter labels accidentally prefixed with EMO: are remapped to ENV: where the label matches. Legacy EMO: Noise levels and ENV: Noise levels (dB) headers are normalized to ENV: Noise levels; keep measurement units consistent and do not include duplicate columns for the same measurement.

A factor is fitted when at least three configured columns are present, but the model also needs enough non-constant numeric data. If a selected column has no numeric values, the app displays an error instead of inventing data.
Weighting choices
- Balanced: equal weights across available wellness factors.
- Custom: user-defined relative weights, normalized to sum to one.
- Data-driven: PCA-derived weights based on the structure of the available factor scores.
Subjective persona presets were removed because their numerical values did not have a documented empirical or expert-elicitation basis.
Optional Groq explanation
The Explanation tab can request a narrative from Groq. Configure GROQ_API_KEY using Streamlit Secrets, an environment variable, or the session-only sidebar field. Never commit a real API key.
Main limitations
- Without benchmark bounds, normalization depends on the reference dataset.
- Bootstrap intervals represent weight-estimation uncertainty only, not measurement error or model-choice uncertainty.
- Data-derived weights reflect statistical variation, not universal buyer priorities.
- Synthetic data are for software testing only.
