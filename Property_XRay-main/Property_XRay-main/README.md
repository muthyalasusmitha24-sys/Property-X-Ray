Property X-Ray · Wellness Score

A Streamlit app that turns raw property parameters into a transparent, statistically validated Wellness Score with a confidence interval.

Pipeline: cleaning → normalisation → correlation → factor analysis → entropy weights → bootstrap → parameter weights → sub-factor scores → explained-variance weights → factor scores → overall score + CI.

Project structure
app.py                       Streamlit UI (tabs, charts, tables)
theme.py                     Visual layer: palette, CSS, Plotly template, UI helpers
wellness_engine.py           Scoring engine (no UI code)
requirements.txt             Python dependencies
.streamlit/config.toml       Streamlit theme
.streamlit/secrets.toml.example   Template for the Groq API key
Run locally
bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
Optional: AI explanation

The Explanation tab can call a Groq-hosted model. Provide an API key in any one of these ways:

.streamlit/secrets.toml containing GROQ_API_KEY = "gsk_..." (copy from secrets.toml.example)
a GROQ_API_KEY environment variable
the sidebar field (kept only for the current session)

Never commit a real key.

Deploy on Streamlit Community Cloud
Push this repo to GitHub (keep app.py at the repo root).
At https://share.streamlit.io choose Create app, select the repo, branch and app.py.
Under Advanced settings pick Python 3.11 or 3.12 and paste your secrets (GROQ_API_KEY = "...").
Click Deploy.

If the free tier runs out of memory, lower the Bootstrap resamples slider or the number of rows.

Using your own data

Upload a CSV or Excel file with one row per property and one column per parameter, named CODE: Parameter (for example ENV: AQI value). Download the blank template from the Data tab. A factor needs at least 3 matching columns. Optional Property_ID column is used as the row label.

Known limitations
Min-max normalisation is relative to the reference set. Set benchmark Min/Max for absolute scoring.
The confidence interval reflects weighting uncertainty only, not measurement error.
Data-driven weights describe what varies in the data, not what matters to a buyer.
The bundled synthetic data exists only to exercise the pipeline.
