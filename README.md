# Modeling Approaches for Antibody Afucosylation Mechanistic · Hybrid Models · AI · ML Pipeline Dashboard

> **BMS Hackathon** · MS Data Science · Rutgers University × Bristol Myers Squibb  
> Built with ♥ by **Team Data Minds** — Sanjith Ganesh & Pranav Senthilkumaran

A full-stack machine learning dashboard that predicts monoclonal antibody (mAb) fucosylation — a Critical Quality Attribute in biopharmaceutical manufacturing — from CHO cell culture bioprocess variables.

---

## What This Does

You upload a bioprocess dataset CSV, the pipeline automatically cleanses it, trains up to 7 ML models on your data, shows you a PowerBI-style results dashboard, and generates a regulatory-style AI report using Gemini. A separate static tab shows the BMS1 vs BMS2 comparison — how all 7 models performed across two dataset sizes (N=500 vs N=10,000).

---

## File Structure

```
bms-pipeline/
│
├── backend/
│   ├── main.py              ← FastAPI backend — all logic lives here
│   ├── requirements.txt     ← Python dependencies
│   └── .env                 ← Your Gemini API key (never commit this)
│
├── frontend/
│   ├── package.json         ← Node dependencies
│   ├── public/
│   │   ├── index.html       ← HTML entry point
│   │   └── bms-logo.png     ← BMS logo (add this yourself)
│   └── src/
│       ├── App.jsx          ← Entire frontend — all pages and components
│       ├── index.js         ← React entry point
│       └── index.css        ← Global styles and animations
│
└── README.md
```

**You only need to change:** `backend/.env` (API key) and `frontend/public/bms-logo.png` (logo image).

---

## Setup

### Requirements
- Python 3.9+ with conda or pip
- Node.js 18+ and npm
- A free Gemini API key from [aistudio.google.com](https://aistudio.google.com)

### Step 1 — Add your Gemini API key

Open `backend/.env`:
```
GEMINI_API_KEY=AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

### Step 2 — Add the BMS logo

Copy your BMS logo PNG to `frontend/public/bms-logo.png`. If missing, the header just shows blank — everything else works fine.

### Step 3 — Start the backend

```bash
cd backend
conda activate msds597        # or your environment
pip install -r requirements.txt   # first time only
uvicorn main:app --reload --port 8000
```

Verify at [http://localhost:8000](http://localhost:8000) — you'll see a JSON status message.

### Step 4 — Start the frontend

Open a **second terminal**:

```bash
cd frontend
npm install       # first time only
npm start
```

Opens automatically at [http://localhost:3000](http://localhost:3000).

---

## Pipeline — How It Works

### Upload & Cleanse
Drop your CSV. Three automated checks run instantly — no buttons to click:
- **Missing values** — finds blank cells, classifies MCAR vs MNAR pattern
- **Outlier detection** — IQR fence + Z-score + IsolationForest, all three at once
- **Batch drift** — KS test comparing each batch to the full dataset

Quality score out of 100. Five tabs of detail. Nothing is modified — inspection only.

### Select & Train
Your file carries over automatically (no re-upload). Pick any of the 7 models, click Train. The backend splits 80/20, scales features (training data only — no leakage), trains each model on your actual CSV, and computes R², RMSE, MAE on the real test set.

### Results Dashboard
Every number comes from your uploaded data — nothing hardcoded:
- KPI cards, model visibility toggles, R² bar chart
- Actual vs Predicted scatter per model
- SHAP values (XGBoost), VIP scores (PLSR), MDI importance (Random Forest)
- Multi-criteria radar and learning curves

### AI Report
Gemini 2.5 Flash writes a 600–700 word scientific summary from your actual results. Written for bioprocess scientists — all ML terms explained in plain language. Export as a formatted A4 PDF.

### BMS1 vs BMS2 (Static Tab)
Always accessible from the top navigation. Shows the static reference experiment: 7 models trained on BMS1 (N=500) vs BMS2 (N=10,000). Includes R² comparison chart, delta table, learning curves, and final rankings. The headline: ANN went from R²=0.05 (failure) to R²=0.77 (+1440%) simply by increasing dataset size.

---

## Dataset Format

| Column | Required | Notes |
|--------|----------|-------|
| `Fucosylation_pct` | Preferred | Target variable to predict |
| `Batch_ID` | Optional | Enables drift analysis |
| All other columns | Yes | Treated as input features |

If `Fucosylation_pct` is not found, the last column is used as the target. Works with any numerical bioprocess CSV.

---

## The 7 Models

| Model | Strength |
|-------|----------|
| Ridge Regression | Linear baseline, interpretable coefficients |
| PLSR | VIP scores, chemometrics gold standard |
| Random Forest | 200 trees, robust nonlinear capture |
| XGBoost + SHAP | Best accuracy + full explainability |
| GPR | Calibrated confidence intervals per prediction |
| ANN | Neural network — needs N > 5,000 to work reliably |
| Hybrid | Auto-engineers physics features from column names |


---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Backend | FastAPI, scikit-learn, XGBoost, SHAP, SciPy, Numpy, Pandas, httpx |
| Frontend | React 18, Chart.js, react-chartjs-2, axios, lucide-react |
| AI | Google Gemini 2.5 Flash |
| Font | DM Sans + DM Mono (Google Fonts) |

---

## Project Background

Built for the MS Data Science BMS Hackathon at Rutgers University × Bristol Myers Squibb. The experiment isolated dataset size as a single variable: BMS1 (N=500) and BMS2 (N=10,000) have identical features, noise, and ground truth — only the number of samples changed.

Key finding: the ANN failed completely at N=500 (12,225 parameters vs 340 training samples = 36× overparameterised, R²=0.05) and fully recovered at N=10,000 (R²=0.77, +1440%). Linear models hit a structural ceiling regardless of data volume. XGBoost+SHAP was the best production model at R²=0.83. GPR was best for small-N settings at R²=0.75 with perfectly calibrated uncertainty.

---

*Built with ♥ by Team Data Minds — Sanjith Ganesh & Pranav Senthilkumaran*
