# BMS Fucosylation · Dynamic ML Pipeline Dashboard

## Folder Structure

```
bms-pipeline/
├── backend/
│   ├── main.py              ← FastAPI backend (all logic, training, cleansing)
│   ├── requirements.txt
│   └── .env                 ← ADD YOUR GEMINI KEY HERE
│
└── frontend/
    ├── package.json
    └── src/
        ├── App.jsx           ← Full pipeline UI
        ├── index.js
        └── index.css
```

## Setup

### 1. Add Gemini API Key

Edit `backend/.env`:
```
GEMINI_API_KEY=AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

### 2. Start Backend (Terminal 1)

```bash
cd backend
conda activate msds597   # or your env
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Verify at http://localhost:8000

### 3. Start Frontend (Terminal 2)

```bash
cd frontend
npm install
npm start
```

Opens at http://localhost:3000

---

## Pipeline Flow

1. **Upload & Cleanse** — Upload any CSV. Detects missing values (MCAR/MNAR), outliers (IQR + Z-score + IsolationForest), batch drift (KS test). No imputation.

2. **Select & Train** — Re-upload CSV, pick models from all 7. Backend trains them live on YOUR data, 80/20 split.

3. **Results Dashboard** — PowerBI-style: KPI cards, R² bar chart, actual vs predicted scatter, SHAP feature importance, multi-criteria radar, learning curves. Toggle any model on/off.

4. **AI Report** — Gemini 2.5 Flash generates 800-1000+ word regulatory narrative from actual results. Export as PDF.

---

## Dataset Format

Your CSV must have:
- Feature columns (any names)
- `Fucosylation_pct` as target (or last column is used)
- Optional: `Batch_ID` for drift analysis

Works with your existing BMS2 CSV: `data/mab_fucosylation_dataset.csv`
