# mAb Fucosylation · ML Pipeline Dashboard

> **BMS Hackathon** · MS Data Science · Rutgers University × Bristol Myers Squibb  
> Built with ♥ by **Team Data Minds** — Sanjith Ganesh & Pranav Senthilkumaran 

🎬 **[Watch Full ML Pipeline Demo Video](https://drive.google.com/file/d/1gwQlxT1jDECCQ1NTCU6lIFG9MACt6dyK/view?usp=sharing)**

🎬 **[Watch AI Report Generation Demo Video](https://drive.google.com/file/d/1yCnAgTfpVyjeYd479Y9KD86_1PqtpzMD/view?usp=sharing)**

A full-stack machine learning dashboard that predicts monoclonal antibody (mAb) fucosylation — a Critical Quality Attribute in biopharmaceutical manufacturing — from CHO cell culture bioprocess variables.

Upload CSV → automated cleansing → train 7 ML models → PowerBI-style results → AI-generated regulatory report.

---

## Platform Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  USER  (Browser · localhost:3000)                                   │
│  ┌──────────────────────┐  ┌───────────────┐  ┌──────────────────┐  │
│  │   Google OAuth       │  │   Email OTP   │  │  localStorage    │  │
│  │   One-click sign-in  │  │   Gmail SMTP  │  │  Token session   │  │
│  └──────────────────────┘  └───────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                              │ axios proxy
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  FRONTEND  React 18 · Chart.js · lucide-react                       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │
│  │ Upload & │ │ Select & │ │ Results  │ │ AI       │ │ BMS1 vs  │   │
│  │ Cleanse  │ │ Train    │ │ Dashboard│ │ Report   │ │ BMS2     │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                              │ HTTP
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  BACKEND  FastAPI · localhost:8000                                  │
│  ┌────────────┐ ┌──────────────────┐ ┌──────────────┐ ┌──────────┐  │
│  │  auth.py   │ │ /upload/cleanse  │ │ /upload/train│ │ /generate│  │
│  │ OAuth·OTP  │ │ IQR+Z+IsoForest  │ │ 7 ML models  │ │ -report  │  │
│  └────────────┘ └──────────────────┘ └──────────────┘ └──────────┘  │
└──────────┬──────────────────────────────────┬───────────────────────┘
           │                                  │
           ▼                                  ▼
┌────────────────────┐              ┌────────────────────────────────┐
│  Data Cleansing    │              │  7 ML Models (scikit-learn)    │
│  SciPy · pandas    │              │  Ridge · PLSR · Random Forest  │
│  Missing MCAR/MNAR │              │  XGBoost+SHAP · GPR            │
│  Outliers 3-method │              │  ANN · Hybrid (physics feats)  │
│  KS-test drift     │              │  R² · RMSE · MAE · SHAP · VIP  │
└────────────────────┘              └────────────────┬───────────────┘
                                                     │
                                                     ▼
                                    ┌────────────────────────────────┐
                                    │  Gemini 2.5 Flash              │
                                    │  Scientific · Executive        │
                                    │  Quick Summary · PDF Export    │
                                    └────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  DEPLOY  Docker · AWS App Runner/ECS · Azure Container Apps         │
└─────────────────────────────────────────────────────────────────────┘
```

**Tech Stack:**

| Layer | Tools |
|-------|-------|
| Frontend | React 18, Chart.js 4, react-chartjs-2, axios, lucide-react, DM Sans |
| Backend | FastAPI, uvicorn, python-dotenv, httpx |
| ML | scikit-learn, XGBoost, SHAP, SciPy, pandas, numpy |
| Auth | Google OAuth 2.0, Gmail SMTP |
| AI | Google Gemini 2.5 Flash (maxOutputTokens: 8192) |
| Deploy | Docker, AWS App Runner / ECS, Azure Container Apps |

---

## File Structure

```
bms-pipeline/
│
├── backend/
│   ├── main.py              ← FastAPI — ML pipeline + mounts auth routes
│   ├── auth.py              ← Google OAuth + Email OTP + Gmail SMTP
│   ├── requirements.txt
│   └── .env                 ← All credentials
│
├── frontend/
│   ├── package.json
│   ├── public/
│   │   ├── index.html
│   │   └── bms-logo.png
│   └── src/
│       ├── App.jsx          ← All pages: Login, Pipeline, Profile, Support, About
│       ├── index.js
│       └── index.css
│
├── BMS Midterm Presentation.pdf  ← Midterm Handoff
├── Results and comparison (BMS1 VS BMS2).docx
├── README.md
├── bms_platform_architecture.svg
│
├── BMS Dataset 1 (N=500)/                            ← Small-data regime (N=500), Data, Results, Models
│
├── BMS Dataset 2 (N=10,000)/                         ← Production-scale regime (N=10,000), Data, Results, Models
│
├── Final Project Report.pdf                          ← Final BMS Project 1 Report
│
└── BMS Final Presentation.pdf                        ← Final Presentation Slides

```

---

## Setup

### Requirements
- Python 3.9+ with conda or pip
- Node.js 18+ and npm
- Google Cloud project (for OAuth)
- Gmail account with App Password (for OTP + support emails)
- Gemini API key (free at [aistudio.google.com](https://aistudio.google.com))

### Step 1 — Fill in `.env`

```dotenv
GEMINI_API_KEY=AIzaSy...

GOOGLE_CLIENT_ID=1234567890.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=GOCSPX-...

GMAIL_ADDRESS=your@gmail.com
GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx

SECRET_KEY=<run: python3 -c "import secrets; print(secrets.token_hex(32))">

FRONTEND_URL=http://localhost:3000
BACKEND_URL=http://localhost:8000
```

### Step 2 — Google Cloud Console

1. [console.cloud.google.com](https://console.cloud.google.com) → APIs & Services → Credentials → OAuth 2.0 Client
2. Authorised JavaScript origins: `http://localhost:3000` and `http://localhost:8000`
3. Authorised redirect URI: `http://localhost:8000/auth/google/callback`
4. OAuth consent screen → Test users → add your Gmail

### Step 3 — Gmail App Password

Google Account → Security → 2-Step Verification → App Passwords → create one for Mail.

### Step 4 — Start backend

```bash
cd backend
conda activate msds597
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Verify: [http://localhost:8000/auth/debug](http://localhost:8000/auth/debug)

### Step 5 — Start frontend

```bash
cd frontend
npm install
npm start
```

Opens at [http://localhost:3000](http://localhost:3000)

---

## How to Use

### Login
Two options on one page — Google OAuth (one click) or Email OTP (6-digit code sent to inbox). Sessions persist 7 days via signed localStorage token.

### Step 1 — Upload & Cleanse
Drop your CSV. Three automated checks: missing value detection (MCAR/MNAR), outlier detection (IQR + Z-score + IsolationForest), batch drift (KS test). Quality score 0–100. No data modified.

### Step 2 — Select & Train
File carries over — no re-upload. Pick models, click Train. Backend splits 80/20, scales on training data only, trains selected models on your real CSV.

### Step 3 — Results Dashboard
Every number from your actual data:
- KPI cards, model visibility toggles
- R² bar chart, actual vs predicted scatter
- SHAP/VIP/MDI feature importance
- Multi-criteria radar, learning curves

### Step 4 — AI Report
Three report types:

| Type | Words | Audience |
|------|-------|----------|
| Scientific | 800–1000 | Bioprocess scientists, regulatory |
| Executive | 400–500 | VPs, C-suite, no jargon |
| Quick Summary | 200–250 | Team meetings, slide notes |

Export as formatted A4 PDF (ICH Q8/Q9 style).

### BMS1 vs BMS2 Tab
Static reference — 7 models across N=500 and N=10,000. ANN's +1440% recovery is the headline finding.

### Profile
Edit display name, upload photo, view account details, sign out.

### Support
Submit tickets by category. Track open/resolved status. Backend sends email via Gmail SMTP.

---

## Data Cleansing Pipeline

The cleansing step runs automatically the moment a CSV is uploaded, before any model sees the data. It performs three independent quality checks and computes an overall quality score out of 100. **No data is modified** — the pipeline is inspection-only, consistent with ICH Q8 which requires the manufacturer to document and justify data handling decisions.

### Step 1 — Missing Value Detection

Every cell in every column is scanned and the pattern of missingness is classified:

**MCAR — Missing Completely At Random:** The gap has no relationship to other variables. A sensor dropout during one batch, a skipped offline measurement. Safe to impute or drop without introducing bias.

**MNAR — Missing Not At Random:** The gap is correlated with other measurements. A pH sensor that stops recording when values go out of range, or a VCD measurement skipped only during high-stress batches. Dropping MNAR rows silently biases the model toward "normal" batches only — a significant risk in regulatory submissions.

The pipeline reports the count and pattern per column, and flags whether the missingness is random or systematic. Scientists can use this to identify measurement system problems before training.

| Output | Description |
|--------|-------------|
| Missing count per column | Number of blank cells per feature |
| MCAR / MNAR classification | Pattern of missingness — random vs systematic |
| Score impact | Missing values deduct from the quality score proportionally |

---

### Step 2 — Outlier Detection (Three Methods Simultaneously)

Three independent detectors run in parallel. Each flags different types of anomaly:

**IQR Fence (1.5×):** Flags values outside 1.5 times the interquartile range. Fast, non-parametric, and fully interpretable — a scientist can verify any flagged value by inspection. Catches univariate extremes such as a GDP-Fucose measurement of 0 or a temperature spike above the biological range.

**Z-Score |z| > 3:** Flags values more than 3 standard deviations from the column mean. Assumes approximate normality. Catches the same extreme univariate outliers as IQR but from a distributional perspective. Agreement between IQR and Z-score provides high confidence.

**IsolationForest (scikit-learn):** A machine learning detector that flags points which are easy to isolate from the rest of the dataset in the full feature space. Critically, this catches **multivariate outliers** — batches where pH, temperature, and VCD are each individually within range but their combination has never been seen before in the dataset. These are invisible to IQR and Z-score but can be the most scientifically important anomalies to investigate.

Results are reported separately per method and combined:

| Method | What it catches | Assumption |
|--------|-----------------|------------|
| IQR Fence | Extreme single-variable values | None (non-parametric) |
| Z-Score \|z\| > 3 | Extreme single-variable values | Approximate normality |
| IsolationForest | Unusual combinations across all features | None (tree-based) |
| Any Method | Union of all three — broadest flag | — |

**Interpretation:** If IQR and Z-score find 0 outliers but IsolationForest finds 25 (as in the BMS1 dataset), it means the flagged batches are not extreme on any single variable but their process condition combinations are unusual. These warrant manual inspection before training — they could be genuine edge-case batches, known process deviations, equipment changes, or scale-up events.

---

### Step 3 — Batch Drift Detection (KS Test)

The Kolmogorov-Smirnov test compares each batch's feature distributions against the overall dataset distribution. A statistically significant difference (p < 0.05) means that batch looks fundamentally different from the rest of the data.

**Why drift matters:** In multi-year, multi-site, or multi-scale manufacturing data, batch drift is extremely common. A batch produced on a 2,000 L bioreactor at Site B looks different from a 200 L batch at Site A. If drifted batches are included in training without flagging, the model learns from a mixed population and its predictions on new batches from a consistent process will be unreliable.

The KS test runs per feature per batch, and a batch is flagged if any feature shows significant drift. Results are shown in the Drift tab with p-values and feature-level breakdown.

| Output | Description |
|--------|-------------|
| Drifted batch count | How many batches differ significantly from the overall dataset |
| Per-feature p-values | Which variables are driving the drift |
| Score impact | Drifted batches deduct from the quality score |

---

### Quality Score

The overall quality score (0–100) combines all three checks:

```
Score = 100
      − (missing_pct × 30)
      − (outlier_pct × 40)
      − (drift_pct   × 30)
```

| Score | Rating | Meaning |
|-------|--------|---------|
| 90–100 | Excellent | Clean dataset, safe to train |
| 75–89 | Good | Minor issues, review flagged rows |
| 60–74 | Fair | Significant issues, investigate before training |
| < 60 | Poor | Dataset needs cleaning before use |

A score of 92.5/100 (as in the BMS1 dataset with 25 IsolationForest outliers) means the data is excellent for training, but those 25 batches should be reviewed — they may represent process edge cases worth understanding scientifically even if they are ultimately included in the model.

---

### Benefit for Scientists

| Check | Scientific value |
|-------|-----------------|
| Missing MCAR/MNAR | Distinguish sensor failures from systematic measurement gaps — informs whether missing data is safe to ignore or signals a process monitoring issue |
| IQR + Z-score | Catch instrument calibration errors, data entry mistakes, and genuine process excursions on individual variables |
| IsolationForest | Find batches with unusual process fingerprints — these are often the most scientifically interesting, representing novel conditions not previously studied |
| KS drift | Detect site-to-site, scale-up, or year-on-year process shifts before they silently bias the predictive model |
| Quality score | Single number for quick assessment, with detailed breakdown for root cause investigation |

---

## The BMS1 vs BMS2 Experiment

### Experimental Design

Two synthetic datasets were generated with identical features, identical noise structure, and identical ground truth relationships. Only three values changed — number of samples, number of batches, and number of test batches. Everything else (feature ranges, target distribution, correlation structure, random seed) was held constant. This isolates **dataset size as the single controlled experimental variable**, enabling a clean comparison of how each model architecture responds to data volume — something rarely possible with real manufacturing data.

### Dataset Characteristics

| Property | BMS1 | BMS2 |
|----------|------|------|
| Total samples | 500 | 10,000 |
| Training samples | 400 | 8,000 |
| Test samples | 100 | 2,000 |
| Batches | 10 | 50 |
| Target mean (Fucosylation_pct) | 87.5% | 88.0% |
| Target std | 5.59% | 5.73% |
| Target range | 67.3% – 100% | 65.1% – 100% |
| Features | 10 | 10 |
| Train/test split | 80/20 | 80/20 |
| Scaler | StandardScaler (fit on train only) | StandardScaler (fit on train only) |

**Features (identical in both):** GDP-Fucose (µM), Manganese (µM), Uridine (mM), pH, Dissolved Oxygen (%), pCO₂ (mmHg), Temperature (°C), Viable Cell Density (10⁶/mL), Osmolality (mOsm), Lactate (g/L)

---

### Full Results — BMS1 (N=500)

| Model | R² Train | R² Test | RMSE | MAE | Notes |
|-------|----------|---------|------|-----|-------|
| Ridge | 0.63 | 0.51 | 3.44 | 2.85 | Linear ceiling — cannot capture nonlinear GDP-Fucose interactions |
| PLSR | 0.63 | 0.51 | 3.45 | 2.85 | Same ceiling as Ridge — VIP identifies GDP-Fucose (1.92), Temp (1.76) as top drivers |
| Random Forest | 0.96 | 0.66 | 3.04 | 2.28 | Train-test gap (0.30) indicates mild overfitting at N=400 train |
| XGBoost+SHAP | 1.00 | 0.74 | 2.82 | 2.16 | Best accuracy at N=500 · SHAP confirms GDP-Fucose × Mn synergy |
| GPR | 0.87 | 0.75 | 2.47 | 1.97 | Best test R² at N=500 · RBF kernel · 95% CI coverage = 95.0% |
| **ANN** | **0.91** | **0.05** | **4.80** | **3.68** | **Complete failure** · 12,225 params vs 340 train samples = 36× overparameterised |
| Hybrid | 0.89 | 0.72 | 2.76 | 2.10 | Physics features (FPI, Energy Charge, Golgi Proxy) add interpretable signal |

**BMS1 rankings:** GPR (0.75) > XGBoost (0.74) > Hybrid (0.72) > RF (0.66) > Ridge/PLSR (0.51) > ANN (0.05)

---

### Full Results — BMS2 (N=10,000)

| Model | R² Train | R² Test | RMSE | MAE | Notes |
|-------|----------|---------|------|-----|-------|
| Ridge | 0.57 | 0.57 | 3.76 | 3.05 | Structural ceiling unchanged — more data cannot fix a misspecified model |
| PLSR | 0.57 | 0.57 | 3.76 | 3.05 | Identical to Ridge — linear latent structure saturated |
| Random Forest | 0.96 | 0.79 | 2.66 | 2.14 | Train-test gap unchanged (0.17) — better generalisation with more data |
| XGBoost+SHAP | 0.93 | 0.83 | 2.47 | 1.96 | Best overall · SHAP interaction terms more reliable with N=8,000 train |
| GPR | 0.79 | 0.79 | 2.61 | 2.09 | Near-perfect train-test alignment · kernel prior dominates at large N |
| **ANN** | **0.85** | **0.77** | **2.74** | **2.20** | **Full recovery** · R² 0.05 → 0.77 (+1,440%) · sufficient data to learn real patterns |
| Hybrid | 0.92 | 0.83 | 2.40 | 1.89 | Best RMSE overall · physics features scale well with data volume |

**BMS2 rankings:** XGBoost/Hybrid (0.83) > GPR/ANN (0.79/0.77) > RF (0.79) > Ridge/PLSR (0.57)

---

### Cross-Experiment Comparison

| Model | BMS1 R² | BMS2 R² | Δ R² | % Change | Interpretation |
|-------|---------|---------|------|----------|----------------|
| Ridge | 0.51 | 0.57 | +0.06 | +12% | Structural ceiling — linear model cannot exploit additional data |
| PLSR | 0.51 | 0.57 | +0.06 | +12% | Same ceiling — latent structure is fully captured at N=500 |
| Random Forest | 0.66 | 0.79 | +0.13 | +20% | Meaningful gain — more splits available, better generalisation |
| XGBoost+SHAP | 0.74 | 0.83 | +0.09 | +12% | Consistent gain — gradient boosting benefits from denser signal |
| GPR | 0.75 | 0.79 | +0.04 | +5% | Small gain — kernel prior already captures structure at N=500 |
| **ANN** | **0.05** | **0.77** | **+0.72** | **+1,440%** | **Catastrophic failure → full recovery** |
| Hybrid | 0.72 | 0.83 | +0.11 | +15% | Physics features scale — domain knowledge compounds with data |

---

### The ANN Finding — Deep Dive

At **N=500 (BMS1):**
- Training samples after 80/20 split: 340
- ANN architecture: 128 → 64 → 32 → 1
- Learnable parameters: 12,225
- Parameters-to-samples ratio: **36× overparameterised**
- Result: R²=0.91 on training (memorises noise), R²=0.05 on test (random predictions)
- Equivalent to solving 12,000 simultaneous equations with only 340 data points

At **N=10,000 (BMS2):**
- Training samples: 8,000
- Same architecture: 128 → 64 → 32 → 1
- Same 12,225 parameters
- Parameters-to-samples ratio: **0.65× (sufficient data)**
- Result: R²=0.85 train, R²=0.77 test — full recovery
- The model now has enough data points to constrain its parameters to real patterns

**Threshold:** The ANN begins recovering meaningfully above approximately N=5,000 training samples. Below this threshold, GPR or tree-based models are consistently more reliable for pharmaceutical bioprocess datasets.

---

### Key Scientific Conclusions

**1. Data volume is model-dependent.** Linear models (Ridge, PLSR) hit a structural ceiling regardless of how much data you add — the problem is model misspecification, not data quantity. Tree-based and deep models continue improving.

**2. Neural networks are dangerous in small pharma datasets.** The 36× overparameterisation at N=500 is not an edge case — it is the typical situation in early-phase biopharmaceutical development where 10–50 batches is standard. Using an ANN on such data without checking the parameter/sample ratio will reliably produce a model that appears to train well but fails completely on new batches.

**3. GPR is the best choice for small N.** It achieved the highest R² at N=500 (0.75) while providing calibrated confidence intervals — making it suitable for risk-based batch release decisions under ICH Q8/Q9.

**4. XGBoost+SHAP is the best production model at scale.** Tied for best R² at N=10,000 (0.83) with full SHAP explainability for regulatory submissions.

**5. The Hybrid model scales best.** Physics-informed feature engineering (FPI, Energy Charge, Golgi Proxy, Stress Index, Enzyme Activity) compounds with data volume, achieving the best RMSE at N=10,000 (2.40%).

---

## The 7 ML Models

| Model | Strength |
|-------|----------|
| Ridge Regression | Linear baseline, interpretable coefficients, RidgeCV auto-tunes α |
| PLSR | VIP scores (>1.0 = significant), chemometrics gold standard, ICH Q8 |
| Random Forest | 200 trees, OOB score, MDI importance, robust nonlinear capture |
| XGBoost + SHAP | Best accuracy + SHAP explainability for regulatory submissions |
| GPR | Calibrated confidence intervals, best for small N, 95% CI coverage |
| ANN | 128→64→32, early stopping, overparameterisation check, needs N>5k |
| Hybrid | Auto-engineers FPI, Energy Charge, Golgi Proxy, Stress Index, Enzyme Activity |

---

## Gemini API

### Free Tier
| Limit | Value |
|-------|-------|
| Requests/day | 1,500 |
| Requests/min | 10 |
| Output tokens | 65,536 per call |
| Price | Free |

### Paid Tier
$0.15 per 1M input tokens · $0.60 per 1M output tokens. One report ≈ $0.001.

---

## Authentication — Passwordless Login

This dashboard uses two passwordless sign-in methods. No passwords are stored anywhere — not in the backend, not in a database, not in a file.

### Google OAuth 2.0 — One-Click SaaS Login

Google OAuth is the same login system used by Slack, Notion, Figma, and most modern SaaS tools. Instead of creating a new account with a username and password, you delegate authentication entirely to Google.

**How it works:**
1. You click **Continue with Google**
2. The backend redirects your browser to Google's consent screen
3. You pick your Google account and approve
4. Google sends a secure one-time code back to the backend (`/auth/google/callback`)
5. The backend exchanges that code for your profile (name, email, picture) via Google's API
6. A signed session token is generated using a secret key and stored in your browser's `localStorage`
7. Every subsequent request includes this token as a `Bearer` header — no cookies, no cross-port issues

**Why this is secure:**
- Your Google password is never seen or touched by this app
- Tokens expire after 7 days
- Revoking access in your Google account immediately invalidates the session

**Setup requires:** A Google Cloud project with OAuth 2.0 credentials and the redirect URI `http://localhost:8000/auth/google/callback` whitelisted. See Setup → Step 2.

---

### Email OTP — Gmail SMTP Passwordless Login

For users who prefer not to use Google OAuth, the dashboard supports magic-code login via email. A 6-digit one-time password (OTP) is generated server-side and delivered to your inbox through Gmail's SMTP server. We can use **Amazon Simple Email Service (SES)** if we plan to deploy on AWS.

**How it works:**
1. You enter your email address and click **Send login code**
2. The backend generates a cryptographically random 6-digit code
3. The code is stored in memory with a 10-minute expiry timestamp
4. The backend connects to Gmail's SMTP server (`smtp.gmail.com:465`) using your App Password and sends a styled HTML email
5. You enter the code in the dashboard
6. The backend validates the code (correct + not expired), deletes it from memory (single-use), builds your user profile from the email address, and returns a signed session token
7. The token is stored in `localStorage` — same as Google OAuth from this point forward

**Why OTP codes are safe:**
- Codes are 6 digits = 1,000,000 possibilities, valid for only 10 minutes
- Each code is deleted immediately after one successful use — replay attacks are impossible
- The Gmail App Password is a 16-character credential specific to this app — it cannot be used to access your Gmail inbox and can be revoked at any time from your Google Account security settings
- No email addresses or codes are written to disk

**Setup requires:** A Gmail account with 2-Step Verification enabled and an App Password generated for this app. See Setup → Step 3.

---

### Why Passwordless?

Traditional username + password login requires storing hashed passwords, handling forgotten passwords, and protecting against credential stuffing attacks. Passwordless login delegates all of that to providers (Google) or to time-limited codes — giving users a faster experience and giving developers one less security surface to maintain. Both methods used here are production-grade patterns used across the pharmaceutical SaaS industry.

---

## Future Extensions

1. **Real-Time Monitoring** — connect to LIMS/MES via REST API, predict per-measurement, alert on drift
2. **Multi-CQA** — change the target column to predict galactosylation, sialylation, or yield
3. **In Silico DOE** — generate synthetic process conditions, find optimal parameter combinations
4. **Transfer Learning** — fine-tune on a new mAb product from a small dataset
5. **Uncertainty-Guided Release** — GPR confidence intervals drive risk-based batch release decisions
6. **Continuous Drift Detection** — KS-test as a production monitoring service

---

## Deployment

### Docker

```dockerfile
# backend/Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```dockerfile
# frontend/Dockerfile
FROM node:18-alpine AS build
WORKDIR /app
COPY package.json .
RUN npm install
COPY . .
RUN npm run build
FROM nginx:alpine
COPY --from=build /app/build /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

```yaml
# docker-compose.yml
version: "3.9"
services:
  backend:
    build: ./backend
    ports: ["8000:8000"]
    env_file: ./backend/.env
  frontend:
    build: ./frontend
    ports: ["3000:80"]
    depends_on: [backend]
```

```bash
docker-compose up --build
```

### AWS App Runner
1. Push to GitHub → App Runner → Create Service → Source: GitHub
2. Point to `backend/`, port 8000, add env vars
3. Frontend: S3 static site + CloudFront HTTPS

### Azure Container Apps
```bash
az group create --name bms-hackathon --location eastus
az containerapp env create --name bms-env --resource-group bms-hackathon --location eastus
az containerapp create --name bms-backend --resource-group bms-hackathon \
  --environment bms-env --image <registry>/bms-backend:latest \
  --target-port 8000 --ingress external
az staticwebapp create --name bms-frontend --resource-group bms-hackathon \
  --source https://github.com/<repo> --branch main \
  --app-location /frontend --output-location build
```

---

*Built with ♥ by Team Data Minds — Sanjith Ganesh & Pranav Senthilkumaran*
