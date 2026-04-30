# mAb Fucosylation · ML Pipeline Dashboard

> **BMS Hackathon** · MS Data Science · Rutgers University × Bristol Myers Squibb  
> Built with ♥ by **Team Data Minds** — Sanjith Ganesh & Pranav Senthilkumaran

🎬 **[Watch Demo Video](https://drive.google.com/file/d/1C3SjgpWfuQdF2zICyJK6sGTAsf1KZcos/view?usp=sharing)**

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
│   └── .env                 ← All credentials (never commit)
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
├── BMS Midterm Presentation.pdf
├── Results and comparison (BMS1 VS BMS2).docx
└── README.md
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

## The BMS1 vs BMS2 Experiment

Two datasets — identical features, noise, and ground truth. Only dataset size changed (N=500 vs N=10,000). This isolates **data volume as the single controlled variable**.

| | BMS1 | BMS2 |
|-|------|------|
| Samples | 500 | 10,000 |
| Batches | 10 | 50 |
| Target mean | 87.5% | 88.0% |
| Target std | 5.59% | 5.73% |

| Model | BMS1 R² | BMS2 R² | Change |
|-------|---------|---------|--------|
| Ridge | 0.51 | 0.57 | +12% |
| PLSR | 0.51 | 0.57 | +12% |
| Random Forest | 0.66 | 0.79 | +20% |
| XGBoost+SHAP | 0.74 | 0.83 | +12% |
| GPR | 0.75 | 0.79 | +5% |
| **ANN** | **0.05** | **0.77** | **+1440%** |
| Hybrid | 0.72 | 0.83 | +15% |

The ANN had 12,225 parameters vs 340 training samples at N=500 — 36× overparameterised. At N=10,000 it fully recovers (+1440%).

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
