# mAb Fucosylation · ML Pipeline Dashboard

> **BMS Hackathon** · MS Data Science · Rutgers University × Bristol Myers Squibb  
> Built with ♥ by **Team Data Minds** — Sanjith Ganesh & Pranav Senthilkumaran

🎬 **[Watch Demo Video](https://drive.google.com/file/d/1C3SjgpWfuQdF2zICyJK6sGTAsf1KZcos/view?usp=sharing)**

---

A full-stack machine learning dashboard that predicts monoclonal antibody (mAb) fucosylation — a Critical Quality Attribute in biopharmaceutical manufacturing — from CHO cell culture bioprocess variables.

Upload your CSV → automated data cleansing → train 7 ML models → PowerBI-style results → AI-generated regulatory report.

---

## File Structure

```
bms-pipeline/
│
├── backend/
│   ├── main.py              ← FastAPI — all logic (cleansing, training, Gemini)
│   ├── requirements.txt     ← Python dependencies
│   └── .env                 ← Your Gemini API key 
│
├── frontend/
│   ├── package.json
│   ├── public/
│   │   ├── index.html
│   │   └── bms-logo.png     ← BMS Logo
│   └── src/
│       ├── App.jsx          ← Entire frontend — all pages and components
│       ├── index.js
│       └── index.css
│
└── README.md
```


---

## Setup

### Requirements
- Python 3.9+ with conda or pip
- Node.js 18+ and npm
- A Gemini API key (free at [aistudio.google.com](https://aistudio.google.com))

### Step 1 — Add your Gemini API key

Edit `backend/.env`:
```
GEMINI_API_KEY=AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

### Step 2 — Add the BMS logo

Copy the Bristol Myers Squibb logo PNG to `frontend/public/bms-logo.png`. If the file is missing the header just shows blank — everything else still works.

### Step 3 — Start the backend

```bash
cd backend
conda activate msds597          # or your Python environment
pip install -r requirements.txt # first time only
uvicorn main:app --reload --port 8000
```

Verify at [http://localhost:8000](http://localhost:8000) — you should see a JSON status response.

### Step 4 — Start the frontend

Open a **second terminal**:
```bash
cd frontend
npm install   # first time only
npm start
```

Opens at [http://localhost:3000](http://localhost:3000).

---

## How to Use

### Step 1 — Upload & Cleanse
Drop your CSV. Three checks run automatically:
- **Missing values** — blank cell detection, MCAR vs MNAR classification
- **Outlier detection** — IQR fence + Z-score + IsolationForest (all three at once)
- **Batch drift** — KS test comparing each batch to the full dataset

Quality score out of 100. Five detail tabs. No data is modified.

### Step 2 — Select & Train
Your file carries over — no re-upload needed. Tick the models you want, click Train. The backend splits 80/20, scales features (training data only — no leakage), trains each model on your real data, and returns R², RMSE, and MAE.

### Step 3 — Results Dashboard
Every number comes from your actual uploaded CSV:
- KPI cards, model visibility toggles (show/hide per model across all charts)
- R² bar chart, actual vs predicted scatter, SHAP/VIP/MDI feature importance
- Multi-criteria radar (accuracy, interpretability, uncertainty, speed, regulatory fitness)
- Learning curves

### Step 4 — AI Report
Three report types — all built from your real trained results:

| Type | Length | Written for |
|------|--------|-------------|
| **Scientific Report** | 800–1000 words | Bioprocess scientists, academic reviewers, regulatory submissions. Thesis-style prose with literature citations and ICH Q8/Q9 context. |
| **Executive Report** | 400–500 words | VPs, CSOs, C-suite. Business impact in plain English — no formulas, no acronyms. Decision-ready. |
| **Quick Summary** | 200–250 words | Team meetings, slide notes. Headline numbers, top features, one recommendation. Direct and punchy. |

Export any report as a formatted A4 PDF.

### BMS1 vs BMS2 (Static Reference Tab)
Always accessible from the top nav. Shows the static reference experiment — 7 models across two dataset sizes. See section below.

---

## The BMS1 vs BMS2 Experiment

This is the core scientific contribution of the hackathon project.

**Experimental design:** Two datasets were created with identical features, noise, and ground truth. Only three values changed — number of samples, number of batches, number of test batches. This isolates **dataset size as the single controlled variable**.

| | BMS1 | BMS2 |
|-|------|------|
| Samples | 500 | 10,000 |
| Batches | 10 | 50 |
| Target mean | 87.5% | 88.0% |
| Target std | 5.59% | 5.73% |
| Train / Test | 400 / 100 | 8,000 / 2,000 |

**Features (both datasets):** GDP-Fucose, Manganese, Uridine, pH, Dissolved Oxygen, pCO₂, Temperature, Viable Cell Density, Osmolality, Lactate.

### Results Summary

| Model | BMS1 R² | BMS2 R² | Change | What it shows |
|-------|---------|---------|--------|---------------|
| Ridge | 0.51 | 0.57 | +12% | Linear structural ceiling — more data doesn't help |
| PLSR | 0.51 | 0.57 | +12% | Same ceiling — linear relationship assumption limits performance |
| Random Forest | 0.66 | 0.79 | +20% | Nonlinear capture improves meaningfully with more data |
| XGBoost+SHAP | 0.74 | 0.83 | +12% | Best production model at scale |
| GPR | 0.75 | 0.79 | +5% | Strong kernel prior already works at N=500 |
| **ANN** | **0.05** | **0.77** | **+1440%** | **Complete failure → full recovery** |
| Hybrid | 0.72 | 0.83 | +15% | Physics features scale with data |

### The ANN Headline Finding

At N=500: the ANN had 12,225 learnable parameters but only 340 training samples — a 36× overparameterisation ratio. Like solving 12,000 equations with 340 data points — the model memorises noise and achieves R²=0.05 (essentially random).

At N=10,000: 8,000 training samples vs 12,225 parameters — the model now has sufficient data to learn real patterns, recovering to R²=0.77 (+1440%).

This demonstrates a fundamental principle of deep learning in pharmaceutical settings: **small datasets and neural networks are incompatible**. For datasets under ~5,000 samples, GPR or tree-based models are more reliable.

---

## The 7 Models

| Model | What it does | Best for |
|-------|-------------|----------|
| Ridge Regression | Linear model with regularisation. Picks the best penalty strength automatically via cross-validation. | Interpretable baseline |
| PLSR | Projects data into latent components. Computes VIP scores — any feature with VIP > 1.0 significantly drives fucosylation. | Academic, ICH Q8 regulatory submissions |
| Random Forest | 200 decision trees vote together. Returns out-of-bag score and MDI feature importance. | Robust nonlinear capture |
| XGBoost + SHAP | Gradient boosting with SHAP explainability — measures each variable's contribution to each individual prediction. | Best accuracy + production deployment |
| GPR | Probabilistic model returning a confidence interval with every prediction. 95% CI coverage measured on test set. | Uncertainty quantification for batch release |
| ANN | Neural network (128→64→32). Includes overparameterisation check — warns if param/sample > 1. | Large datasets only (N > 5,000) |
| Hybrid | Auto-detects column names, engineers 5 physics features (FPI, Energy Charge, Golgi Proxy, Stress Index, Enzyme Activity), then trains gradient boosting. | Domain knowledge fusion |

---

## Gemini API — Free Tier and Paid Usage

### Free Tier (Google AI Studio)
| Limit | Value |
|-------|-------|
| Requests per minute | 10 |
| Requests per day | 1,500 |
| Tokens per minute | 1,000,000 |
| Context window | 1,048,576 tokens |
| Output tokens per request | 65,536 |
| Price | Free |

For this dashboard — one report generation = one API call. At 1,500 requests/day on the free tier, that is well within any hackathon or research use case.

If you hit the per-minute rate limit (10 RPM) you will see a `429` error. Wait 60 seconds and try again.

### Paid Tier (Google AI Vertex or API)
Gemini 2.5 Flash is billed per token on the paid tier:

| | Input tokens | Output tokens |
|-|-------------|---------------|
| Price | $0.15 per 1M tokens | $0.60 per 1M tokens |
| Minimum billing | Per 1K tokens | Per 1K tokens |

A typical report generation call sends roughly 800–1,000 input tokens and returns 1,000–1,200 output tokens — costing approximately **$0.001 per report** (less than a tenth of a cent). Running 1,000 reports would cost roughly $1.00.

To upgrade from free to paid: create a Google Cloud project, enable the Generative Language API, add a billing account, and switch your API key from AI Studio to Cloud Console. No code changes needed — the endpoint URL stays the same.

---

## Dataset Format

Your CSV must have:

| Column | Required | Description |
|--------|----------|-------------|
| `Fucosylation_pct` | Preferred | The target to predict |
| `Batch_ID` | Optional | Enables batch drift analysis |
| All other columns | Yes | Treated as input features |

If `Fucosylation_pct` is not found, the last column is used as the target automatically. The pipeline works with any numerical bioprocess CSV — not just the BMS dataset.

---

## Future Extensions — Benefits for Scientists

This pipeline was designed to generalise. Here are the most meaningful extensions for biopharmaceutical scientists:

### 1. Real-Time Process Monitoring
Connect the trained model to a LIMS (Laboratory Information Management System) or MES (Manufacturing Execution System) via the REST API. As each process measurement comes in during a batch run, the model predicts the end-of-batch fucosylation level. If the predicted value drifts outside the acceptable range, an alert fires — giving operators time to adjust pH, temperature, or feeding strategy before the batch is lost.

### 2. Multi-CQA Extension
The pipeline currently predicts fucosylation. The same architecture handles any Critical Quality Attribute — galactosylation, sialylation, high molecular weight species, or product yield — with no code changes. Simply change the target column in your CSV.

### 3. Process Optimisation (In Silico DOE)
Once a model is trained, you can generate thousands of synthetic process conditions and ask the model which combinations maximise fucosylation (or minimise it, depending on your target). This is a virtual Design of Experiments — screening a much larger space than is feasible in the lab, at zero material cost.

### 4. Transfer Learning Across Products
A model trained on one mAb product can be fine-tuned on a small dataset from a new product using transfer learning. This is particularly valuable in early clinical manufacturing where N is small — the new model starts from a learned prior rather than from scratch.

### 5. Uncertainty-Guided Batch Release
The GPR model provides a confidence interval with every prediction. When the interval is narrow, the prediction is reliable and the batch can be provisionally released. When the interval is wide, the model is uncertain — flag the batch for additional analytical testing before release. This creates a risk-based, model-informed release strategy aligned with ICH Q8/Q9.

### 6. Automated Drift Detection in Production
The cleansing module already runs KS-test batch drift detection. In a production setting, this becomes a continuous monitoring service — automatically comparing incoming process data to the training distribution and alerting when batch conditions drift outside the model's reliable operating range.

---

## Deployment — Docker + Cloud

### Dockerise the Application

**Step 1 — Backend Dockerfile** (`backend/Dockerfile`):
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Step 2 — Frontend Dockerfile** (`frontend/Dockerfile`):
```dockerfile
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

**Step 3 — docker-compose.yml** (project root):
```yaml
version: "3.9"
services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    env_file:
      - ./backend/.env

  frontend:
    build: ./frontend
    ports:
      - "3000:80"
    depends_on:
      - backend
```

**Build and run:**
```bash
docker-compose up --build
```

---

### Deploy on AWS

**Option A — AWS App Runner (simplest, no infrastructure knowledge needed)**
1. Push your repo to GitHub or ECR
2. Go to AWS App Runner → Create Service → Source: GitHub
3. Point to `backend/` directory, set port 8000
4. Add `GEMINI_API_KEY` as an environment variable in the console
5. Deploy. App Runner handles scaling automatically.
6. For the frontend: upload the built React app to an S3 bucket, enable static website hosting, put CloudFront in front for HTTPS.

**Option B — AWS ECS + Fargate (production-grade)**
```bash
# Push images to ECR
aws ecr create-repository --repository-name bms-backend
aws ecr create-repository --repository-name bms-frontend
docker build -t bms-backend ./backend
docker push <account>.dkr.ecr.<region>.amazonaws.com/bms-backend

# Create ECS cluster, task definition, and service via console or Terraform
```

**Estimated AWS cost:** App Runner runs at ~$0.005/vCPU-hour. At 0.25 vCPU, that is roughly **$1/month** for a light-traffic research tool.

---

### Deploy on Azure

**Option A — Azure Container Apps (simplest)**
```bash
# Login and create resource group
az login
az group create --name bms-hackathon --location eastus
az containerapp env create --name bms-env --resource-group bms-hackathon --location eastus

# Deploy backend
az containerapp create \
  --name bms-backend \
  --resource-group bms-hackathon \
  --environment bms-env \
  --image <your-registry>/bms-backend:latest \
  --target-port 8000 \
  --ingress external \
  --env-vars GEMINI_API_KEY=secretref:gemini-key

# Deploy frontend (Static Web Apps)
az staticwebapp create \
  --name bms-frontend \
  --resource-group bms-hackathon \
  --source https://github.com/<your-repo> \
  --location eastus \
  --branch main \
  --app-location /frontend \
  --output-location build
```

**Option B — Azure App Service**
1. Create an App Service Plan (B1 tier, ~$13/month)
2. Deploy backend container via `az webapp create --deployment-container-image-name`
3. Set `GEMINI_API_KEY` under Configuration → Application Settings
4. Deploy frontend to Azure Static Web Apps (free tier available)

---

### Environment Variables for Deployment

| Variable | Where to set | Value |
|----------|-------------|-------|
| `GEMINI_API_KEY` | Backend environment | Your Gemini API key |
| `REACT_APP_API_URL` | Frontend build | Backend URL (e.g. `https://api.yourdomain.com`) |

**Important:** Update the `proxy` field in `frontend/package.json` from `http://localhost:8000` to your deployed backend URL before building for production. Or set `REACT_APP_API_URL` and update the axios calls in `App.jsx` to use `process.env.REACT_APP_API_URL`.

---

## Common Issues

| Problem | Fix |
|---------|-----|
| AI report shows only ~40 words | Old `maxOutputTokens` was 1500 — current `main.py` uses 8192. Make sure you have the latest version. |
| "Cannot parse CSV" | Save as plain CSV not `.xlsx`. In Excel: File → Save As → CSV (Comma delimited). |
| "All models failed" | Fewer than 20 complete rows after dropping NaN. Clean your data first. |
| Port 8000 in use | `lsof -ti:8000 \| xargs kill` |
| Port 3000 in use | `lsof -ti:3000 \| xargs kill` |
| Logo not showing | Put `bms-logo.png` in `frontend/public/` not `frontend/src/`. |
| Gemini 429 error | Free tier rate limit (10 req/min). Wait 60 seconds and try again. |
| GPR very slow | GPR is capped at 300 training samples automatically. For N > 50,000 consider skipping GPR. |

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Backend | FastAPI, scikit-learn, XGBoost, SHAP, SciPy, pandas, numpy, httpx, python-dotenv |
| Frontend | React 18, Chart.js 4, react-chartjs-2, axios, lucide-react |
| AI | Google Gemini 2.5 Flash |
| Fonts | DM Sans + DM Mono (Google Fonts) |
| Deployment | Docker, AWS App Runner / ECS, Azure Container Apps / Static Web Apps |

---

## Project Background

Built for the MS Data Science BMS Hackathon at Rutgers University × Bristol Myers Squibb. The scientific question was: **how does dataset size affect ML model performance in pharmaceutical bioprocessing?**

BMS1 (N=500) and BMS2 (N=10,000) are identical in every way except sample count. Training 7 models on both datasets and comparing results isolates data volume as the sole variable.

Key finding: neural networks are dangerous in small pharmaceutical datasets. The ANN achieved R²=0.05 at N=500 (catastrophic failure) and R²=0.77 at N=10,000 (full recovery). Linear models are reliable but hit a ceiling. XGBoost+SHAP is the best production choice. GPR is the best choice when you need uncertainty estimates for risk-based decision making.

---

*Built with ♥ by Team Data Minds — Sanjith Ganesh & Pranav Senthilkumaran*
