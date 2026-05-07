#  Sales Forecasting System

A production-quality, end-to-end time-series forecasting system for US state-level beverage sales.

## What It Does

- Trains **4 forecasting models** per state: SARIMA, Prophet, XGBoost, LSTM
- **Automatically selects the best model** based on RMSE
- **Forecasts the next 8 weeks** of sales for each of 43 US states
- Exposes everything via a clean **FastAPI REST API**
- Trainable in **Google Colab** (T4 GPU), deployable to **Render**

---

## Project Structure

```
forecasting_project/
├── app/
│   ├── __init__.py
│   ├── main.py           ← FastAPI application (all endpoints)
│   ├── models.py         ← SARIMA, Prophet, XGBoost, LSTM classes
│   ├── preprocessing.py  ← Data loading, cleaning, feature engineering
│   └── trainer.py        ← Full training pipeline (CLI entry point)
├── notebooks/
│   └── training_notebook.ipynb   ← Google Colab training notebook
├── data/
│   └── sales_data.xlsx   ← Your dataset (place it here)
├── artifacts/            ← Created automatically during training
│   ├── all_results.json
│   └── <State>/
│       ├── summary.json
│       ├── sarima.pkl
│       ├── prophet.pkl
│       ├── xgboost.pkl
│       ├── lstm.pkl
│       └── forecast_plot.png
├── requirements.txt
├── Procfile              ← Render deployment
├── render.yaml           ← Render deployment config
└── COMPLETE_SETUP_GUIDE.md
```

---

## Quick Start

### Option A — Train in Google Colab (Recommended)

1. Open `notebooks/training_notebook.ipynb` in Google Colab
2. Set runtime to **T4 GPU** (`Runtime → Change runtime type`)
3. Run all cells
4. Download `artifacts_export.zip` from the last cell
5. Extract into this project's `artifacts/` folder

### Option B — Train Locally (CPU only, slower)

```bash
# Install dependencies
pip install -r requirements.txt

# Train (skip LSTM for speed, or keep --no-lstm off to include it)
python -m app.trainer --data data/sales_data.xlsx --artifacts artifacts --no-lstm

# With LSTM (needs more time/RAM)
python -m app.trainer --data data/sales_data.xlsx --artifacts artifacts
```

---

## Running the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open your browser: **http://localhost:8000/docs**

---

## API Endpoints

| Method | Endpoint     | Description                              |
|--------|-------------|------------------------------------------|
| GET    | `/`          | Welcome message                          |
| GET    | `/health`    | Health check + training status           |
| POST   | `/train`     | Trigger background training              |
| GET    | `/states`    | List all trained states                  |
| GET    | `/models`    | Model metrics for a specific state       |
| GET    | `/forecast`  | 8-week forecast for a state              |
| GET    | `/metrics`   | Evaluation metrics (one state or all)    |

### Example Requests

```bash
# Health check
curl http://localhost:8000/health

# Forecast for California (best model)
curl "http://localhost:8000/forecast?state=California"

# Forecast using a specific model
curl "http://localhost:8000/forecast?state=Texas&model=prophet"

# Metrics for all states
curl http://localhost:8000/metrics

# Metrics for one state
curl "http://localhost:8000/metrics?state=New+York"

# Trigger training (all states, with LSTM)
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"train_lstm": true}'

# Trigger training (specific states, no LSTM — fast)
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"states": ["California", "Texas"], "train_lstm": false}'
```

---

## Models

| Model   | Library       | Strengths                              |
|---------|--------------|----------------------------------------|
| SARIMA  | statsmodels  | Handles trends & seasonality directly  |
| Prophet | Meta/Prophet | Robust to outliers, easy seasonality   |
| XGBoost | xgboost      | Fast, captures non-linear patterns     |
| LSTM    | TensorFlow   | Deep learning, captures complex trends |

### Evaluation Metrics

- **RMSE** — Root Mean Squared Error (primary selection criterion)
- **MAE** — Mean Absolute Error
- **MAPE** — Mean Absolute Percentage Error

The model with the lowest RMSE on the 8-week hold-out set is automatically selected as best.

---

## Deployment (Render)

1. Push this project to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your GitHub repo
4. Render auto-detects `render.yaml`
5. Deploy!

See `COMPLETE_SETUP_GUIDE.md` for step-by-step instructions.

---

## Dataset Format

| Column   | Type   | Description                    |
|----------|--------|-------------------------------|
| State    | string | US state name                  |
| Date     | date   | Transaction / period date      |
| Total    | float  | Sales amount ($)               |
| Category | string | Product category (Beverages)   |
