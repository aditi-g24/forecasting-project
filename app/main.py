"""
main.py
-------
FastAPI application exposing the forecasting system as a REST API.

Endpoints:
  GET  /            - Welcome message
  GET  /health      - Health check
  POST /train       - Trigger training pipeline
  GET  /forecast    - Get 8-week forecast for a state
  GET  /metrics     - Get model evaluation metrics for a state
  GET  /states      - List all available trained states
  GET  /models      - List all models available for a state
"""

import json
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ──────────────────────────────────────────────────────────────────────────────
# APP SETUP
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Sales Forecasting API",
    description="Multi-model time-series forecasting for US state-level beverage sales.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths (can be overridden via environment variables)
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
DATA_PATH     = os.getenv("DATA_PATH", "data/sales_data.xlsx")

# Training status tracker
_training_status = {"status": "idle", "message": "No training has been triggered yet."}


# ──────────────────────────────────────────────────────────────────────────────
# HELPER
# ──────────────────────────────────────────────────────────────────────────────

def _load_summary(state: str) -> dict:
    """Load the summary.json for a state, raise 404 if not found."""
    state_key = state.replace(" ", "_")
    summary_path = ARTIFACTS_DIR / state_key / "summary.json"
    if not summary_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No trained models found for state '{state}'. "
                   f"Run POST /train first."
        )
    with open(summary_path) as f:
        return json.load(f)


def _load_all_results() -> dict:
    """Load master all_results.json."""
    path = ARTIFACTS_DIR / "all_results.json"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="No training results found. Run POST /train first."
        )
    with open(path) as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# REQUEST SCHEMAS
# ──────────────────────────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    states:     Optional[list[str]] = None  # None → train all states
    train_lstm: bool = True                 # Set False for faster CPU-only training


# ──────────────────────────────────────────────────────────────────────────────
# BACKGROUND TRAINING TASK
# ──────────────────────────────────────────────────────────────────────────────

def _run_training(states, train_lstm):
    global _training_status
    _training_status = {"status": "running", "message": "Training in progress…"}
    try:
        from app.trainer import run_pipeline
        run_pipeline(
            data_path=DATA_PATH,
            artifacts_dir=str(ARTIFACTS_DIR),
            states=states,
            train_lstm=train_lstm,
        )
        _training_status = {
            "status":  "completed",
            "message": "Training finished successfully.",
        }
    except Exception as e:
        _training_status = {"status": "failed", "message": str(e)}


# ──────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["General"])
def root():
    return {
        "message": "Sales Forecasting API is running 🚀",
        "docs":    "/docs",
        "redoc":   "/redoc",
    }


@app.get("/health", tags=["General"])
def health():
    artifacts_exist = ARTIFACTS_DIR.exists()
    results_exist   = (ARTIFACTS_DIR / "all_results.json").exists()
    return {
        "status":          "healthy",
        "artifacts_dir":   str(ARTIFACTS_DIR),
        "artifacts_exist": artifacts_exist,
        "models_trained":  results_exist,
        "training_status": _training_status["status"],
    }


@app.post("/train", tags=["Training"])
def train(request: TrainRequest, background_tasks: BackgroundTasks):
    """
    Trigger model training in the background.
    - `states`: list of state names to train (omit or pass `null` for ALL states)
    - `train_lstm`: set to `false` to skip LSTM (much faster, no GPU required)
    """
    global _training_status
    if _training_status["status"] == "running":
        return JSONResponse(
            status_code=409,
            content={"detail": "Training is already running. Check GET /health for status."}
        )
    background_tasks.add_task(_run_training, request.states, request.train_lstm)
    _training_status = {"status": "starting", "message": "Training queued."}
    return {
        "message":    "Training started in the background.",
        "states":     request.states or "all",
        "train_lstm": request.train_lstm,
        "tip":        "Poll GET /health to check training status.",
    }


@app.get("/states", tags=["Data"])
def list_states():
    """List all states that have been trained."""
    results = _load_all_results()
    return {"states": sorted(results.keys()), "count": len(results)}


@app.get("/models", tags=["Data"])
def list_models(state: str):
    """List models available for a given state and their metrics."""
    summary = _load_summary(state)
    return {
        "state":      state,
        "best_model": summary["best_model"],
        "models":     summary["metrics"],
    }


@app.get("/forecast", tags=["Forecasting"])
def get_forecast(state: str, model: Optional[str] = None, weeks: int = 8):
    """
    Return the 8-week forecast for a state.

    - `state`  : US state name (e.g. `California`)
    - `model`  : optional — specify `sarima`, `prophet`, `xgboost`, or `lstm`;
                 defaults to the best model
    - `weeks`  : number of forecast weeks to return (max 8)
    """
    if weeks < 1 or weeks > 8:
        raise HTTPException(status_code=400, detail="`weeks` must be between 1 and 8.")

    summary = _load_summary(state)
    chosen  = model or summary["best_model"]

    if model and model not in summary["metrics"]:
        available = list(summary["metrics"].keys())
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model}' not found for {state}. Available: {available}"
        )

    # If the user requested a non-best model we need to reload it and re-forecast
    if model and model != summary["best_model"]:
        forecast = _forecast_with_model(state, model, weeks)
    else:
        forecast = summary["forecast"][:weeks]

    return {
        "state":      state,
        "model_used": chosen,
        "weeks":      weeks,
        "forecast":   forecast,
    }


def _forecast_with_model(state: str, model_name: str, weeks: int) -> list:
    """Load a saved model and generate a fresh forecast."""
    from app.preprocessing import load_data, clean_and_resample
    from app import models as M

    state_key  = state.replace(" ", "_")
    model_path = ARTIFACTS_DIR / state_key / f"{model_name}.pkl"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Saved model file not found: {model_path}")

    model_cls_map = {
        "sarima":  M.SARIMAModel,
        "prophet": M.ProphetModel,
        "xgboost": M.XGBoostModel,
        "lstm":    M.LSTMModel,
    }
    m = model_cls_map[model_name].load(str(model_path))

    df       = load_data(DATA_PATH)
    weekly   = clean_and_resample(df, state)
    last_date = weekly["ds"].max()

    from app.preprocessing import make_future_dates
    future_dates = make_future_dates(last_date, weeks)
    preds = m.predict(weeks)

    return [
        {"date": str(d.date()), "sales": round(v, 2)}
        for d, v in zip(future_dates["ds"], preds)
    ]


@app.get("/metrics", tags=["Evaluation"])
def get_metrics(state: Optional[str] = None):
    """
    Return model evaluation metrics.
    - If `state` is provided, returns metrics for that state only.
    - If omitted, returns a summary across all states.
    """
    all_results = _load_all_results()

    if state:
        if state not in all_results:
            raise HTTPException(status_code=404, detail=f"State '{state}' not found.")
        s = all_results[state]
        return {
            "state":      state,
            "best_model": s["best_model"],
            "metrics":    s["metrics"],
        }

    # Aggregate summary across all states
    agg = {}
    for model_name in ["sarima", "prophet", "xgboost", "lstm"]:
        rmses, maes, mapes, count = [], [], [], 0
        for s_data in all_results.values():
            if model_name in s_data.get("metrics", {}):
                m = s_data["metrics"][model_name]
                rmses.append(m["rmse"])
                maes.append(m["mae"])
                mapes.append(m["mape"])
                count += 1
        if rmses:
            agg[model_name] = {
                "states_trained": count,
                "avg_rmse":  round(sum(rmses) / len(rmses), 2),
                "avg_mae":   round(sum(maes)  / len(maes),  2),
                "avg_mape":  round(sum(mapes) / len(mapes), 4),
            }

    best_counts = {}
    for s_data in all_results.values():
        bm = s_data.get("best_model", "unknown")
        best_counts[bm] = best_counts.get(bm, 0) + 1

    return {
        "total_states":    len(all_results),
        "model_summary":   agg,
        "best_model_wins": best_counts,
    }
