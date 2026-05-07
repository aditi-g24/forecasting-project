"""
trainer.py
----------
Orchestrates the full training pipeline for all models and all states.

Usage (standalone):
    python -m app.trainer --data data/sales_data.xlsx --artifacts artifacts/
"""

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.preprocessing import (
    load_data, clean_and_resample, train_test_split_ts, make_future_dates
)
from app.models import SARIMAModel, ProphetModel, XGBoostModel, LSTMModel


# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

MODEL_CLASSES = {
    "sarima":   SARIMAModel,
    "prophet":  ProphetModel,
    "xgboost":  XGBoostModel,
    "lstm":     LSTMModel,
}

FORECAST_WEEKS = 8
TEST_WEEKS     = 8


# ──────────────────────────────────────────────────────────────────────────────
# SINGLE-STATE TRAINER
# ──────────────────────────────────────────────────────────────────────────────

def train_state(state: str, df: pd.DataFrame, artifacts_dir: Path, train_lstm: bool = True):
    """
    Train all four models for one state, evaluate them on the hold-out set,
    select the best model, and save everything.

    Returns a dict with metrics + forecast for that state.
    """
    print(f"\n{'='*60}")
    print(f"  Training: {state}")
    print(f"{'='*60}")

    state_dir = artifacts_dir / state.replace(" ", "_")
    state_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data
    weekly   = clean_and_resample(df, state)
    train_df, test_df = train_test_split_ts(weekly, test_weeks=TEST_WEEKS)

    results    = {}
    forecasts  = {}

    # ── 1. SARIMA ────────────────────────────────────────────────────────────
    try:
        t0 = time.time()
        m = SARIMAModel(order=(1, 1, 1), seasonal_order=(1, 1, 0, 52))
        m.fit(train_df)
        metrics = m.evaluate(test_df)
        future  = m.predict(FORECAST_WEEKS)
        m.save(str(state_dir / "sarima.pkl"))
        results["sarima"]   = metrics
        forecasts["sarima"] = future
        print(f"  SARIMA  | RMSE={metrics['rmse']:.0f}  MAE={metrics['mae']:.0f}  MAPE={metrics['mape']:.2f}%  ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"  SARIMA  | FAILED: {e}")

    # ── 2. Prophet ───────────────────────────────────────────────────────────
    try:
        t0 = time.time()
        m = ProphetModel()
        m.fit(train_df)
        metrics = m.evaluate(test_df)
        future  = m.predict(FORECAST_WEEKS)
        m.save(str(state_dir / "prophet.pkl"))
        results["prophet"]   = metrics
        forecasts["prophet"] = future
        print(f"  Prophet | RMSE={metrics['rmse']:.0f}  MAE={metrics['mae']:.0f}  MAPE={metrics['mape']:.2f}%  ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"  Prophet | FAILED: {e}")

    # ── 3. XGBoost ───────────────────────────────────────────────────────────
    try:
        t0 = time.time()
        m = XGBoostModel()
        m.fit(train_df)
        metrics = m.evaluate(test_df)
        future  = m.predict(FORECAST_WEEKS)
        m.save(str(state_dir / "xgboost.pkl"))
        results["xgboost"]   = metrics
        forecasts["xgboost"] = future
        print(f"  XGBoost | RMSE={metrics['rmse']:.0f}  MAE={metrics['mae']:.0f}  MAPE={metrics['mape']:.2f}%  ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"  XGBoost | FAILED: {e}")

    # ── 4. LSTM ──────────────────────────────────────────────────────────────
    if train_lstm:
        try:
            t0 = time.time()
            m = LSTMModel(epochs=60, batch_size=16)
            m.fit(train_df)
            metrics = m.evaluate(test_df)
            future  = m.predict(FORECAST_WEEKS)
            m.save(str(state_dir / "lstm.pkl"))
            results["lstm"]   = metrics
            forecasts["lstm"] = future
            print(f"  LSTM    | RMSE={metrics['rmse']:.0f}  MAE={metrics['mae']:.0f}  MAPE={metrics['mape']:.2f}%  ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"  LSTM    | FAILED: {e}")

    # ── Select best model (lowest RMSE) ──────────────────────────────────────
    best_model = min(results, key=lambda k: results[k]["rmse"])
    best_forecast = forecasts[best_model]

    print(f"\n  ★ Best model for {state}: {best_model.upper()} (RMSE={results[best_model]['rmse']:.0f})")

    # ── Generate future dates ────────────────────────────────────────────────
    last_date    = weekly["ds"].max()
    future_dates = make_future_dates(last_date, FORECAST_WEEKS)

    forecast_df = pd.DataFrame({
        "ds":    future_dates["ds"],
        "yhat":  best_forecast,
        "model": best_model,
    })

    # ── Save per-state summary ────────────────────────────────────────────────
    summary = {
        "state":       state,
        "best_model":  best_model,
        "metrics":     results,
        "forecast":    [
            {"date": str(row["ds"].date()), "sales": round(row["yhat"], 2)}
            for _, row in forecast_df.iterrows()
        ],
    }
    with open(state_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── Plot ─────────────────────────────────────────────────────────────────
    _plot_forecast(weekly, forecast_df, results, state, state_dir)

    return summary


# ──────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ──────────────────────────────────────────────────────────────────────────────

def _plot_forecast(weekly, forecast_df, results, state, out_dir):
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # ── Top panel: historical + forecast ─────────────────────────────────────
    ax = axes[0]
    ax.plot(weekly["ds"], weekly["y"], label="Historical", color="steelblue", linewidth=1.5)
    ax.plot(forecast_df["ds"], forecast_df["yhat"], label="Forecast (best)", color="orange",
            linewidth=2, linestyle="--", marker="o", markersize=5)
    ax.set_title(f"{state} — 8-Week Sales Forecast", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales ($)")
    ax.legend()
    ax.grid(alpha=0.3)

    # ── Bottom panel: model comparison bar chart ──────────────────────────────
    ax2 = axes[1]
    models = list(results.keys())
    rmses  = [results[m]["rmse"] for m in models]
    colors = ["#4CAF50" if m == min(results, key=lambda k: results[k]["rmse"]) else "#90CAF9"
              for m in models]
    bars = ax2.bar(models, rmses, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_title("Model Comparison — RMSE (lower is better)", fontsize=12)
    ax2.set_ylabel("RMSE")
    for bar, v in zip(bars, rmses):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(rmses) * 0.01,
                 f"{v:,.0f}", ha="center", fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "forecast_plot.png", dpi=100, bbox_inches="tight")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(data_path: str, artifacts_dir: str, states: list = None,
                 train_lstm: bool = True):
    """
    Train all models for every state (or a subset).
    Saves a master `all_results.json` in artifacts_dir.
    """
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(data_path)
    all_states = sorted(df["state"].unique())
    if states:
        all_states = [s for s in all_states if s in states]

    print(f"\nTraining on {len(all_states)} states …\n")

    all_results = {}
    for state in all_states:
        try:
            summary = train_state(state, df, artifacts_dir, train_lstm=train_lstm)
            all_results[state] = summary
        except Exception as e:
            print(f"  ERROR for {state}: {e}")

    # Save master results
    with open(artifacts_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\n✅  Training complete.  Results saved to {artifacts_dir}/all_results.json")
    return all_results


# ──────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train forecasting models")
    parser.add_argument("--data",      default="data/sales_data.xlsx")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--states",    nargs="*", default=None,
                        help="Subset of states to train (default: all)")
    parser.add_argument("--no-lstm",   action="store_true",
                        help="Skip LSTM training (faster, no GPU needed)")
    args = parser.parse_args()

    run_pipeline(
        data_path=args.data,
        artifacts_dir=args.artifacts,
        states=args.states,
        train_lstm=not args.no_lstm,
    )
