# COMPLETE SETUP GUIDE
# Time-Series Sales Forecasting System
# ====================================================
# Assume the reader is a complete beginner.
# Every step is explained from scratch.
# ====================================================


## TABLE OF CONTENTS

1.  What Is This Project?
2.  Project Folder Explained
3.  Phase 1 — Train in Google Colab
4.  Phase 2 — Download Trained Models
5.  Phase 3 — Set Up Project Locally (VS Code)
6.  Phase 4 — Run the FastAPI Backend
7.  Phase 5 — Test the API
8.  Phase 6 — Deploy to Render (Public)
9.  How to Retrain Later
10. How to Update the Dataset
11. Troubleshooting Common Errors
12. API Reference (All Endpoints)
13. Postman Examples
14. Curl Examples


---
## 1. WHAT IS THIS PROJECT?
---

This system:
  - Takes historical weekly sales data for 43 US states
  - Trains 4 different forecasting models on each state:
      SARIMA, Prophet, XGBoost, LSTM
  - Automatically picks the most accurate model per state
  - Forecasts the next 8 weeks of sales
  - Exposes all of this via a REST API

Think of it like a "sales prediction machine" that you can ask questions:
  "Hey API, what will California's sales look like for the next 8 weeks?"


---
## 2. PROJECT FOLDER EXPLAINED
---

forecasting_project/
│
├── app/                        ← The Python source code
│   ├── __init__.py             ← Makes app/ a Python package
│   ├── main.py                 ← FastAPI web server (all API endpoints)
│   ├── models.py               ← SARIMA, Prophet, XGBoost, LSTM classes
│   ├── preprocessing.py        ← Data cleaning + feature engineering
│   └── trainer.py              ← Training pipeline (runs all models)
│
├── notebooks/
│   └── training_notebook.ipynb ← Google Colab notebook (run this first!)
│
├── data/
│   └── sales_data.xlsx         ← Your dataset (put it here)
│
├── artifacts/                  ← Auto-created during training
│   ├── all_results.json        ← Summary of all states + metrics
│   └── California/             ← One folder per state
│       ├── summary.json        ← Best model + metrics for this state
│       ├── sarima.pkl          ← Saved SARIMA model
│       ├── prophet.pkl         ← Saved Prophet model
│       ├── xgboost.pkl         ← Saved XGBoost model
│       ├── lstm.pkl            ← Saved LSTM model
│       └── forecast_plot.png   ← Chart of the forecast
│
├── requirements.txt            ← Python libraries to install
├── Procfile                    ← Used by Render for deployment
├── render.yaml                 ← Render deployment configuration
├── README.md                   ← Project overview
└── COMPLETE_SETUP_GUIDE.md     ← THIS FILE


---
## 3. PHASE 1 — TRAIN IN GOOGLE COLAB
---

WHY COLAB?
Google Colab gives you a free T4 GPU which makes LSTM training
10x faster than a normal laptop CPU.


### STEP 3.1 — OPEN GOOGLE COLAB

1. Go to: https://colab.research.google.com
2. Sign in with your Google account
3. Click: File → Upload notebook
4. Upload: notebooks/training_notebook.ipynb
   (This file is inside the project folder you received)


### STEP 3.2 — ENABLE GPU (IMPORTANT!)

Without a GPU, LSTM training will be very slow.

1. In Colab, click: Runtime (top menu)
2. Click: Change runtime type
3. Under "Hardware accelerator", select: T4 GPU
4. Click: Save

You should see "T4" in the top-right corner of Colab if successful.


### STEP 3.3 — RUN THE NOTEBOOK CELL BY CELL

Each "cell" is a block of code. Run them in order from top to bottom.

HOW TO RUN A CELL:
  - Click on the cell
  - Press: Shift + Enter
  OR
  - Click the ▶ play button on the left side of the cell

Go through each step:

  STEP 1 — Check GPU
    Expected output: "✅ GPU is available!"
    If you see "No GPU detected", go back to Step 3.2.

  STEP 2 — Install Libraries
    Expected output: "✅ All libraries installed!"
    This takes ~2-4 minutes. Wait for it to finish.

  STEP 3 — Mount Google Drive
    A popup will appear asking for permission.
    Click: Allow
    Expected output: "✅ Google Drive mounted."

  STEP 4 — Upload Dataset
    A file chooser will appear.
    Navigate to your Excel file (sales_data.xlsx or Forecasting_Case-_Study.xlsx)
    Click: Open
    Expected output: "✅ Uploaded and saved to: /content/drive/..."

  STEP 5 — Preprocessing
    Expected output: Dataset shape, states count, date range.
    No errors means success.

  STEP 6 — Model Functions
    Expected output: "✅ Model functions defined."
    This just loads the code. No training yet.

  STEP 7 — TRAIN ALL MODELS  ← THE MAIN TRAINING STEP
    This trains SARIMA + Prophet + XGBoost + LSTM for all 43 states.

    Expected duration:
      - Without LSTM: ~15-25 minutes
      - With LSTM:    ~45-90 minutes (depends on GPU)

    Expected output (for each state):
      ── California ──
        SARIMA  RMSE=   4,234,512  MAE=   3,100,000  MAPE=  5.23%  (12.3s)
        Prophet RMSE=   3,891,201  MAE=   2,900,000  MAPE=  4.87%  (8.1s)
        XGBoost RMSE=   4,567,890  MAE=   3,400,000  MAPE=  5.91%  (3.2s)
        LSTM    RMSE=   3,512,000  MAE=   2,700,000  MAPE=  4.21%  (45.1s)
        ★ Best: LSTM (RMSE=3,512,000)

    NOTE: If TRAIN_LSTM = True is taking too long, you can:
      - Change it to: TRAIN_LSTM = False  (in the Step 7 cell)
      - Re-run just that cell
      - This skips LSTM and only trains the other 3 models

  STEP 8 — Visualise Results
    Expected output: Charts showing model comparison and forecasts.

  STEP 9 — Summary Table
    Expected output: A table showing each state + its best model + metrics.

  STEP 10 — Download Artifacts
    Expected output: A file download starts in your browser.
    A file named: artifacts_export.zip  will download.
    SAVE THIS FILE — you need it in Phase 3.


### STEP 3.4 — VERIFY TRAINING WORKED

After Step 7, in the left sidebar of Colab, click the folder icon 📁.
Navigate to: drive/MyDrive/forecasting_project/artifacts/

You should see:
  - all_results.json
  - One folder per state (e.g. California/, Texas/, New_York/, etc.)

If these exist, training was successful!


---
## 4. PHASE 2 — DOWNLOAD TRAINED MODELS
---

After Step 10 in the notebook, your browser should have started downloading:
  artifacts_export.zip

If the download didn't start automatically:
1. In Colab's left sidebar, click the 📁 folder icon
2. Navigate to: /content/
3. Find: artifacts_export.zip
4. Right-click it → Download

SAVE IT somewhere you can find it. We'll use it in the next phase.


---
## 5. PHASE 3 — SET UP PROJECT LOCALLY (VS CODE)
---

### STEP 5.1 — INSTALL PYTHON

Check if Python is installed:
  Open Terminal (Mac/Linux) or Command Prompt (Windows)
  Type: python --version
  Expected: Python 3.10.x or higher

If Python is not installed:
  Go to: https://www.python.org/downloads/
  Download Python 3.11 (recommended)
  Install it (check "Add Python to PATH" on Windows!)


### STEP 5.2 — INSTALL VS CODE (OPTIONAL BUT RECOMMENDED)

Download VS Code from: https://code.visualstudio.com/
Install it normally.
Open VS Code.


### STEP 5.3 — PLACE THE PROJECT FILES

You should have received the project as a folder called:
  forecasting_project/

Put it somewhere easy to find, like:
  Windows: C:\Users\YourName\forecasting_project\
  Mac:     /Users/YourName/forecasting_project/
  Linux:   /home/yourname/forecasting_project/


### STEP 5.4 — EXTRACT ARTIFACTS

Take the artifacts_export.zip you downloaded from Colab.

Extract it into: forecasting_project/artifacts/

After extracting, your folder should look like:
  forecasting_project/
  └── artifacts/
      ├── all_results.json
      ├── California/
      │   ├── summary.json
      │   ├── sarima.pkl
      │   ├── prophet.pkl
      │   └── ...
      └── Texas/
          └── ...

IMPORTANT: The artifacts folder must be INSIDE forecasting_project/
(not inside a subfolder like artifacts/artifacts_export/artifacts/)


### STEP 5.5 — OPEN TERMINAL IN THE PROJECT FOLDER

Windows:
  1. Open Command Prompt
  2. Type: cd C:\Users\YourName\forecasting_project
  3. Press Enter

Mac/Linux:
  1. Open Terminal
  2. Type: cd /Users/YourName/forecasting_project
  3. Press Enter

In VS Code:
  1. Open the forecasting_project folder in VS Code
  2. Press: Ctrl + ` (backtick) to open the integrated terminal


### STEP 5.6 — CREATE A VIRTUAL ENVIRONMENT

A virtual environment keeps this project's libraries separate from
other Python projects on your computer.

In the terminal:

  Windows:
    python -m venv venv
    venv\Scripts\activate

  Mac/Linux:
    python -m venv venv
    source venv/bin/activate

EXPECTED: Your terminal prompt now shows (venv) at the start.
Example: (venv) C:\Users\YourName\forecasting_project>

If you see (venv), the environment is active!


### STEP 5.7 — INSTALL REQUIRED LIBRARIES

Make sure you're in the project folder with (venv) active, then:

  pip install -r requirements.txt

This installs all libraries listed in requirements.txt.

EXPECTED: Lots of text scrolling, ending with:
  "Successfully installed ..."

This takes 3-8 minutes. Be patient!

Common issues:
  - If pip install fails for prophet: try  pip install prophet --no-build-isolation
  - If tensorflow fails: try  pip install tensorflow-cpu  (for CPU-only)


---
## 6. PHASE 4 — RUN THE FASTAPI BACKEND
---

### STEP 6.1 — START THE SERVER

Make sure:
  - You are in the forecasting_project/ folder
  - The (venv) is active (you see it in the terminal prompt)
  - The artifacts/ folder exists with trained models

Then run:

  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

EXPECTED OUTPUT:
  INFO:     Will watch for changes in these directories: ['.']
  INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
  INFO:     Started reloader process [12345] using WatchFiles
  INFO:     Started server process [12346]
  INFO:     Waiting for application startup.
  INFO:     Application startup complete.

If you see this, the server is running! 🎉


### STEP 6.2 — OPEN THE API IN BROWSER

Open your web browser and go to:
  http://localhost:8000/docs

You will see the Swagger UI — an interactive page where you can
test ALL the API endpoints directly in the browser!

Also try:
  http://localhost:8000       ← Simple welcome message
  http://localhost:8000/health  ← Health status


### STEP 6.3 — STOP THE SERVER

Press: Ctrl + C  in the terminal to stop the server.


---
## 7. PHASE 5 — TEST THE API
---

The API must be running (see Phase 4) before testing.


### METHOD A — SWAGGER UI (EASIEST)

Go to: http://localhost:8000/docs

Click on any endpoint (e.g. GET /forecast)
Click "Try it out"
Fill in the parameters (e.g. state = California)
Click "Execute"
See the response below


### METHOD B — CURL (TERMINAL)

Open a NEW terminal window (keep the server running in the first one).

```bash
# 1. Welcome message
curl http://localhost:8000/

# 2. Health check
curl http://localhost:8000/health

# 3. List all trained states
curl http://localhost:8000/states

# 4. Get 8-week forecast for California (best model auto-selected)
curl "http://localhost:8000/forecast?state=California"

# 5. Get forecast using a specific model
curl "http://localhost:8000/forecast?state=Texas&model=prophet"

# 6. Get forecast for only 4 weeks
curl "http://localhost:8000/forecast?state=Florida&weeks=4"

# 7. Get model metrics for New York
curl "http://localhost:8000/metrics?state=New+York"

# 8. Get aggregated metrics for ALL states
curl http://localhost:8000/metrics

# 9. List models and metrics for a state
curl "http://localhost:8000/models?state=California"

# 10. Trigger training for 2 states (no LSTM, fast)
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"states": ["California", "Texas"], "train_lstm": false}'
```

### METHOD C — POSTMAN

1. Download Postman: https://www.postman.com/downloads/
2. Open Postman
3. Click "New" → "HTTP Request"
4. Enter URL: http://localhost:8000/forecast
5. Add query param: state = California
6. Click "Send"
7. See the JSON response


### EXPECTED FORECAST RESPONSE

```json
{
  "state": "California",
  "model_used": "lstm",
  "weeks": 8,
  "forecast": [
    {"date": "2024-01-07", "sales": 234567890.12},
    {"date": "2024-01-14", "sales": 238901234.56},
    {"date": "2024-01-21", "sales": 241234567.89},
    {"date": "2024-01-28", "sales": 239876543.21},
    {"date": "2024-02-04", "sales": 243210987.65},
    {"date": "2024-02-11", "sales": 245678901.23},
    {"date": "2024-02-18", "sales": 248901234.56},
    {"date": "2024-02-25", "sales": 251234567.89}
  ]
}
```

### EXPECTED METRICS RESPONSE

```json
{
  "state": "California",
  "best_model": "lstm",
  "metrics": {
    "sarima":  {"rmse": 4234512.00, "mae": 3100000.00, "mape": 5.23},
    "prophet": {"rmse": 3891201.00, "mae": 2900000.00, "mape": 4.87},
    "xgboost": {"rmse": 4567890.00, "mae": 3400000.00, "mape": 5.91},
    "lstm":    {"rmse": 3512000.00, "mae": 2700000.00, "mape": 4.21}
  }
}
```


---
## 8. PHASE 6 — DEPLOY TO RENDER (PUBLIC URL)
---

Render gives you a free public URL for your API.
Anyone on the internet can then access it!


### STEP 8.1 — PUSH TO GITHUB

You need a GitHub account. If you don't have one:
  Go to: https://github.com  and create an account.

Install Git: https://git-scm.com/downloads

In your terminal (inside forecasting_project/):

  git init
  git add .
  git commit -m "Initial commit: forecasting project"

Create a new repository on GitHub:
  1. Go to https://github.com/new
  2. Repository name: forecasting-project
  3. Make it Public or Private (your choice)
  4. Click "Create repository"

Then push your code (copy the commands GitHub shows you):

  git remote add origin https://github.com/YOURUSERNAME/forecasting-project.git
  git branch -M main
  git push -u origin main

IMPORTANT: The artifacts/ folder is in .gitignore and will NOT be pushed.
You will re-upload it after deployment (see Step 8.4).


### STEP 8.2 — CREATE A RENDER ACCOUNT

Go to: https://render.com
Click: "Get Started for Free"
Sign up using your GitHub account


### STEP 8.3 — CREATE A WEB SERVICE

1. In Render dashboard, click: "New +"
2. Click: "Web Service"
3. Connect your GitHub account if prompted
4. Find your repository: forecasting-project
5. Click: "Connect"

Fill in these settings:

  Name:              sales-forecasting-api
  Region:            Oregon (US West) — or closest to you
  Branch:            main
  Root Directory:    (leave empty)
  Runtime:           Python 3
  Build Command:     pip install -r requirements.txt
  Start Command:     uvicorn app.main:app --host 0.0.0.0 --port $PORT

Scroll down to "Environment Variables":
  Click "Add Environment Variable"
  Key:   ARTIFACTS_DIR
  Value: artifacts

  Click "Add Environment Variable" again
  Key:   DATA_PATH
  Value: data/sales_data.xlsx

Choose the FREE plan (it's fine for a portfolio project).

Click: "Create Web Service"

Render will now build and deploy your API.
This takes 3-7 minutes.

EXPECTED: Build logs scroll by. At the end you see:
  "Your service is live 🎉"


### STEP 8.4 — UPLOAD ARTIFACTS TO RENDER

Since we didn't push artifacts/ to GitHub (it's in .gitignore),
we need to get the models onto Render another way.

OPTION A — Use the /train endpoint (recommended)

Once your service is live, call the training endpoint:

  curl -X POST https://your-service-name.onrender.com/train \
    -H "Content-Type: application/json" \
    -d '{"train_lstm": false}'

This trains all models directly on Render's server.
NOTE: Free Render instances have limited RAM/CPU, so use train_lstm: false.

Monitor training status:
  curl https://your-service-name.onrender.com/health

Wait until status shows "completed".

OPTION B — Add persistent disk (paid feature)

If you want to upload pre-trained models:
1. In Render, go to your service → Disks
2. Add a disk mounted at /opt/render/project/src/artifacts
3. SSH into the service and upload your pkl files

OPTION A is simpler for a free tier.


### STEP 8.5 — TEST YOUR PUBLIC API

Replace YOUR-SERVICE-NAME with your actual Render service name:

  curl https://YOUR-SERVICE-NAME.onrender.com/health
  curl "https://YOUR-SERVICE-NAME.onrender.com/forecast?state=California"

Your API is now publicly accessible!

Note: Free Render services "sleep" after 15 minutes of inactivity.
The first request after sleeping takes ~30 seconds. This is normal.


---
## 9. HOW TO RETRAIN LATER
---

If you get new data or want to retrain with updated parameters:

OPTION A — Retrain via API (simplest)

  curl -X POST http://localhost:8000/train \
    -H "Content-Type: application/json" \
    -d '{"train_lstm": true}'

OPTION B — Retrain via command line

  python -m app.trainer \
    --data data/sales_data.xlsx \
    --artifacts artifacts

OPTION C — Retrain specific states only

  python -m app.trainer \
    --data data/sales_data.xlsx \
    --artifacts artifacts \
    --states California Texas "New York"

OPTION D — Retrain in Colab

  Open the notebook again in Colab.
  Re-run from Step 7.
  Download the new artifacts_export.zip.
  Replace your local artifacts/ folder.


---
## 10. HOW TO UPDATE THE DATASET
---

1. Place your new Excel file in: forecasting_project/data/sales_data.xlsx
   (overwrite the old file)

2. The file must have these columns:
     State    — US state name (string)
     Date     — Date (any format, will be auto-parsed)
     Total    — Sales amount (number)
     Category — Product category (string)

3. Retrain the models (see Section 9)

4. The API will automatically use the new models after retraining.


---
## 11. TROUBLESHOOTING COMMON ERRORS
---

### Error: "ModuleNotFoundError: No module named 'prophet'"
Solution:
  pip install prophet --no-build-isolation
  (If on Mac with M1/M2, also try: pip install pystan==2.19.1.1 prophet)

### Error: "ModuleNotFoundError: No module named 'tensorflow'"
Solution:
  pip install tensorflow
  (On older Macs: pip install tensorflow-macos)

### Error: "FileNotFoundError: data/sales_data.xlsx not found"
Solution:
  Make sure your Excel file is at: forecasting_project/data/sales_data.xlsx
  The file name must be exactly: sales_data.xlsx

### Error: "404 No trained models found for state 'California'"
Solution:
  Training hasn't been done yet, or artifacts weren't placed correctly.
  Check that forecasting_project/artifacts/California/summary.json exists.
  If not, run training first.

### Error: "Address already in use: port 8000"
Solution:
  Another process is using port 8000.
  Either stop that process, or run on a different port:
  uvicorn app.main:app --reload --port 8001

### Error: uvicorn not found
Solution:
  Make sure your virtual environment is active!
  Windows: venv\Scripts\activate
  Mac/Linux: source venv/bin/activate
  Then retry.

### SARIMA training hangs / takes too long
Solution:
  SARIMA with seasonal_order=(1,1,0,52) can be slow for some states.
  In models.py, change seasonal_order to (0,1,0,52) for faster training.

### Out of memory during LSTM training
Solution:
  In notebooks/training_notebook.ipynb, find: TRAIN_LSTM = True
  Change it to: TRAIN_LSTM = False
  This skips LSTM entirely.

### Render deployment fails with "Build failed"
Solution:
  Check the build logs in Render dashboard.
  Most common issue: a library can't be installed.
  Try removing tensorflow from requirements.txt if you aren't using
  LSTM in production (use train_lstm: false).

### Render API returns 500 error after deployment
Solution:
  The models haven't been trained on Render yet.
  Call: POST /train  with train_lstm: false
  Wait for it to complete (check GET /health).


---
## 12. API REFERENCE
---

BASE URL (local):  http://localhost:8000
BASE URL (Render): https://your-service.onrender.com


### GET /
Returns a welcome message.
No parameters.

Response:
{
  "message": "Sales Forecasting API is running 🚀",
  "docs": "/docs",
  "redoc": "/redoc"
}


### GET /health
Returns health status and training state.

Response:
{
  "status": "healthy",
  "artifacts_dir": "artifacts",
  "artifacts_exist": true,
  "models_trained": true,
  "training_status": "completed"
}

training_status values:
  "idle"      — No training triggered yet
  "starting"  — Training just queued
  "running"   — Training in progress
  "completed" — Training finished successfully
  "failed"    — Training encountered an error


### POST /train
Triggers model training in the background.

Request body:
{
  "states": null,       ← null or omit = train ALL states
  "train_lstm": true    ← false = skip LSTM (faster)
}

Response:
{
  "message": "Training started in the background.",
  "states": "all",
  "train_lstm": true,
  "tip": "Poll GET /health to check training status."
}

Poll GET /health until training_status = "completed".


### GET /states
Lists all states that have been trained.

Response:
{
  "states": ["Alabama", "Arizona", "Arkansas", ...],
  "count": 43
}


### GET /models?state=California
Lists all available models for a state with their metrics.

Parameters:
  state  (required) — US state name

Response:
{
  "state": "California",
  "best_model": "lstm",
  "models": {
    "sarima":  {"rmse": 4234512.0, "mae": 3100000.0, "mape": 5.23},
    "prophet": {"rmse": 3891201.0, "mae": 2900000.0, "mape": 4.87},
    "xgboost": {"rmse": 4567890.0, "mae": 3400000.0, "mape": 5.91},
    "lstm":    {"rmse": 3512000.0, "mae": 2700000.0, "mape": 4.21}
  }
}


### GET /forecast?state=California&model=prophet&weeks=8
Returns the sales forecast for a state.

Parameters:
  state  (required)  — US state name (e.g. California, New York)
  model  (optional)  — Model to use: sarima, prophet, xgboost, lstm
                       Default: best model (lowest RMSE)
  weeks  (optional)  — Number of weeks to forecast (1–8). Default: 8

Response:
{
  "state": "California",
  "model_used": "lstm",
  "weeks": 8,
  "forecast": [
    {"date": "2024-01-07", "sales": 234567890.12},
    ...
  ]
}


### GET /metrics
Returns evaluation metrics.

Parameters:
  state  (optional) — If provided, returns metrics for that state only.
                      If omitted, returns aggregated stats for all states.

Single state response:
{
  "state": "California",
  "best_model": "lstm",
  "metrics": {
    "sarima":  {"rmse": ..., "mae": ..., "mape": ...},
    ...
  }
}

All states response:
{
  "total_states": 43,
  "model_summary": {
    "sarima":  {"states_trained": 43, "avg_rmse": ..., "avg_mae": ..., "avg_mape": ...},
    "prophet": {...},
    "xgboost": {...},
    "lstm":    {...}
  },
  "best_model_wins": {
    "lstm":    18,
    "prophet": 14,
    "sarima":   7,
    "xgboost":  4
  }
}


---
## 13. POSTMAN EXAMPLES
---

To import these into Postman:
1. Open Postman
2. Click "Import" (top left)
3. Paste the URL or JSON

COLLECTION SETUP:
  Create a new Collection called "Sales Forecasting API"
  Set base URL variable: {{base_url}} = http://localhost:8000

REQUESTS:

  1. Health Check
     Method: GET
     URL: {{base_url}}/health

  2. List States
     Method: GET
     URL: {{base_url}}/states

  3. Forecast — Best Model
     Method: GET
     URL: {{base_url}}/forecast
     Params: state = California

  4. Forecast — Specific Model
     Method: GET
     URL: {{base_url}}/forecast
     Params:
       state = Texas
       model = prophet

  5. Forecast — Short Term
     Method: GET
     URL: {{base_url}}/forecast
     Params:
       state = Florida
       weeks = 4

  6. Metrics — Single State
     Method: GET
     URL: {{base_url}}/metrics
     Params: state = New York

  7. Metrics — All States
     Method: GET
     URL: {{base_url}}/metrics

  8. Train — All States No LSTM
     Method: POST
     URL: {{base_url}}/train
     Body (JSON):
     {
       "states": null,
       "train_lstm": false
     }

  9. Train — Specific States With LSTM
     Method: POST
     URL: {{base_url}}/train
     Body (JSON):
     {
       "states": ["California", "Texas", "New York"],
       "train_lstm": true
     }


---
## 14. CURL EXAMPLES
---

Run these in your terminal while the API server is running.

# ── BASIC ────────────────────────────────────────────────────────────────────

# Welcome
curl http://localhost:8000/

# Health
curl http://localhost:8000/health

# List states
curl http://localhost:8000/states

# ── FORECASTS ────────────────────────────────────────────────────────────────

# California — best model
curl "http://localhost:8000/forecast?state=California"

# Texas — Prophet model specifically
curl "http://localhost:8000/forecast?state=Texas&model=prophet"

# Florida — SARIMA, only 4 weeks
curl "http://localhost:8000/forecast?state=Florida&model=sarima&weeks=4"

# New York — XGBoost
curl "http://localhost:8000/forecast?state=New+York&model=xgboost"

# ── METRICS ─────────────────────────────────────────────────────────────────

# All states summary
curl http://localhost:8000/metrics

# California only
curl "http://localhost:8000/metrics?state=California"

# ── MODELS ──────────────────────────────────────────────────────────────────

# Models for California
curl "http://localhost:8000/models?state=California"

# ── TRAINING ────────────────────────────────────────────────────────────────

# Train all states (no LSTM — fast)
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"train_lstm": false}'

# Train all states (with LSTM)
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"train_lstm": true}'

# Train specific states only
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"states": ["California", "Texas", "Florida"], "train_lstm": false}'

# ── PRETTY PRINT JSON (install jq first: https://jqlang.github.io/jq/) ──────

curl "http://localhost:8000/forecast?state=California" | python3 -m json.tool


---
## FINAL CHECKLIST — VERIFY EVERYTHING WORKS
---

Before submitting or presenting your project, run through this checklist:

Phase 1 — Colab Training
  [ ] GPU was enabled in Colab
  [ ] All notebook cells ran without errors
  [ ] artifacts_export.zip was downloaded

Phase 3 — Local Setup
  [ ] Python 3.10+ is installed
  [ ] Virtual environment is active (venv)
  [ ] pip install -r requirements.txt completed without errors
  [ ] artifacts/ folder is inside forecasting_project/
  [ ] artifacts/all_results.json exists
  [ ] At least a few state folders exist (e.g. artifacts/California/)

Phase 4 — API Running
  [ ] uvicorn app.main:app --reload runs without errors
  [ ] http://localhost:8000 shows welcome message
  [ ] http://localhost:8000/docs loads Swagger UI

Phase 5 — API Testing
  [ ] GET /health returns {"status": "healthy", "models_trained": true}
  [ ] GET /states returns a list of state names
  [ ] GET /forecast?state=California returns 8 date+sales pairs
  [ ] GET /metrics returns model comparison data
  [ ] POST /train does not crash (may take a while to complete)

Phase 6 — Deployment (optional)
  [ ] Code pushed to GitHub
  [ ] Render service created
  [ ] Build succeeded on Render
  [ ] Training triggered via POST /train on Render
  [ ] Public URL responds to GET /forecast?state=California

Congratulations! Your forecasting system is complete. 🎉
