# Heart Attack Prediction

ML pipeline that classifies cardiac risk levels (Low / Moderate / High) from multiplex biomarker measurements, exposes a FastAPI backend, and ships with a small HTML dashboard for manual scoring.

## Project Layout

- `Heart_attack_dataset.csv` – raw dataset (has a metadata line that we skip on load).
- `model/train.py` – trains several classifiers, picks the best, and writes artifacts to `model/`.
- `backend/main.py` – FastAPI service that loads the artifacts and serves `/predict`.
- `frontend/index.html` – static UI that posts biomarker values to the API.
- `requirements.txt` – Python dependencies for both training and serving.

## Prerequisites

- Python ≥ 3.10 (tested on 3.12).
- `pip` and a working C/C++ build chain (for LightGBM/XGBoost wheels if binaries are missing).

## Setup & Run

# Create a virtual environment (macOS)
python3 -m venv venv
source venv/bin/activate

# Create a virtual environment (Windows)
python -m venv venv
venv\Scripts\activate

# Install dependencies (macOS)
pip3 install -r requirements.txt

# Install dependencies (Windows)
pip install -r requirements.txt

# Train or refresh the ML model (macOS)
python3 model/train.py

# Train or refresh the ML model (Windows)
python model/train.py

# Start the FastAPI backend on port 8000 (same for both OS)
uvicorn backend.main:app --host 127.0.0.1 --port 8000

# Run the frontend using a simple local server (macOS)
cd frontend
python3 -m http.server 8080

# Run the frontend using a simple local server (Windows)
cd frontend
python -m http.server 8080



Steps 1‑4 must finish before the API can answer predictions. Step 5 is only needed if you want to load the HTML page via `http://127.0.0.1:8080/`; alternatively open `frontend/index.html` directly in a browser.

## API Reference

- `GET /` – health probe.
- `POST /predict` – accepts JSON with the biomarker fields used during training (see `backend/main.py::BiomarkerInput`). Example payload:

```json
{
  "I_620": 585.7,
  "I_540": 735.3,
  "R620": 1.485,
  "R540": 3.025,
  "cTnI_ng_mL": 14.35,
  "Myoglobin_ng_mL": 15.025,
  "Raw_Fluorescence_au": 2149.01,
  "DeltaF_au": 949.01,
  "Calculated_Troponin_ng_mL": 1.1863,
  "Peak_Current_uA": 280.739,
  "BNP_pg_mL": 623.42
}
```

Response:

```json
{
  "risk_category": "High",
  "probability": {
    "High": 0.82,
    "Low": 0.03,
    "Moderate": 0.15
  },
  "confidence": 0.82,
  "interpretation": "High Troponin I levels detected (Strong MI Indicator). Elevated BNP levels suggest potential heart failure."
}
```

## Frontend

The single-page dashboard in `frontend/index.html` mirrors the JSON schema above, sends inputs to `http://localhost:8000/predict`, and renders the response plus class probabilities using Chart.js. Ensure the backend is running before clicking “Analyze Risk”.

## Troubleshooting

- **Permission errors under `~/.matplotlib` or CPython cache**: export `MPLCONFIGDIR=/tmp/mpl` (or another writable dir) before training.
- **`Model not loaded` from API**: re-run `python3 model/train.py` so `model/pipeline.pkl`, `model/label_encoder.pkl`, and `model/features.pkl` exist, then restart Uvicorn.
- **Frontend CORS issues**: serve it via `python3 -m http.server` rather than opening the file directly, or adjust the backend CORS settings in `backend/main.py`.

Feel free to extend this stack with monitoring, dockerization, or additional endpoints as needed.