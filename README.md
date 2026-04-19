# Cardiac Risk Prediction AI

An end-to-end Machine Learning system that classifies cardiac risk levels (**Low**, **Moderate**, **High**) based on multiplex biomarker measurements. The system features a robust FastAPI backend, a comprehensive HTML5 dashboard, and automated model explanation via SHAP.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-v0.136.0-green.svg)

## 🚀 Key Features

- **Advanced ML Pipeline**: Automated training with Stratified 5-Fold Cross-Validation and SMOTE for class imbalance.
- **Explainable AI (XAI)**: Integrated SHAP explainer to provide feature-level importance for every prediction.
- **Robust API**: FastAPI backend with Pydantic validation, structured logging, rate limiting (slowapi), and API-Key security.
- **Modern Dashboard**: Responsive UI with Chart.js visualization, prediction history, and real-time biomarker monitoring simulations.
- **Production Ready**: Fully dockerized with Docker Compose support and GitHub Actions CI for linting and testing.

## 🏗️ Architecture

Refer to [README_ARCH.md](./README_ARCH.md) for a detailed system diagram and component breakdown.

## 📁 Project Structure

- `backend/` - FastAPI server and API logic.
- `frontend/` - Responsive dashboard (HTML/JS/CSS).
- `model/` - Training scripts and serialized artifacts (`.pkl`).
- `data/` - Training dataset (`Heart_attack_dataset.csv`).
- `tests/` - Pytest suite for API validation.
- `Dockerfile` & `docker-compose.yml` - Containerization and orchestration.
- `.github/workflows/` - CI/CD pipeline.

## 🛠️ Quick Start

### Prerequisites
- Python 3.12+
- Docker & Docker Compose (optional)

### Local Setup
1. **Clone and Setup Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   cp .env.example .env
   ```

2. **Train the Model**:
   ```bash
   python model/train.py
   ```

3. **Run the API**:
   ```bash
   uvicorn backend.main:app --host 0.0.0.0 --port 8000
   ```

4. **Access the Dashboard**:
   Open `http://localhost:8000` in your browser.

### Docker Setup
```bash
docker-compose up --build
```

## 🔌 API Reference

Protected endpoints require the `X-API-Key` header.

- `GET /health` - Check API and model artifact status.
- `POST /predict` - Get risk prediction from biomarkers.
- `GET /explain/{prediction_id}` - Get SHAP explanations for a prediction.

### Example Prediction Payload
```json
{
  "cTnI_ng_mL": 14.35,
  "BNP_pg_mL": 623.42,
  "Myoglobin_ng_mL": 15.02,
  "Peak_Current_uA": 280.74,
  "I_620": 585.7,
  "I_540": 735.3,
  "R620": 1.485,
  "R540": 3.025,
  "Raw_Fluorescence_au": 2149.0,
  "DeltaF_au": 949.0,
  "Calculated_Troponin_ng_mL": 1.186
}
```

## 🧪 Testing & Quality
Run the automated test suite:
```bash
pytest tests/
```
The project uses `flake8` for linting and `pytest` for unit/integration testing, both of which are enforced via GitHub Actions.

## 📄 License
This project is licensed under the MIT License.
