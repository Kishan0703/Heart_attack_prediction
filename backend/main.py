from fastapi import FastAPI, HTTPException, Security, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field, validator
import joblib
import pandas as pd
import numpy as np
import os
import logging
import uuid
from typing import Dict, List, Any, Optional
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from dotenv import load_dotenv

load_dotenv()

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("cardiac-api")

# Initialize Rate Limiter
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Cardiac Risk Prediction API", description="Predicts Heart Attack Risk based on Biomarkers")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# API Key Security
API_KEY = os.getenv("API_KEY", "your_key_here")
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    """
    Dependency to validate the X-API-Key header.
    
    Args:
        api_key_header: The value of the X-API-Key header.
        
    Returns:
        The validated API key.
        
    Raises:
        HTTPException: 403 error if the key is invalid.
    """
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(
        status_code=403,
        detail="Could not validate credentials",
    )

# Serve frontend static files
frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend')
if os.path.isdir(frontend_path):
    app.mount('/static', StaticFiles(directory=frontend_path), name='static')

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model artifacts and cache
pipeline = None
label_encoder = None
features = None
shap_explainer = None
scaler = None
prediction_cache: Dict[str, pd.DataFrame] = {}

@app.on_event("startup")
def load_model():
    """
    FastAPI startup event to load ML model artifacts from the model/ directory.
    """
    global pipeline, label_encoder, features, shap_explainer, scaler
    try:
        pipeline = joblib.load("model/pipeline.pkl")
        label_encoder = joblib.load("model/label_encoder.pkl")
        features = joblib.load("model/features.pkl")
        scaler = joblib.load("model/scaler.pkl")
        if os.path.exists("model/shap_explainer.pkl"):
            shap_explainer = joblib.load("model/shap_explainer.pkl")
        logger.info("Model artifacts loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")

class BiomarkerInput(BaseModel):
    """
    Pydantic schema for biomarker input data with validation ranges.
    """
    I_620: float = Field(..., ge=0, le=10000)
    I_540: float = Field(..., ge=0, le=10000)
    R620: float = Field(..., ge=0, le=100)
    R540: float = Field(..., ge=0, le=100)
    cTnI_ng_mL: float = Field(..., ge=0, le=500)
    Myoglobin_ng_mL: float = Field(..., ge=0, le=1000)
    Raw_Fluorescence_au: float = Field(..., ge=0, le=20000)
    DeltaF_au: float = Field(..., ge=0, le=20000)
    Calculated_Troponin_ng_mL: float = Field(..., ge=0, le=500)
    Peak_Current_uA: float = Field(..., ge=0, le=5000)
    BNP_pg_mL: float = Field(..., ge=0, le=10000)

    class Config:
        json_schema_extra = {
            "example": {
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
        }

@app.get("/health")
def health():
    """
    Endpoint to check the health of the API and verify that model artifacts are loaded.
    
    Returns:
        JSON with health status and artifact availability.
        
    Raises:
        HTTPException: 503 error if any critical artifact is missing.
    """
    status = {
        "pipeline": pipeline is not None,
        "label_encoder": label_encoder is not None,
        "features": features is not None,
        "shap_explainer": shap_explainer is not None
    }
    if all(status.values()):
        return {"status": "healthy", "artifacts": status}
    raise HTTPException(status_code=503, detail={"status": "unhealthy", "artifacts": status})

@app.get("/")
def root():
    """
    Root endpoint serving the frontend dashboard or a simple health message.
    """
    index_file = os.path.join(frontend_path, 'index.html')
    if os.path.isfile(index_file):
        return FileResponse(index_file)
    return {"message": "Cardiac Risk Prediction API is running"}

@app.post("/predict")
@limiter.limit("100/minute")
def predict(request: Request, data: BiomarkerInput, api_key: str = Depends(get_api_key)):
    """
    Endpoint to predict heart attack risk based on input biomarkers.
    
    Args:
        request: The FastAPI request object (required for rate limiting).
        data: BiomarkerInput containing the values for prediction.
        api_key: Validated API key.
        
    Returns:
        Prediction results including risk category, probabilities, and interpretation.
        
    Raises:
        HTTPException: 500 error if model is not loaded or prediction fails.
    """
    if not pipeline or not label_encoder:
        logger.error("Predict called but model not loaded")
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame with correct column names
        input_data = data.dict()
        input_data['Raw_Fluorescence_a.u.'] = input_data.pop('Raw_Fluorescence_au')
        input_data['DeltaF_a.u.'] = input_data.pop('DeltaF_au')
        
        df_input = pd.DataFrame([input_data])
        df_input = df_input[features]
        
        # Predict
        prediction_idx = pipeline.predict(df_input)[0]
        prediction_label = label_encoder.inverse_transform([prediction_idx])[0]
        probs = pipeline.predict_proba(df_input)[0]
        
        try:
            clf_classes = pipeline.named_steps['classifier'].classes_
        except Exception:
            clf_classes = list(range(len(probs)))

        class_probs = {}
        for idx, cls in enumerate(clf_classes):
            label = label_encoder.inverse_transform([int(cls)])[0]
            class_probs[label] = float(probs[idx])
        
        prediction_id = str(uuid.uuid4())
        prediction_cache[prediction_id] = df_input
        
        # Limit cache size
        if len(prediction_cache) > 100:
            prediction_cache.pop(next(iter(prediction_cache)))

        logger.info(f"Prediction made: ID={prediction_id}, Category={prediction_label}, Confidence={float(np.max(probs)):.4f}")

        interpretation = []
        if data.cTnI_ng_mL > 50:
            interpretation.append("High Troponin I levels detected.")
        if data.BNP_pg_mL > 400:
            interpretation.append("Elevated BNP levels detected.")
        if data.Myoglobin_ng_mL > 50:
            interpretation.append("High Myoglobin levels detected.")
        if not interpretation:
            interpretation.append("Biomarker levels are within typical ranges.")

        return {
            "prediction_id": prediction_id,
            "risk_category": prediction_label,
            "probability": class_probs,
            "confidence": float(np.max(probs)),
            "interpretation": " ".join(interpretation)
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/explain/{prediction_id}")
@limiter.limit("100/minute")
def explain(request: Request, prediction_id: str, api_key: str = Depends(get_api_key)):
    """
    Endpoint to return SHAP explanations for a previous prediction.
    
    Args:
        request: The FastAPI request object.
        prediction_id: The UUID of the prediction to explain.
        api_key: Validated API key.
        
    Returns:
        Top 5 features contributing to the prediction based on SHAP values.
        
    Raises:
        HTTPException: 404 if ID not found, 501 if explainer unavailable, 500 on error.
    """
    if not shap_explainer:
        raise HTTPException(status_code=501, detail="SHAP explainer not available")
    
    if prediction_id not in prediction_cache:
        raise HTTPException(status_code=404, detail="Prediction ID not found or expired")
    
    try:
        df_input = prediction_cache[prediction_id]
        X_transformed = scaler.transform(df_input)
        
        shap_values = shap_explainer.shap_values(X_transformed)
        
        prediction_idx = pipeline.predict(df_input)[0]
        
        if isinstance(shap_values, list):
            sv = shap_values[prediction_idx][0]
        else:
            if len(shap_values.shape) == 3:
                sv = shap_values[0, :, prediction_idx]
            else:
                sv = shap_values[0]

        feature_importance = []
        for i, feat in enumerate(features):
            feature_importance.append({
                "feature": feat,
                "shap_value": float(sv[i])
            })
            
        feature_importance.sort(key=lambda x: abs(x['shap_value']), reverse=True)
        
        return {
            "prediction_id": prediction_id,
            "top_features": feature_importance[:5]
        }
    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
