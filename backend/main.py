from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

# Initialize App
app = FastAPI(title="Cardiac Risk Prediction API", description="Predicts Heart Attack Risk based on Biomarkers")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model artifacts
pipeline = None
label_encoder = None
features = None

@app.on_event("startup")
def load_model():
    global pipeline, label_encoder, features
    try:
        pipeline = joblib.load("model/pipeline.pkl")
        label_encoder = joblib.load("model/label_encoder.pkl")
        features = joblib.load("model/features.pkl")
        print("Model artifacts loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        # In production, we might want to crash if model fails to load
        pass

# Define Input Schema
class BiomarkerInput(BaseModel):
    I_620: float
    I_540: float
    R620: float
    R540: float
    cTnI_ng_mL: float
    Myoglobin_ng_mL: float
    Raw_Fluorescence_au: float  # Changed a.u. to au for valid python identifier, will map back
    DeltaF_au: float
    Calculated_Troponin_ng_mL: float
    Peak_Current_uA: float
    BNP_pg_mL: float

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

@app.get("/")
def root():
    return {"message": "Cardiac Risk Prediction API is running"}

@app.post("/predict")
def predict(data: BiomarkerInput):
    if not pipeline or not label_encoder:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame with correct column names
        input_data = data.dict()
        
        # Map 'au' fields back to 'a.u.' keys
        input_data['Raw_Fluorescence_a.u.'] = input_data.pop('Raw_Fluorescence_au')
        input_data['DeltaF_a.u.'] = input_data.pop('DeltaF_au')
        
        # Ensure correct order of features
        df_input = pd.DataFrame([input_data])
        df_input = df_input[features]
        
        # Predict
        prediction_idx = pipeline.predict(df_input)[0]
        prediction_label = label_encoder.inverse_transform([prediction_idx])[0]
        
        # Get Probabilities
        probs = pipeline.predict_proba(df_input)[0]
        # Map probabilities to class labels using the classifier's classes_ ordering
        try:
            clf_classes = pipeline.named_steps['classifier'].classes_
        except Exception:
            # Fallback: assume classes are 0..n-1
            clf_classes = list(range(len(probs)))

        class_probs = {}
        for idx, cls in enumerate(clf_classes):
            # cls is the encoded class value used by the classifier (e.g., 0,1,2)
            label = label_encoder.inverse_transform([int(cls)])[0]
            class_probs[label] = float(probs[idx])
        
        # Interpretation logic (simplified rule-based explanation alongside ML)
        interpretation = []
        if data.cTnI_ng_mL > 50: # Example threshold
            interpretation.append("High Troponin I levels detected (Strong MI Indicator).")
        if data.BNP_pg_mL > 400:
            interpretation.append("Elevated BNP levels suggest potential heart failure.")
        if data.Myoglobin_ng_mL > 50:
            interpretation.append("High Myoglobin indicates muscle/cardiac damage.")
            
        if not interpretation:
            interpretation.append("Biomarker levels are within typical ranges for this risk category.")

        return {
            "risk_category": prediction_label,
            "probability": class_probs,
            "confidence": float(np.max(probs)),
            "interpretation": " ".join(interpretation)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

