import os
import re
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any

# ==========================================
# Initialize FastAPI App
# ==========================================
app = FastAPI(title="Heart Monitoring API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# CONSTANTS & MAPPINGS
# ==========================================
HEA_MAPPING = {
    # Outcome measures
    "pH": "pH",
    "BDecf": "BDecf",
    "pCO2": "pCO2",
    "BE": "BE",
    "Apgar1": "Apgar1",
    "Apgar5": "Apgar5",
    # Fetus/Neonate descriptors
    "Gest. weeks": "Gest. Weeks",
    "Weight(g)": "Weight(g)",
    "Sex": "Sex",
    # Maternal risk factors
    "Age": "Age",
    "Gravidity": "Gravidity",
    "Parity": "Parity",
    "Diabetes": "Diabetes",
    "Hypertension": "Hypertension",
    "Preeclampsia": "Preeclampsia",
    "Liq. praecox": "Liq.",
    "Pyrexia": "Pyrexia",
    "Meconium": "Meconium",
    # Delivery descriptors
    "Presentation": "Presentation",
    "Induced": "Induced",
    "I.stage": "I.stage",
    "NoProgress": "NoProgress",
    "CK/KP": "CK/KP",
    "II.stage": "II.stage",
    "Deliv. type": "Deliv. type",
    # Signal information
    "dbID": "dbID",
    "Rec. type": "Rec. type",
    "Pos. II.st.": "Pos. II.st.",
    "Sig2Birth": "Sig2Birth"
}

# ==========================================
# Model Loading
# ==========================================
MODELS = {
    "adult": {"model": None},
    "fetal": {"model": None, "scaler": None, "feature_names": None}
}

def load_models():
    # Adult
    try:
        MODELS["adult"]["model"] = joblib.load('Models/adult_chf_classifier.pkl')
        print("✅ Adult Model Loaded")
    except Exception as e:
        print(f"⚠️ Adult Model NOT found: {e}")

    # Fetal
    try:
        MODELS["fetal"]["model"] = joblib.load('ctu_heart_failure_model.pkl')
        MODELS["fetal"]["scaler"] = joblib.load('scaler_ctu.pkl')
        if hasattr(MODELS["fetal"]["scaler"], "feature_names_in_"):
             MODELS["fetal"]["feature_names"] = list(MODELS["fetal"]["scaler"].feature_names_in_)
        else:
             MODELS["fetal"]["feature_names"] = list(HEA_MAPPING.values()) + ['Mean FHR', 'Std FHR', 'Mean UC']
        print(f"✅ Fetal Model Loaded")
    except Exception as e:
        print(f"⚠️ Fetal Model NOT found: {e}")

load_models()

# ==========================================
# Pydantic Models
# ==========================================
class AdultHRVInput(BaseModel):
    sdnn: float
    rmssd: float
    pnn50: float
    mean_hr: float

class FetalCTUInput(BaseModel):
    features: Dict[str, Any]

class PredictionResponse(BaseModel):
    group: str
    prediction: str
    risk_score: float
    status: str

# ==========================================
# Utilities
# ==========================================
def parse_header_content(content: str) -> Dict[str, float]:
    extracted = {}
    lines = content.splitlines()
    for line in lines:
        line = line.strip()
        if line.startswith('#'):
            parts = line[1:].strip().split()
            if len(parts) >= 2:
                raw_key = " ".join(parts[:-1]).strip()
                value_str = parts[-1]
                if raw_key in HEA_MAPPING:
                    try:
                        clean_key = HEA_MAPPING[raw_key]
                        extracted[clean_key] = float(value_str)
                    except ValueError:
                        continue 
    return extracted

# ==========================================
# Endpoints
# ==========================================
@app.post("/upload/fetal-header")
async def process_fetal_header(file: UploadFile = File(...)):
    try:
        content = await file.read()
        text_content = content.decode('utf-8', errors='ignore')
        
        extracted_data = parse_header_content(text_content)
        simulation_params = {
            'Mean FHR': extracted_data.get('Mean FHR', 140.0), 
            'Std FHR': extracted_data.get('Std FHR', 15.0)    
        }
        
        return {
            "filename": file.filename,
            "extracted_features": extracted_data,
            "simulation_params": simulation_params,
            "message": "File parsed successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/adult", response_model=PredictionResponse)
def predict_adult(data: AdultHRVInput):
    if MODELS["adult"]["model"] is None:
        raise HTTPException(status_code=503, detail="Adult Model is not loaded on the server.")

    try:
        features = pd.DataFrame([[data.sdnn, data.rmssd, data.pnn50, data.mean_hr]], 
                              columns=['sdnn', 'rmssd', 'pnn50', 'mean_hr'])
        prediction = int(MODELS["adult"]["model"].predict(features)[0])
        prob = float(MODELS["adult"]["model"].predict_proba(features)[0][1])

        return {
            "group": "Adult",
            "prediction": "Heart Failure Detected" if prediction == 1 else "Healthy",
            "risk_score": prob,
            "status": "Critical" if prob > 0.7 else "Normal"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/fetal", response_model=PredictionResponse)
def predict_fetal(data: FetalCTUInput):
    if MODELS["fetal"]["model"] is None or MODELS["fetal"]["scaler"] is None:
        raise HTTPException(status_code=503, detail="Fetal Model is not loaded on the server.")

    try:
        feature_names = MODELS["fetal"]["feature_names"]
        input_data = data.features
        
        # Create DF with 0s for all expected model features
        df = pd.DataFrame(0, index=[0], columns=feature_names)
        
        # Fill DF with available input data
        for col in feature_names:
            if col in input_data:
                df[col] = input_data[col]
        
        # Scale & Predict
        X_scaled = MODELS["fetal"]["scaler"].transform(df)
        prediction = int(MODELS["fetal"]["model"].predict(X_scaled)[0])
        
        if hasattr(MODELS["fetal"]["model"], "predict_proba"):
            prob = float(MODELS["fetal"]["model"].predict_proba(X_scaled)[0][1])
        else:
            prob = 1.0 if prediction == 1 else 0.0

        return {
            "group": "Fetal",
            "prediction": "Pathological" if prediction == 1 else "Normal",
            "risk_score": prob,
            "status": "Action Required" if prob > 0.5 else "Stable"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)