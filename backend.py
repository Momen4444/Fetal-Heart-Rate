"""
Heart Monitoring Backend API
============================
FastAPI backend for Adult CHF and Fetal Heart Rate prediction.
Inference pipelines match the training scripts exactly.
"""

import os
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any

# ==========================================
# Initialize FastAPI App
# ==========================================
app = FastAPI(
    title="Heart Monitoring API",
    description="API for Adult Heart Failure and Fetal Heart Rate prediction",
    version="1.0.0"
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your React app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# Model Storage
# ==========================================
MODELS = {
    "adult": {
        "model": None,
        "feature_names": ['sdnn', 'rmssd', 'pnn50', 'mean_hr']  # From AdultCHFModel.py
    },
    "fetal": {
        "model": None,
        "scaler": None,
        "feature_names": None  # Will be loaded from ctu_feature_names.pkl
    }
}

# ==========================================
# Model Loading
# ==========================================
def load_models():
    """Load all trained models and scalers."""
    
    # --- Adult CHF Model ---
    # From AdultCHFModel.py: saved to 'Models/adult_chf_classifier.pkl'
    adult_model_path = 'Models/adult_chf_classifier.pkl'
    try:
        if os.path.exists(adult_model_path):
            MODELS["adult"]["model"] = joblib.load(adult_model_path)
            print(f"✅ Adult CHF Model loaded from {adult_model_path}")
        else:
            print(f"⚠️ Adult model not found at {adult_model_path}. Using mock mode.")
    except Exception as e:
        print(f"❌ Error loading Adult model: {e}")

    # --- Fetal (CTU) Model ---
    # From HeartFailurePrediction_CTU.py:
    #   - Model: 'Models/ctu_full_model.pkl'
    #   - Scaler: 'Models/scaler_ctu_full.pkl'
    #   - Feature names: 'Models/ctu_feature_names.pkl'
    fetal_model_path = 'Models/ctu_full_model.pkl'
    fetal_scaler_path = 'Models/scaler_ctu_full.pkl'
    fetal_features_path = 'Models/ctu_feature_names.pkl'
    
    try:
        # Load feature names first
        if os.path.exists(fetal_features_path):
            MODELS["fetal"]["feature_names"] = joblib.load(fetal_features_path)
            print(f"✅ Fetal feature names loaded: {len(MODELS['fetal']['feature_names'])} features")
        
        # Load scaler
        if os.path.exists(fetal_scaler_path):
            MODELS["fetal"]["scaler"] = joblib.load(fetal_scaler_path)
            print(f"✅ Fetal scaler loaded from {fetal_scaler_path}")
        
        # Load model
        if os.path.exists(fetal_model_path):
            MODELS["fetal"]["model"] = joblib.load(fetal_model_path)
            print(f"✅ Fetal CTU Model loaded from {fetal_model_path}")
        else:
            print(f"⚠️ Fetal model not found at {fetal_model_path}. Using mock mode.")
            
    except Exception as e:
        print(f"❌ Error loading Fetal model: {e}")

# Load models on startup
load_models()

# ==========================================
# Pydantic Models (Request/Response Schemas)
# ==========================================

# --- Adult Input ---
# Matches AdultCHFModel.py HRV features: sdnn, rmssd, pnn50, mean_hr
class AdultHRVInput(BaseModel):
    """
    Adult HRV (Heart Rate Variability) features.
    These are extracted from ECG signals using R-peak detection.
    
    Reference values:
    - Healthy: SDNN > 100ms, RMSSD > 40ms, pNN50 > 10%
    - Heart Failure: SDNN < 50ms, RMSSD < 20ms, pNN50 < 3%
    """
    sdnn: float = Field(..., description="Standard Deviation of NN intervals (ms)", ge=0)
    rmssd: float = Field(..., description="Root Mean Square of Successive Differences (ms)", ge=0)
    pnn50: float = Field(..., description="Percentage of successive intervals > 50ms (%)", ge=0, le=100)
    mean_hr: float = Field(..., description="Mean Heart Rate (bpm)", ge=20, le=250)

# --- Fetal Input ---
# Matches HeartFailurePrediction_CTU.py features from final-interpolated.csv
class FetalCTUInput(BaseModel):
    """
    Fetal CTU (Cardiotocography) features.
    These match the CTU-CHB database features used in training.
    """
    # Blood gas analysis (using dataset means as defaults)
    pH: float = Field(default=7.24, description="Umbilical artery pH", ge=6.8, le=7.6)
    BDecf: float = Field(default=4.33, description="Base Deficit in extracellular fluid")
    pCO2: float = Field(default=7.07, description="Partial pressure of CO2 (kPa)")
    BE: float = Field(default=-6.14, description="Base Excess")
    
    # Apgar scores
    Apgar1: float = Field(default=8.42, description="Apgar score at 1 minute", ge=0, le=10)
    Apgar5: float = Field(default=9.20, description="Apgar score at 5 minutes", ge=0, le=10)
    
    # Pregnancy info
    gest_weeks: float = Field(default=39.98, alias="Gest. Weeks", description="Gestational weeks", ge=20, le=45)
    weight_g: float = Field(default=3422.48, alias="Weight(g)", description="Birth weight (grams)", ge=500, le=6000)
    Sex: float = Field(default=1.47, description="Sex (1=Male, 2=Female)")
    Age: float = Field(default=29.49, description="Maternal age", ge=15, le=55)
    Gravidity: float = Field(default=1.43, description="Number of pregnancies", ge=1)
    Parity: float = Field(default=0.47, description="Number of previous deliveries", ge=0)
    
    # Medical conditions (0=No, 1=Yes)
    Diabetes: float = Field(default=0.10, description="Diabetes (0=No, 1=Yes)")
    Hypertension: float = Field(default=0.07, description="Hypertension (0=No, 1=Yes)")
    Preeclampsia: float = Field(default=0.02, description="Preeclampsia (0=No, 1=Yes)")
    
    # Labor info
    Liq_: float = Field(default=0.25, alias="Liq.", description="Liquor (amniotic fluid)")
    Pyrexia: float = Field(default=0.01, description="Pyrexia/Fever (0=No, 1=Yes)")
    Meconium: float = Field(default=0.10, description="Meconium staining (0=No, 1=Yes)")
    Presentation: float = Field(default=1.16, description="Fetal presentation")
    Induced: float = Field(default=0.43, description="Induced labor (0=No, 1=Yes)")
    
    # Labor stages
    i_stage: float = Field(default=218.58, alias="I.stage", description="Duration of first stage (minutes)")
    NoProgress: float = Field(default=0.09, description="No progress indicator")
    CK_KP: float = Field(default=0.02, alias="CK/KP", description="CK/KP indicator")
    ii_stage: float = Field(default=11.93, alias="II.stage", description="Duration of second stage (minutes)")
    
    # Delivery info
    deliv_type: float = Field(default=1.06, alias="Deliv. type", description="Delivery type")
    dbID: float = Field(default=1055334.0, description="Database ID")
    rec_type: float = Field(default=1.79, alias="Rec. type", description="Recording type")
    pos_ii_st: float = Field(default=13523.98, alias="Pos. II.st.", description="Position in second stage")
    Sig2Birth: float = Field(default=0.0, description="Signal to birth time")
    
    # FHR (Fetal Heart Rate) signal features
    Mean_FHR: float = Field(default=105.22, description="Mean Fetal Heart Rate (bpm)", ge=60, le=200)
    Mean_UC: float = Field(default=18.24, description="Mean Uterine Contractions")
    Median_FHR: float = Field(default=128.99, description="Median Fetal Heart Rate (bpm)")
    Median_UC: float = Field(default=12.97, description="Median Uterine Contractions")
    Std_FHR: float = Field(default=55.44, description="Std Dev of Fetal Heart Rate")
    Std_UC: float = Field(default=18.80, description="Std Dev of Uterine Contractions")
    RMS_FHR: float = Field(default=120.74, description="RMS of Fetal Heart Rate")
    RMS_UC: float = Field(default=26.53, description="RMS of Uterine Contractions")
    Peak_to_RMS_FHR: float = Field(default=70.87, description="Peak to RMS ratio of FHR")
    Peak_to_RMS_UC: float = Field(default=72.73, description="Peak to RMS ratio of UC")
    Peak_FHR: float = Field(default=191.61, description="Peak Fetal Heart Rate")
    Peak_UC: float = Field(default=99.26, description="Peak Uterine Contractions")

    model_config = ConfigDict(populate_by_name=True)

# --- Simplified Fetal Input (for UI with fewer fields) ---
class FetalSimpleInput(BaseModel):
    """
    Simplified fetal input for basic UI.
    Missing values will be filled with dataset means.
    """
    Mean_FHR: float = Field(default=105.22, description="Mean Fetal Heart Rate (bpm)")
    Std_FHR: float = Field(default=55.44, description="Variability (Std Dev of FHR)")
    Mean_UC: float = Field(default=18.24, description="Mean Uterine Contractions")
    pH: Optional[float] = Field(default=None, description="Umbilical artery pH (if available)")
    gest_weeks: Optional[float] = Field(default=None, description="Gestational weeks")

# --- Response Models ---
class PredictionResponse(BaseModel):
    group: str
    prediction: str
    risk_score: float
    status: str
    details: Optional[Dict[str, Any]] = None

# ==========================================
# Utility Functions
# ==========================================

def get_fetal_default_values() -> Dict[str, float]:
    """
    Returns default/mean values for CTU features.
    These are the actual population means from the CTU-CHB database.
    """
    return {
        'pH': 7.24,
        'BDecf': 4.33,
        'pCO2': 7.07,
        'BE': -6.14,
        'Apgar1': 8.42,
        'Apgar5': 9.20,
        'Gest. Weeks': 39.98,
        'Weight(g)': 3422.48,
        'Sex': 1.47,
        'Age': 29.49,
        'Gravidity': 1.43,
        'Parity': 0.47,
        'Diabetes': 0.10,
        'Hypertension': 0.07,
        'Preeclampsia': 0.02,
        'Liq.': 0.25,
        'Pyrexia': 0.01,
        'Meconium': 0.10,
        'Presentation': 1.16,
        'Induced': 0.43,
        'I.stage': 218.58,
        'NoProgress': 0.09,
        'CK/KP': 0.02,
        'II.stage': 11.93,
        'Deliv. type': 1.06,
        'dbID': 1055334.0,
        'Rec. type': 1.79,
        'Pos. II.st.': 13523.98,
        'Sig2Birth': 0.0,
        'Mean_FHR': 105.22,
        'Mean_UC': 18.24,
        'Median_FHR': 128.99,
        'Median_UC': 12.97,
        'Std_FHR': 55.44,
        'Std_UC': 18.80,
        'RMS_FHR': 120.74,
        'RMS_UC': 26.53,
        'Peak_to_RMS_FHR': 70.87,
        'Peak_to_RMS_UC': 72.73,
        'Peak_FHR': 191.61,
        'Peak_UC': 99.26
    }

# ==========================================
# API Endpoints
# ==========================================

@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "models": {
            "adult": "loaded" if MODELS["adult"]["model"] is not None else "mock",
            "fetal": "loaded" if MODELS["fetal"]["model"] is not None else "mock"
        }
    }

@app.get("/features/adult")
def get_adult_features():
    """Get the expected features for adult prediction."""
    return {
        "features": MODELS["adult"]["feature_names"],
        "description": "HRV features extracted from ECG"
    }

@app.get("/features/fetal")
def get_fetal_features():
    """Get the expected features for fetal prediction."""
    return {
        "features": MODELS["fetal"]["feature_names"],
        "description": "CTU-CHB cardiotocography features"
    }

@app.post("/predict/adult", response_model=PredictionResponse)
def predict_adult(data: AdultHRVInput):
    """
    Predict Adult Heart Failure from HRV features.
    
    Inference matches AdultCHFModel.py:
    - Input: DataFrame with columns ['sdnn', 'rmssd', 'pnn50', 'mean_hr']
    - No scaling required (model handles it internally if needed)
    """
    try:
        # Create DataFrame matching training format
        feature_names = MODELS["adult"]["feature_names"]
        features = pd.DataFrame([[
            data.sdnn,
            data.rmssd,
            data.pnn50,
            data.mean_hr
        ]], columns=feature_names)
        
        if MODELS["adult"]["model"] is not None:
            # Real model prediction
            model = MODELS["adult"]["model"]
            prediction = int(model.predict(features)[0])
            
            # Get probability if available
            if hasattr(model, "predict_proba"):
                probability = float(model.predict_proba(features)[0][1])
            else:
                probability = 1.0 if prediction == 1 else 0.0
        else:
            # Mock prediction based on HRV thresholds
            # Low HRV indicates heart failure risk
            score = 0
            if data.sdnn < 50: score += 1
            if data.rmssd < 20: score += 1
            if data.pnn50 < 3: score += 1
            if data.mean_hr > 100 or data.mean_hr < 50: score += 1
            
            prediction = 1 if score >= 2 else 0
            probability = min(0.95, score * 0.25) if prediction == 1 else max(0.05, 0.3 - score * 0.1)
        
        # Determine status based on risk score
        if probability > 0.7:
            status = "Critical - Immediate attention required"
        elif probability > 0.5:
            status = "Warning - Further evaluation recommended"
        else:
            status = "Normal"
        
        return PredictionResponse(
            group="Adult",
            prediction="Heart Failure Detected" if prediction == 1 else "Healthy",
            risk_score=round(probability, 4),
            status=status,
            details={
                "input_features": {
                    "sdnn": data.sdnn,
                    "rmssd": data.rmssd,
                    "pnn50": data.pnn50,
                    "mean_hr": data.mean_hr
                },
                "model_type": "real" if MODELS["adult"]["model"] else "mock"
            }
        )
        
    except Exception as e:
        print(f"Adult Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/fetal", response_model=PredictionResponse)
def predict_fetal(data: FetalCTUInput):
    """
    Predict Fetal outcome from CTU features.
    
    Inference matches HeartFailurePrediction_CTU.py:
    - Input: DataFrame with columns matching ctu_feature_names.pkl
    - Apply StandardScaler transform (scaler_ctu_full.pkl)
    - Predict with RandomForest model (ctu_full_model.pkl)
    """
    try:
        feature_names = MODELS["fetal"]["feature_names"]
        
        if feature_names is None:
            # Fallback feature names
            feature_names = list(get_fetal_default_values().keys())
        
        # Build input dictionary mapping Pydantic fields to actual feature names
        input_dict = {
            'pH': data.pH,
            'BDecf': data.BDecf,
            'pCO2': data.pCO2,
            'BE': data.BE,
            'Apgar1': data.Apgar1,
            'Apgar5': data.Apgar5,
            'Gest. Weeks': data.gest_weeks,
            'Weight(g)': data.weight_g,
            'Sex': data.Sex,
            'Age': data.Age,
            'Gravidity': data.Gravidity,
            'Parity': data.Parity,
            'Diabetes': data.Diabetes,
            'Hypertension': data.Hypertension,
            'Preeclampsia': data.Preeclampsia,
            'Liq.': data.Liq_,
            'Pyrexia': data.Pyrexia,
            'Meconium': data.Meconium,
            'Presentation': data.Presentation,
            'Induced': data.Induced,
            'I.stage': data.i_stage,
            'NoProgress': data.NoProgress,
            'CK/KP': data.CK_KP,
            'II.stage': data.ii_stage,
            'Deliv. type': data.deliv_type,
            'dbID': data.dbID,
            'Rec. type': data.rec_type,
            'Pos. II.st.': data.pos_ii_st,
            'Sig2Birth': data.Sig2Birth,
            'Mean_FHR': data.Mean_FHR,
            'Mean_UC': data.Mean_UC,
            'Median_FHR': data.Median_FHR,
            'Median_UC': data.Median_UC,
            'Std_FHR': data.Std_FHR,
            'Std_UC': data.Std_UC,
            'RMS_FHR': data.RMS_FHR,
            'RMS_UC': data.RMS_UC,
            'Peak_to_RMS_FHR': data.Peak_to_RMS_FHR,
            'Peak_to_RMS_UC': data.Peak_to_RMS_UC,
            'Peak_FHR': data.Peak_FHR,
            'Peak_UC': data.Peak_UC
        }
        
        # Create DataFrame with correct column order
        features = pd.DataFrame([[input_dict.get(col, 0.0) for col in feature_names]], 
                               columns=feature_names)
        
        if MODELS["fetal"]["model"] is not None and MODELS["fetal"]["scaler"] is not None:
            # Apply scaling (matching training pipeline)
            features_scaled = MODELS["fetal"]["scaler"].transform(features)
            
            # Predict
            model = MODELS["fetal"]["model"]
            prediction = int(model.predict(features_scaled)[0])
            
            # Get probability
            if hasattr(model, "predict_proba"):
                probability = float(model.predict_proba(features_scaled)[0][1])
            else:
                probability = 1.0 if prediction == 1 else 0.0
        else:
            # Mock prediction based on key indicators
            risk_factors = 0
            if data.pH < 7.15: risk_factors += 2
            if data.Apgar1 < 7: risk_factors += 1
            if data.Apgar5 < 7: risk_factors += 2
            if data.Mean_FHR < 110 or data.Mean_FHR > 160: risk_factors += 1
            if data.Std_FHR > 25: risk_factors += 1
            if data.Meconium > 0: risk_factors += 1
            
            prediction = 1 if risk_factors >= 3 else 0
            probability = min(0.95, risk_factors * 0.15) if prediction == 1 else max(0.05, 0.4 - risk_factors * 0.1)
        
        # Determine status
        if probability > 0.6:
            status = "Pathological - C-section may be required"
        elif probability > 0.4:
            status = "Suspicious - Close monitoring required"
        else:
            status = "Normal - Continue monitoring"
        
        return PredictionResponse(
            group="Fetal",
            prediction="Pathological" if prediction == 1 else "Normal",
            risk_score=round(probability, 4),
            status=status,
            details={
                "key_indicators": {
                    "Mean_FHR": data.Mean_FHR,
                    "Std_FHR": data.Std_FHR,
                    "pH": data.pH,
                    "Apgar1": data.Apgar1,
                    "Apgar5": data.Apgar5
                },
                "model_type": "real" if MODELS["fetal"]["model"] else "mock"
            }
        )
        
    except Exception as e:
        print(f"Fetal Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/fetal/simple", response_model=PredictionResponse)
def predict_fetal_simple(data: FetalSimpleInput):
    """
    Simplified fetal prediction endpoint.
    Uses default values for missing features.
    Good for basic UI with limited input fields.
    """
    try:
        # Get default values
        defaults = get_fetal_default_values()
        
        # Override with provided values
        defaults['Mean_FHR'] = data.Mean_FHR
        defaults['Median_FHR'] = data.Mean_FHR  # Use same value
        defaults['Std_FHR'] = data.Std_FHR
        defaults['Mean_UC'] = data.Mean_UC
        defaults['Median_UC'] = data.Mean_UC
        
        if data.pH is not None:
            defaults['pH'] = data.pH
        if data.gest_weeks is not None:
            defaults['Gest. Weeks'] = data.gest_weeks
        
        # Create full input for the main endpoint
        full_input = FetalCTUInput(
            pH=defaults['pH'],
            BDecf=defaults['BDecf'],
            pCO2=defaults['pCO2'],
            BE=defaults['BE'],
            Apgar1=defaults['Apgar1'],
            Apgar5=defaults['Apgar5'],
            gest_weeks=defaults['Gest. Weeks'],
            weight_g=defaults['Weight(g)'],
            Sex=defaults['Sex'],
            Age=defaults['Age'],
            Gravidity=defaults['Gravidity'],
            Parity=defaults['Parity'],
            Diabetes=defaults['Diabetes'],
            Hypertension=defaults['Hypertension'],
            Preeclampsia=defaults['Preeclampsia'],
            Liq_=defaults['Liq.'],
            Pyrexia=defaults['Pyrexia'],
            Meconium=defaults['Meconium'],
            Presentation=defaults['Presentation'],
            Induced=defaults['Induced'],
            i_stage=defaults['I.stage'],
            NoProgress=defaults['NoProgress'],
            CK_KP=defaults['CK/KP'],
            ii_stage=defaults['II.stage'],
            deliv_type=defaults['Deliv. type'],
            dbID=defaults['dbID'],
            rec_type=defaults['Rec. type'],
            pos_ii_st=defaults['Pos. II.st.'],
            Sig2Birth=defaults['Sig2Birth'],
            Mean_FHR=defaults['Mean_FHR'],
            Mean_UC=defaults['Mean_UC'],
            Median_FHR=defaults['Median_FHR'],
            Median_UC=defaults['Median_UC'],
            Std_FHR=defaults['Std_FHR'],
            Std_UC=defaults['Std_UC'],
            RMS_FHR=defaults['RMS_FHR'],
            RMS_UC=defaults['RMS_UC'],
            Peak_to_RMS_FHR=defaults['Peak_to_RMS_FHR'],
            Peak_to_RMS_UC=defaults['Peak_to_RMS_UC'],
            Peak_FHR=defaults['Peak_FHR'],
            Peak_UC=defaults['Peak_UC']
        )
        
        # Call main prediction
        return predict_fetal(full_input)
        
    except Exception as e:
        print(f"Fetal Simple Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# ==========================================
# Run Server
# ==========================================
if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print("Starting Heart Monitoring API Server")
    print("="*50)
    print(f"Adult Model: {'Loaded' if MODELS['adult']['model'] else 'Mock Mode'}")
    print(f"Fetal Model: {'Loaded' if MODELS['fetal']['model'] else 'Mock Mode'}")
    print("="*50 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
