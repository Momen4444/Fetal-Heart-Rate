import joblib
import numpy as np

try:
    model = joblib.load('Models/ctu_heart_failure_model.pkl')
    print("Model loaded successfully.")
    if hasattr(model, 'feature_names_in_'):
        print("Feature names:")
        for name in model.feature_names_in_:
            print(name)
    else:
        print("Model does not have feature_names_in_ attribute.")
        # Try to infer from a dummy prediction if possible, but usually this attribute exists for sklearn/xgboost
except Exception as e:
    print(f"Error: {e}")
