import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib

# ==========================================
# CONFIGURATION
# ==========================================
REPO_DIR = "CTU-CHB-Intrapartum-Cardiotocography-Caesarean-Section-Prediction"
DATA_FILE_CTU = os.path.join(REPO_DIR, "database", "final-interpolated.csv")

# ==========================================
# UTILS (Adapted from utils/evaluate.py)
# ==========================================
def evaluate_model(model, features, target):
    pred = model.predict(features)
    acc = accuracy_score(target, pred)
    print("Accuracy:- %.2f%%" % (acc * 100.0))
    print('Confusion matrix :- \n', confusion_matrix(target, pred))
    print("Classification Report:-\n", classification_report(target, pred))
    return acc

def load_and_process_data():
    print("--- Loading Dataset ---")
    
    # Load CTU DF
    print(f"Loading CTU data from {DATA_FILE_CTU}...")
    if not os.path.exists(DATA_FILE_CTU):
        print(f"Error: {DATA_FILE_CTU} not found.")
        return None, None

    df = pd.read_csv(DATA_FILE_CTU)
    
    # Drop identifiers and target
    drop_cols = ['ID', 'target', 'Unnamed: 0']
    # Also drop potential leakage or metadata if necessary, but user asked for "whole features"
    # We will trust the user's request.
    
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df['target']
    
    # Handle NaNs if any
    X = X.fillna(X.mean())
    
    print(f"Data Shape: {X.shape}")
    print(f"Target Distribution:\n{y.value_counts()}")
    
    return X, y

# ==========================================
# MAIN PIPELINE
# ==========================================
def main():
    print("--- Starting CTU Full Feature Training Pipeline ---")
    
    # 1. Load Data
    X, y = load_and_process_data()
    
    if X is None:
        return

    print(f"Features ({len(X.columns)}): {list(X.columns)}")
    
    # Save feature names for inference
    joblib.dump(list(X.columns), 'Models/ctu_feature_names.pkl')

    # 2. Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    # 3. Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    # Save scaler
    os.makedirs('Models', exist_ok=True)
    joblib.dump(sc, 'Models/scaler_ctu_full.pkl')

    # 4. Model Training (Random Forest)
    print("\n--- Training Random Forest ---")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    
    evaluate_model(rf, X_test, y_test)
    
    # 5. Save Model
    print("\n--- Saving Model ---")
    joblib.dump(rf, 'Models/ctu_full_model.pkl')
    print("Model saved as 'Models/ctu_full_model.pkl'")

if __name__ == "__main__":
    main()
