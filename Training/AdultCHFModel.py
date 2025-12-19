import os
import glob
import numpy as np
import pandas as pd
import wfdb
from wfdb import processing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import joblib

# ==========================================
# CONFIGURATION
# ==========================================
# Path to your local BIDMC Congestive Heart Failure Database
# It should contain files like chf01.dat, chf01.hea
BIDMC_DIR = "./BIDMC Congestive Heart Failure"

# Path where 'Normal' healthy control data should be located
NORMAL_DIR = "./data/nsrdb_control/"

class ECG_Processor:
    """
    Handles raw ECG processing: R-peak detection and HRV feature extraction.
    """
    
    @staticmethod
    def get_hrv_features(record_path):
        try:
            # 1. Load the Record
            # record_path should be the base path (without extension)
            record = wfdb.rdrecord(record_path)
            
            # Select the first channel (usually the best ECG lead)
            sig = record.p_signal[:, 0]
            fs = record.fs
            
            # 2. Detect R-Peaks (Heartbeats)
            # We use the XQRS algorithm which is robust for ECG
            # This can take time for long records, so we'll limit to first 15 minutes for demo speed
            limit_samples = min(len(sig), 15 * 60 * fs) 
            sig_segment = sig[:limit_samples]
            
            # verbose=False suppresses the progress bar
            qrs_inds = processing.xqrs_detect(sig=sig_segment, fs=fs, verbose=False)
            
            if len(qrs_inds) < 10:
                print(f"  [Warn] Too few peaks detected in {record_path}")
                return None

            # 3. Calculate RR Intervals (Time between beats in milliseconds)
            # qrs_inds is in samples. Convert to ms: (diff / fs) * 1000
            rr_intervals = np.diff(qrs_inds) / fs * 1000
            
            # Filter Artifacts (Physiological limits: 300ms to 2000ms)
            rr_intervals = rr_intervals[(rr_intervals > 300) & (rr_intervals < 2000)]
            
            if len(rr_intervals) < 10:
                return None

            # 4. Extract HRV Features (Time-Domain)
            # SDNN: Standard Deviation of NN intervals (Gold standard for overall variability)
            sdnn = np.std(rr_intervals)
            
            # RMSSD: Root Mean Square of Successive Differences (Vagal tone/Parasympathetic)
            diff_rr = np.diff(rr_intervals)
            rmssd = np.sqrt(np.mean(diff_rr ** 2))
            
            # pNN50: Percentage of successive intervals differing by > 50ms
            nn50 = np.sum(np.abs(diff_rr) > 50)
            pnn50 = (nn50 / len(diff_rr)) * 100
            
            # Mean Heart Rate
            mean_hr = 60000 / np.mean(rr_intervals)
            
            return {
                'sdnn': sdnn,
                'rmssd': rmssd,
                'pnn50': pnn50,
                'mean_hr': mean_hr,
                'record': os.path.basename(record_path)
            }
            
        except Exception as e:
            print(f"  [Error] Processing {record_path}: {e}")
            return None

def main():
    print("--- Starting Adult Heart Failure (CHF) Training Pipeline ---")
    
    # Locating BIDMC (Sick) files
    chf_files = glob.glob(os.path.join(BIDMC_DIR, "*.hea"))
    # Locating NSRDB (Healthy) files
    nsr_files = glob.glob(os.path.join(NORMAL_DIR, "*.hea"))
    
    # Check for BIDMC files
    if not chf_files:
        print(f"ERROR: No BIDMC files found in '{BIDMC_DIR}'")
        print("Please ensure your local BIDMC dataset is in the correct folder.")
        print(f"Current Working Directory: {os.getcwd()}")
        return

    # Check for Control files (Since auto-download is removed)
    if not nsr_files:
        print(f"ERROR: No NSRDB control files found in '{NORMAL_DIR}'")
        print("Please manually download the control subjects to this folder.")
        return

    print(f"\nFound {len(chf_files)} CHF (Sick) records.")
    print(f"Found {len(nsr_files)} NSR (Healthy) records.")
    
    features_list = []
    labels = [] # 1 = Heart Failure, 0 = Healthy
    
    processor = ECG_Processor()
    
    # 2. Process CHF (Sick) Group
    print("\nProcessing CHF (Heart Failure) Data...")
    for file_path in chf_files:
        # print(f"  Analyzing {os.path.basename(file_path)}...")
        base_path = os.path.splitext(file_path)[0]
        feats = processor.get_hrv_features(base_path)
        if feats:
            features_list.append(feats)
            labels.append(1) # Class 1: Sick
            
    # 3. Process NSR (Healthy) Group
    print("\nProcessing NSR (Healthy Control) Data...")
    for file_path in nsr_files:
        # print(f"  Analyzing {os.path.basename(file_path)}...")
        base_path = os.path.splitext(file_path)[0]
        feats = processor.get_hrv_features(base_path)
        if feats:
            features_list.append(feats)
            labels.append(0) # Class 0: Healthy

    # 4. Create DataFrame
    if not features_list:
        print("No valid features extracted. Exiting.")
        return

    df = pd.DataFrame(features_list)
    # Drop the 'record' name for training
    X = df.drop(columns=['record'])
    y = np.array(labels)
    
    print("\nData Summary:")
    print(df.groupby(pd.Series(labels, name='Class')).mean(numeric_only=True))
    print("\n(Note: Lower SDNN/RMSSD is typical of Heart Failure)")

    # 5. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 6. Define Models and Param Grids
    
    # Pipeline for SVM (needs scaling)
    svm_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True))
    ])
    
    models_params = {
        'RandomForest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            }
        },
        'XGBoost': {
            'model': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        },
        'SVM': {
            'model': svm_pipe,
            'params': {
                'svm__C': [0.1, 1, 10],
                'svm__kernel': ['rbf', 'linear'],
                'svm__gamma': ['scale', 'auto']
            }
        }
    }
    
    best_estimators = {}
    results = []
    
    print("\n--- Hyperparameter Tuning ---")
    for name, mp in models_params.items():
        print(f"Tuning {name}...")
        clf = GridSearchCV(mp['model'], mp['params'], cv=5, scoring='accuracy', n_jobs=-1)
        clf.fit(X_train, y_train)
        
        print(f"  Best Params: {clf.best_params_}")
        print(f"  Best CV Score: {clf.best_score_:.4f}")
        
        best_estimators[name] = clf.best_estimator_
        
        # Evaluate on Test Set
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results.append({'Model': name, 'Accuracy': acc, 'Best_Model': clf.best_estimator_})
        print(f"  Test Accuracy: {acc:.4f}")

    # 7. Ensemble (Voting Classifier)
    print("\nTraining Ensemble (Voting Classifier)...")
    voting_clf = VotingClassifier(
        estimators=[
            ('rf', best_estimators['RandomForest']),
            ('xgb', best_estimators['XGBoost']),
            ('svm', best_estimators['SVM'])
        ],
        voting='soft'
    )
    voting_clf.fit(X_train, y_train)
    y_pred_ens = voting_clf.predict(X_test)
    acc_ens = accuracy_score(y_test, y_pred_ens)
    results.append({'Model': 'Ensemble', 'Accuracy': acc_ens, 'Best_Model': voting_clf})
    print(f"  Ensemble Test Accuracy: {acc_ens:.4f}")
    
    # 8. Select Best Model
    best_result = max(results, key=lambda x: x['Accuracy'])
    print(f"\n--- Best Model: {best_result['Model']} ({best_result['Accuracy']:.4f}) ---")
    
    print("\nClassification Report (Best Model):")
    print(classification_report(y_test, best_result['Best_Model'].predict(X_test), target_names=['Healthy', 'Heart Failure']))
    
    # 9. Save Model
    # Ensure Models directory exists
    os.makedirs('Models', exist_ok=True)
    model_path = 'Models/adult_chf_classifier.pkl'
    joblib.dump(best_result['Best_Model'], model_path)
    print(f"Saved best model to {model_path}")

if __name__ == "__main__":
    main()