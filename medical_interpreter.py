import os
import joblib
import pandas as pd
import numpy as np

class MedicalInterpreter:
    def __init__(self):
        self.adult_model = None
        self.fetal_model = None
        self.fetal_scaler = None
        self.feature_names = None
        self.load_models()

    def load_models(self):
        try:
            if os.path.exists('Models/adult_chf_classifier.pkl'):
                self.adult_model = joblib.load('Models/adult_chf_classifier.pkl')
            
            # Load CTU Full Model (Retrained)
            if os.path.exists('Models/ctu_full_model.pkl'):
                self.fetal_model = joblib.load('Models/ctu_full_model.pkl')
                print("Loaded CTU Full Model")
            # Fallback to Binary Model
            elif os.path.exists('Models/ctu_binary_model.pkl'):
                self.fetal_model = joblib.load('Models/ctu_binary_model.pkl')
            # Fallback to Notebook Model
            elif os.path.exists('Models/fetal_health_best_model.pkl'):
                self.fetal_model = joblib.load('Models/fetal_health_best_model.pkl')
            # Fallback to old CTU model
            elif os.path.exists('Models/ctu_heart_failure_model.pkl'):
                self.fetal_model = joblib.load('Models/ctu_heart_failure_model.pkl')
                
            if os.path.exists('Models/scaler_ctu_full.pkl'):
                self.fetal_scaler = joblib.load('Models/scaler_ctu_full.pkl')
            elif os.path.exists('Models/scaler_ctu_binary.pkl'):
                self.fetal_scaler = joblib.load('Models/scaler_ctu_binary.pkl')
            elif os.path.exists('Models/fetal_health_scaler.pkl'):
                self.fetal_scaler = joblib.load('Models/fetal_health_scaler.pkl')
            elif os.path.exists('Models/scaler_ctu.pkl'):
                self.fetal_scaler = joblib.load('Models/scaler_ctu.pkl')
            
            if os.path.exists('Models/ctu_feature_names.pkl'):
                self.feature_names = joblib.load('Models/ctu_feature_names.pkl')
                
        except Exception as e:
            print(f"Error loading models: {e}")

    def interpret_adult_hrv(self, metrics):
        # 1. Rule-based Diagnostic
        sdnn = metrics['SDNN']
        diagnosis = "NORMAL"
        severity = "Low Risk"
        details = []

        # Time Domain Rules
        if sdnn < 50:
            diagnosis = "POTENTIAL HEART FAILURE"
            severity = "CRITICAL"
            details.append("Severely reduced SDNN (<50ms).")
        elif sdnn < 100:
            diagnosis = "AT RISK"
            severity = "Moderate"
            details.append("Moderate reduction in variability.")
            
        if metrics['RMSSD'] < 20: details.append("Low RMSSD (Reduced Parasympathetic).")
        
        # Frequency Domain Rules
        if metrics['LF/HF Ratio'] > 5: details.append("High LF/HF Ratio (Sympathetic Stress).")
        if metrics['VLF Power'] < 100: details.append("Low VLF Power (Renin-Angiotensin System dysfunction).")
        
        # Non-linear Rules
        if metrics['SD1'] < 10: details.append("Low SD1 (Reduced short-term variability).")
        if metrics['SD2'] < 30: details.append("Low SD2 (Reduced long-term variability).")
        
        if not details: details.append("Autonomic function normal.")
            
        rules_text = f"Diagnosis: {diagnosis}\nSeverity: {severity}\n\nNotes:\n- " + "\n- ".join(details)

        # 2. AI Confidence
        ai_text = "AI Model Not Loaded"
        if self.adult_model:
            features = pd.DataFrame([{
                'sdnn': metrics['SDNN'],
                'rmssd': metrics['RMSSD'],
                'pnn50': metrics['pNN50'],
                'mean_hr': metrics['Mean HR']
            }])
            try:
                prob = self.adult_model.predict_proba(features)[0]
                hf_prob = prob[1]
                
                pred_label = "HEART FAILURE" if hf_prob > 0.5 else "HEALTHY"
                confidence = hf_prob if hf_prob > 0.5 else 1 - hf_prob
                
                ai_text = f"Prediction: {pred_label}\nConfidence: {confidence*100:.1f}%\n(Probability of HF: {hf_prob:.2f})"
            except Exception as e:
                ai_text = f"AI Error: {str(e)}"
        
        return rules_text, ai_text

    def interpret_fetal_ctg(self, metrics):
        # 1. Rule-based (Only if signal metrics available)
        rules_text = "Rule-based analysis unavailable (No raw signal)."
        if 'Baseline' in metrics and metrics['Baseline'] > 0:
            baseline = metrics['Baseline']
            stv = metrics['STV']
            ltv = metrics.get('LTV', 0)
            sinusoidal = metrics.get('Sinusoidal_Idx', 0)
            
            status = "REASSURING"
            details = []
            
            # Baseline Rules
            if baseline < 110: 
                status = "BRADYCARDIA"
                details.append("Baseline < 110 bpm.")
            elif baseline > 160: 
                status = "TACHYCARDIA"
                details.append("Baseline > 160 bpm.")
                
            # Variability Rules
            if stv < 3: 
                status = "NON-REASSURING"
                details.append("Critically low STV (<3ms).")
            if ltv < 5:
                details.append("Low LTV (Reduced long-term variability).")
                
            # Deceleration Rules
            if metrics.get('Dec_Count', 0) > 2:
                status = "PATHOLOGICAL"
                details.append(f"Frequent Decelerations ({metrics['Dec_Count']} detected).")
                
            # Sinusoidal Pattern
            if sinusoidal > 0.1: # Threshold for significant sinusoidal component
                status = "CRITICAL"
                details.append("Sinusoidal Pattern Detected (Severe Anemia/Hypoxia).")
                
            if not details: details.append("Normal ranges.")
            rules_text = f"Status: {status}\n\nNotes:\n- " + "\n- ".join(details)

        # 2. AI Confidence
        ai_text = "AI Model Not Loaded"
        if self.fetal_model and self.fetal_scaler and 'ML_Features' in metrics:
            try:
                X = metrics['ML_Features']
                
                # Ensure columns match what the scaler expects
                X_scaled = self.fetal_scaler.transform(X)
                
                # Check if model supports predict_proba
                if hasattr(self.fetal_model, "predict_proba"):
                    probs = self.fetal_model.predict_proba(X_scaled)[0]
                    pred_class = np.argmax(probs)
                    confidence = probs[pred_class]
                    
                    # Check if binary (CTU) or multiclass (f_health)
                    if len(probs) == 2:
                        # Binary: 0=Normal, 1=Pathological
                        labels = {0: "NORMAL", 1: "PATHOLOGICAL"}
                        pred_label = labels.get(pred_class, "UNKNOWN")
                        ai_text = (f"Prediction: {pred_label}\n"
                                   f"Confidence: {confidence*100:.1f}%\n"
                                   f"(Normal: {probs[0]:.2f}, Pathological: {probs[1]:.2f})")
                    else:
                        # Multiclass: 0=Normal, 1=Suspect, 2=Pathological
                        labels = {0: "NORMAL", 1: "SUSPECT", 2: "PATHOLOGICAL"}
                        pred_label = labels.get(pred_class, "UNKNOWN")
                        ai_text = (f"Prediction: {pred_label}\n"
                                   f"Confidence: {confidence*100:.1f}%\n"
                                   f"(N: {probs[0]:.2f}, S: {probs[1]:.2f}, P: {probs[2]:.2f})")
                else:
                    pred_class = self.fetal_model.predict(X_scaled)[0]
                    # Assume binary if not proba, or check classes
                    labels = {0: "NORMAL", 1: "PATHOLOGICAL"}
                    pred_label = labels.get(pred_class, "UNKNOWN")
                    ai_text = f"Prediction: {pred_label}"

            except Exception as e:
                ai_text = f"AI Error: {str(e)}"
                
        return rules_text, ai_text
