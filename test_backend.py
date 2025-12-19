"""Test script for the Heart Monitoring Backend API"""

from fastapi.testclient import TestClient
from backend import app

client = TestClient(app)

# Test health check
print('=== Health Check ===')
response = client.get('/')
print(response.json())

# Test adult prediction (healthy HRV values)
print('\n=== Adult Prediction (Healthy) ===')
response = client.post('/predict/adult', json={
    'sdnn': 120,
    'rmssd': 45,
    'pnn50': 15,
    'mean_hr': 72
})
print(response.json())

# Test adult prediction (unhealthy HRV values)
print('\n=== Adult Prediction (Heart Failure) ===')
response = client.post('/predict/adult', json={
    'sdnn': 35,
    'rmssd': 15,
    'pnn50': 2,
    'mean_hr': 95
})
print(response.json())

# Test fetal prediction with dataset mean values (should be ~50/50)
print('\n=== Fetal Prediction (Dataset Mean Values) ===')
response = client.post('/predict/fetal/simple', json={
    'Mean_FHR': 105.22,
    'Std_FHR': 55.44,
    'Mean_UC': 18.24
})
print(response.json())

# Test fetal prediction (healthier values - good pH, normal HR)
print('\n=== Fetal Prediction (Healthier Values) ===')
response = client.post('/predict/fetal/simple', json={
    'Mean_FHR': 130,
    'Std_FHR': 40,
    'Mean_UC': 15,
    'pH': 7.30
})
print(response.json())

# Test fetal prediction (concerning values - low pH, abnormal HR)
print('\n=== Fetal Prediction (Concerning Values) ===')
response = client.post('/predict/fetal/simple', json={
    'Mean_FHR': 80,
    'Std_FHR': 80,
    'Mean_UC': 50,
    'pH': 7.05
})
print(response.json())
