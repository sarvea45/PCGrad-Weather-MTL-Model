import json
import os
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
import numpy as np

def calculate_metrics_task_a(y_true, y_pred):
    """Regression metrics for Temperature"""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    mae = mean_absolute_error(y_true, y_pred)
    return mae

def calculate_metrics_task_b(y_true, y_pred, threshold=0.5):
    """Classification metrics for Precipitation"""
    y_pred_bin = (np.array(y_pred).flatten() > threshold).astype(int)
    y_true = np.array(y_true).flatten().astype(int)
    acc = accuracy_score(y_true, y_pred_bin)
    f1 = f1_score(y_true, y_pred_bin, zero_division=0)
    return acc, f1

def save_final_metrics(model_type, mae_a, acc_b, f1_b):
    filepath = 'results/final_metrics.json'
    os.makedirs('results', exist_ok=True)
    
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            try:
                data = json.load(f)
            except:
                data = {}
    else:
        data = {}
        
    data[model_type] = {
        "task_a": {
            "mae": float(mae_a)
        },
        "task_b": {
            "accuracy": float(acc_b),
            "f1_score": float(f1_b)
        }
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
