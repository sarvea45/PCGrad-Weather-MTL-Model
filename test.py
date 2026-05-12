import traceback
from dataset import generate_ap_weather_data
from model import build_mtl_model
import numpy as np
import umap

try:
    _, _, input_shape, X_test, y_temp_test, y_rain_test = generate_ap_weather_data()
    model = build_mtl_model(input_shape=input_shape)
    model.load_weights('results/pcgrad_weights.h5')
    
    idx = np.random.choice(len(X_test), min(len(X_test), 1000), replace=False)
    X_sample = X_test[idx]
    y_temp_sample = y_temp_test[idx]
    y_rain_sample = y_rain_test[idx]
    
    reprs = model.backbone.predict(X_sample)
    print("reprs shape:", reprs.shape)
    
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(reprs)
    print("embedding shape:", embedding.shape)
except Exception as e:
    traceback.print_exc()
