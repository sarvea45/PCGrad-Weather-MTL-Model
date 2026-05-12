import numpy as np
import pandas as pd
import tensorflow as tf

def generate_ap_weather_data(num_days=2000, sequence_length=7, batch_size=32):
    """
    Generates synthetic daily weather data mimicking Coastal Andhra Pradesh.
    - Input: Sequence of past 7 days (Temp, Humidity, WindSpeed)
    - Task A (Regression): Predict exact Temperature for the next day.
    - Task B (Classification): Predict Rain (1) or No Rain (0) for the next day.
    """
    # 1. Base seasonal patterns (365 day cycle)
    np.random.seed(42)
    time = np.arange(num_days)
    
    # Coastal AP: Temp peaks in May (day 150), dips in Dec. Base ~30°C, amp ~8°C
    temp = 30 + 8 * np.sin(2 * np.pi * (time - 60) / 365) + np.random.normal(0, 1.5, num_days)
    
    # Humidity is high during monsoons (June-Oct). Base ~70%, amp ~15%
    humidity = 70 + 15 * np.sin(2 * np.pi * (time - 120) / 365) + np.random.normal(0, 5, num_days)
    humidity = np.clip(humidity, 40, 100)
    
    # Wind speed spikes during cyclone season (Oct/Nov)
    wind_speed = 15 + 5 * np.sin(2 * np.pi * (time - 250) / 365) + np.random.normal(0, 3, num_days)
    wind_speed = np.clip(wind_speed, 0, 50)
    
    # Create tabular dataframe
    df = pd.DataFrame({'Temp': temp, 'Humidity': humidity, 'WindSpeed': wind_speed})
    
    # Normalize input features for the neural network
    df_norm = (df - df.mean()) / df.std()
    
    # 2. Generate Targets
    # Task A: Next day's temperature (we keep this un-normalized for clear MSE interpretation)
    next_day_temp = df['Temp'].shift(-1)
    
    # Task B: Rain tomorrow? (High probability if today's humidity is > 80%)
    rain_prob = 1 / (1 + np.exp(-(df['Humidity'] - 80) / 5)) 
    rain_tomorrow = (np.random.rand(num_days) < rain_prob).astype(int).shift(-1)
    
    df_norm['Target_Temp'] = next_day_temp
    df_norm['Target_Rain'] = rain_tomorrow
    
    # Drop the last row because of the shift(-1) NaN
    df_norm = df_norm.dropna()
    
    # 3. Create Spatio-Temporal Sequences (Sliding Window)
    X, y_temp, y_rain = [], [], []
    features = ['Temp', 'Humidity', 'WindSpeed']
    
    for i in range(len(df_norm) - sequence_length):
        X.append(df_norm[features].iloc[i : i + sequence_length].values)
        y_temp.append(df_norm['Target_Temp'].iloc[i + sequence_length - 1])
        y_rain.append(df_norm['Target_Rain'].iloc[i + sequence_length - 1])
        
    X = np.array(X, dtype=np.float32)
    y_temp = np.array(y_temp, dtype=np.float32)
    y_rain = np.array(y_rain, dtype=np.float32)
    
    # 4. Build tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((X, (y_temp, y_rain)))
    
    # Split into Train (80%) and Val (20%)
    train_size = int(0.8 * len(X))
    train_ds = dataset.take(train_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = dataset.skip(train_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    input_shape = X.shape[1:] # (sequence_length, num_features)
    
    # Yielding raw data for Streamlit UMAP without breaking training signature
    return train_ds, val_ds, input_shape, X[train_size:], y_temp[train_size:], y_rain[train_size:]

if __name__ == "__main__":
    train, val, shape, _, _, _ = generate_ap_weather_data()
    print(f"Data generated successfully. Input shape: {shape}")
    for x, (y_t, y_r) in train.take(1):
        print(f"Batch X shape: {x.shape}")
        print(f"Batch Target Temp shape: {y_t.shape}")
        print(f"Batch Target Rain shape: {y_r.shape}")
