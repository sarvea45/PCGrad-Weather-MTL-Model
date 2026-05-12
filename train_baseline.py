import os
import pandas as pd
import tensorflow as tf
from dataset import generate_ap_weather_data
from model import build_mtl_model
from utils import calculate_metrics_task_a, calculate_metrics_task_b, save_final_metrics

def main():
    os.makedirs('results', exist_ok=True)
    train_ds, val_ds, input_shape, _, _, _ = generate_ap_weather_data()
    model = build_mtl_model(input_shape=input_shape)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn_a = tf.keras.losses.MeanSquaredError()  # Regression for Temp
    loss_fn_b = tf.keras.losses.BinaryCrossentropy() # Classification for Rain
    
    epochs = 30
    metrics_log = []
    
    @tf.function
    def train_step(inputs, labels_a, labels_b):
        with tf.GradientTape() as tape:
            pred_a, pred_b = model(inputs, training=True)
            labels_a = tf.expand_dims(labels_a, -1)
            labels_b = tf.expand_dims(labels_b, -1)
            loss_a = loss_fn_a(labels_a, pred_a)
            loss_b = loss_fn_b(labels_b, pred_b)
            # Both losses scaled to prevent one completely dominating
            total_loss = loss_a + loss_b * 10.0
            
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss_a, loss_b
        
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for step, (inputs, (labels_a, labels_b)) in enumerate(train_ds):
            loss_a, loss_b = train_step(inputs, labels_a, labels_b)
            
        # Validation
        val_true_a, val_pred_a = [], []
        val_true_b, val_pred_b = [], []
        for val_inputs, (val_labels_a, val_labels_b) in val_ds:
            p_a, p_b = model(val_inputs, training=False)
            val_true_a.extend(val_labels_a.numpy())
            val_pred_a.extend(p_a.numpy()[:, 0])
            val_true_b.extend(val_labels_b.numpy())
            val_pred_b.extend(p_b.numpy()[:, 0])
            
        mae_a = calculate_metrics_task_a(val_true_a, val_pred_a)
        acc_b, f1_b = calculate_metrics_task_b(val_true_b, val_pred_b)
        
        print(f"Task A (Temp) - MAE: {mae_a:.4f}")
        print(f"Task B (Rain) - Acc: {acc_b:.4f}, F1: {f1_b:.4f}")
        
        metrics_log.append({
            'epoch': epoch + 1,
            'task_a_mae': mae_a,
            'task_b_acc': acc_b,
            'task_b_f1': f1_b
        })
        
    df = pd.DataFrame(metrics_log)
    df.to_csv('results/baseline_metrics.csv', index=False)
    
    # Save final metrics
    final_mae_a = metrics_log[-1]['task_a_mae']
    final_acc_b = metrics_log[-1]['task_b_acc']
    final_f1_b = metrics_log[-1]['task_b_f1']
    save_final_metrics('baseline', final_mae_a, final_acc_b, final_f1_b)
    
    model.save_weights('results/baseline_weights.h5')
    print("Baseline training complete.")

if __name__ == "__main__":
    main()
