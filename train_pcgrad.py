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
    loss_fn_a = tf.keras.losses.MeanSquaredError()
    loss_fn_b = tf.keras.losses.BinaryCrossentropy()
    
    epochs = 30
    metrics_log = []
    conflict_log = []
    global_step = 0
    
    @tf.function
    def train_step(inputs, labels_a, labels_b):
        with tf.GradientTape(persistent=True) as tape:
            pred_a, pred_b = model(inputs, training=True)
            labels_a = tf.expand_dims(labels_a, -1)
            labels_b = tf.expand_dims(labels_b, -1)
            loss_a = loss_fn_a(labels_a, pred_a)
            loss_b = loss_fn_b(labels_b, pred_b)
            # Scale classification loss slightly so gradients are more balanced
            loss_b_scaled = loss_b * 10.0
            
        grad_a = tape.gradient(loss_a, model.backbone.trainable_variables)
        grad_b = tape.gradient(loss_b_scaled, model.backbone.trainable_variables)
        
        grad_head_a = tape.gradient(loss_a, model.head_a.trainable_variables)
        grad_head_b = tape.gradient(loss_b_scaled, model.head_b.trainable_variables)
        del tape
        
        # Flatten backbone gradients
        flat_grad_a = tf.concat([tf.reshape(g, [-1]) for g in grad_a], axis=0)
        flat_grad_b = tf.concat([tf.reshape(g, [-1]) for g in grad_b], axis=0)
        
        # Calculate cosine similarity
        dot_product = tf.reduce_sum(flat_grad_a * flat_grad_b)
        norm_a = tf.norm(flat_grad_a)
        norm_b = tf.norm(flat_grad_b)
        cosine_sim = dot_product / (norm_a * norm_b + 1e-8)
        
        def true_fn():
            proj_a = []
            proj_b = []
            for g_a, g_b in zip(grad_a, grad_b):
                dot = tf.reduce_sum(g_a * g_b)
                norm_sq_a = tf.reduce_sum(tf.square(g_a)) + 1e-8
                norm_sq_b = tf.reduce_sum(tf.square(g_b)) + 1e-8
                g_a_proj = g_a - (dot / norm_sq_b) * g_b
                g_b_proj = g_b - (dot / norm_sq_a) * g_a
                proj_a.append(g_a_proj)
                proj_b.append(g_b_proj)
            return proj_a, proj_b
            
        def false_fn():
            return grad_a, grad_b
            
        processed_grad_a, processed_grad_b = tf.cond(cosine_sim < 0.0, true_fn, false_fn)
        
        final_gradients_backbone = [g_a + g_b for g_a, g_b in zip(processed_grad_a, processed_grad_b)]
        
        all_vars = model.backbone.trainable_variables + model.head_a.trainable_variables + model.head_b.trainable_variables
        all_grads = final_gradients_backbone + grad_head_a + grad_head_b
        
        optimizer.apply_gradients(zip(all_grads, all_vars))
        
        return loss_a, loss_b, cosine_sim
        
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for step, (inputs, (labels_a, labels_b)) in enumerate(train_ds):
            loss_a, loss_b, cosine_sim = train_step(inputs, labels_a, labels_b)
            
            conflict_log.append({
                'step': global_step,
                'cosine_similarity': float(cosine_sim.numpy())
            })
            global_step += 1
            
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
        
    pd.DataFrame(conflict_log).to_csv('results/gradient_conflict.csv', index=False)
    pd.DataFrame(metrics_log).to_csv('results/pcgrad_metrics.csv', index=False)
    
    # Save final metrics
    final_mae_a = metrics_log[-1]['task_a_mae']
    final_acc_b = metrics_log[-1]['task_b_acc']
    final_f1_b = metrics_log[-1]['task_b_f1']
    save_final_metrics('pcgrad', final_mae_a, final_acc_b, final_f1_b)
    
    model.save_weights('results/pcgrad_weights.h5')
    print("PCGrad training complete.")

if __name__ == "__main__":
    main()
