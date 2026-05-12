import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import umap.umap_ as umap
import plotly.express as px
from dataset import generate_ap_weather_data
from model import build_mtl_model

st.set_page_config(page_title="Coastal AP Weather - PCGrad Dashboard", layout="wide")

st.title("Multi-Task Weather Forecasting with PCGrad")

# 1. Gradient Conflict Monitor
st.header("Gradient Conflict Monitor")
st.markdown("<div data-testid='gradient-conflict-monitor'></div>", unsafe_allow_html=True)
try:
    df_conflict = pd.read_csv('results/gradient_conflict.csv')
    fig_conflict = px.line(df_conflict, x='step', y='cosine_similarity', title='Cosine Similarity of Backbone Gradients (Temp vs Rain)')
    fig_conflict.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig_conflict, use_container_width=True)
except FileNotFoundError:
    st.warning("gradient_conflict.csv not found. Train the PCGrad model first.")

# 2. Performance Dashboard
st.header("Performance Dashboard")
st.markdown("<div data-testid='performance-dashboard'></div>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
try:
    df_base = pd.read_csv('results/baseline_metrics.csv')
    df_pc = pd.read_csv('results/pcgrad_metrics.csv')
    
    with col1:
        st.subheader("Task A: Temperature MAE (Lower is Better)")
        chart_data_a = pd.DataFrame({
            'Epoch': df_base['epoch'],
            'Baseline': df_base['task_a_mae'],
            'PCGrad': df_pc['task_a_mae']
        }).set_index('Epoch')
        st.line_chart(chart_data_a)
        
    with col2:
        st.subheader("Task B: Precipitation Accuracy (Higher is Better)")
        chart_data_b = pd.DataFrame({
            'Epoch': df_base['epoch'],
            'Baseline': df_base['task_b_acc'],
            'PCGrad': df_pc['task_b_acc']
        }).set_index('Epoch')
        st.line_chart(chart_data_b)
except FileNotFoundError:
    st.warning("Metrics CSVs not found. Train both models first.")

# 3. Representation Inspector
st.header("Shared Representation Inspector")
st.markdown("<div data-testid='representation-inspector'></div>", unsafe_allow_html=True)
if st.button("Generate UMAP Plots"):
    with st.spinner("Loading model and generating embeddings..."):
        try:
            _, _, input_shape, X_test, y_temp_test, y_rain_test = generate_ap_weather_data()
            model = build_mtl_model(input_shape=input_shape)
            try:
                model.load_weights('results/pcgrad_weights.h5')
            except Exception:
                st.warning("Pre-trained weights not found at 'results/pcgrad_weights.h5'. Generating UMAP using an un-trained model for demonstration.")
            
            # Subsample for UMAP performance
            idx = np.random.choice(len(X_test), min(len(X_test), 1000), replace=False)
            X_sample = X_test[idx]
            y_temp_sample = y_temp_test[idx]
            y_rain_sample = y_rain_test[idx]
            
            # Get representations
            reprs = model.backbone.predict(X_sample)
            
            reducer = umap.UMAP(n_components=2, random_state=42)
            embedding = reducer.fit_transform(reprs)
            
            # Discretize temperature for better coloring
            temp_labels = np.where(y_temp_sample > 32, 'Hot (>32C)', 'Cool (<32C)')
            rain_labels = np.where(y_rain_sample == 1, 'Rain', 'No Rain')
            
            df_umap = pd.DataFrame({
                'UMAP_1': embedding[:, 0],
                'UMAP_2': embedding[:, 1],
                'Temperature': temp_labels,
                'Precipitation': rain_labels
            })
            
            col_u1, col_u2 = st.columns(2)
            with col_u1:
                fig_a = px.scatter(df_umap, x='UMAP_1', y='UMAP_2', color='Temperature', title="UMAP Colored by Temperature")
                st.plotly_chart(fig_a, use_container_width=True)
            with col_u2:
                fig_b = px.scatter(df_umap, x='UMAP_1', y='UMAP_2', color='Precipitation', title="UMAP Colored by Precipitation")
                st.plotly_chart(fig_b, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error generating UMAP plots: {e}")
