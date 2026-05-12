# Spatio-Temporal Weather Forecasting with PCGrad

This project implements a Multi-Task Learning (MTL) model in TensorFlow using a 1D-CNN backbone to forecast Coastal Andhra Pradesh weather patterns. It predicts two conflicting targets simultaneously:
- **Task A (Regression):** Temperature (°C) for the next day.
- **Task B (Classification):** Probability of Precipitation (Rain/No Rain) for the next day.

It leverages Projecting Conflicting Gradients (PCGrad) to mitigate gradient conflicts (negative transfer) during training, which occur naturally when optimizing for both extreme heatwaves and monsoon rains. A Streamlit application is provided to monitor real-time cosine similarity of gradients, track performance, and inspect the shared 1D-CNN representation space using UMAP.

## Setup and Running

Ensure Docker and Docker Compose are installed.

1. Clone the repository.
2. The `.env` file should be copied from `.env.example`.
3. Run `docker-compose up --build`.
4. Access the Streamlit dashboard at `http://localhost:8501`.

## Files
- `dataset.py`: Generates the synthetic Coastal AP Spatio-Temporal weather data.
- `train_baseline.py`: Trains the naive summation model.
- `train_pcgrad.py`: Trains the model with PCGrad gradient surgery.
- `generate_model_summary.py`: Generates `results/model_architecture.txt`.
- `app.py`: Streamlit dashboard.
