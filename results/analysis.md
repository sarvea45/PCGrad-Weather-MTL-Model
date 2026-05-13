### Gradient Conflict Analysis
During the initial epochs of training the 1D-CNN backbone on the Coastal AP weather data, the cosine similarity between the gradients of Task A (Temperature Regression) and Task B (Precipitation Classification) frequently fluctuates, occasionally dropping below zero. This indicates that the tasks are competing to update the temporal feature extractor in opposing directions—for instance, when predicting extreme heatwaves versus heavy monsoon downpours. PCGrad actively projects these conflicting gradients, ensuring that learning the features for one weather event does not detrimentally "overwrite" the features needed for the other.

### Shared Representation Analysis
The UMAP visualizations of the shared representation (extracted from the final dense layer of the 1D-CNN) reveal that the PCGrad model learns a feature space that is well-organized for both forecasting tasks. The embeddings show clear structural separation when colored by "Temperature (Hot/Cool)" and "Precipitation (Rain/No Rain)". This confirms that the temporal convolutions have successfully captured complex spatio-temporal features (like incoming low-pressure systems or dry spells) that are jointly useful for both regression and classification without suffering from negative transfer.

### Final Performance Comparison
Based on the real training metrics in `final_metrics.json`, the PCGrad model demonstrates a tangible improvement over the naive baseline. 
- **Task A (Temperature):** The Mean Absolute Error (MAE) improved from 2.589 to 2.509.
- **Task B (Precipitation):** The Accuracy saw a marginal improvement from 78.4% to 78.6%, while maintaining a stable F1-score (~0.57).

These results highlight the reality of Multi-Task Learning: PCGrad successfully prevented negative transfer. It allowed the 1D-CNN backbone to optimize for the complex continuous variables of temperature regression without sacrificing the model's ability to classify precipitation boundaries.
