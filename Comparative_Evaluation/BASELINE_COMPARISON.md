# Baseline Comparison Results

Generated: 2026-05-22 10:35:01

**Configuration:**
- Epochs: 50
- Devices: cuda:0
- Trials: 1
- Repeats: 1
- Batch size: 16
- Data: 264 timesteps × 190 features

## Results Table

| RSE | RAE | Corr | elapsed_time | status | Coverage |
|-----|-----|------|--------------|--------|----------|
| BMTGNN | 5.9315 | 6.9052 | 0.0 | 110.1621 | completed | nan |
| BR-MTGNN | 0.4284 | 0.2505 | 0.913 | 27.652 | completed | 0.9503 |
| LSTM_M | 2.6753 | 1.8674 | 0.1805 | 11.4809 | completed | nan |
| LSTM_U | 2.2522 | 1.2349 | 0.2444 | 14.317 | completed | nan |
| MTGNN | 5.6867 | 6.5561 | 0.0 | 54.1381 | completed | nan |
| Transformer_M | 3.9313 | 2.7456 | 0.0819 | 15.5693 | completed | nan |
| Transformer_U | 1.8306 | 1.2718 | 0.0198 | 40.8489 | completed | nan |
| DCRNN | 0.4305 | 0.2548 | 0.0 | 6.6121 | completed | nan |
| AGCRN | 0.9346 | 0.5173 | 0.1199 | 7.058 | completed | nan |
| PatchTST | 0.6743 | 0.355 | 0.1449 | 5.9447 | completed | nan |
| TFT | 0.642 | 0.3563 | 0.1901 | 110.0308 | completed | nan |
| TimesFM | 1.4051 | 0.7154 | 0.102 | 29.6937 | completed | nan |
| Prophet | 7.7073 | 4.0637 | 0.0 | 0.3326 | completed | nan |


## Metric Definitions

### Primary Metrics
- **RAE**: Relative Absolute Error (lower is better)
- **RSE**: Relative Squared Error (lower is better)
- **Corr**: Pearson Correlation (higher is better, -1 to 1)
- **Coverage**: Prediction interval coverage (target: ~0.95 for 95%)
- **elapsed_time**: Execution time in seconds

## Model Descriptions

| Model | Type | Description |
|-------|------|-------------|
| BMTGNN | Graph-Temporal | Bayesian-Multivariate Temporal Graph Neural Network |
| BR-MTGNN | Graph-Temporal | Bayesian Recurrent-Multivariate Temporal Graph Neural Network |
| LSTM_M | Recurrent | LSTM with Multivariate input |
| LSTM_U | Recurrent | LSTM with Univariate input |
| MTGNN | Graph-Temporal | Multivariate Temporal Graph Neural Network |
| Transformer_M | Attention | Transformer with Multivariate input |
| Transformer_U | Attention | Transformer with Univariate input |
| DCRNN | Graph-Recurrent | Diffusion Convolutional Recurrent Neural Network |
| AGCRN | Graph-Recurrent | Adaptive Graph Convolutional Recurrent Network |
| PatchTST | Patch-Attention | Patch Time Series Transformer (channel-independent) |
| TFT | Attention | Temporal Fusion Transformer |
| TimesFM | Foundation | TimesFM 1.0 pretrained backbone (frozen) + fine-tuned linear head |
| Prophet | Classical-Probabilistic | Facebook Prophet additive model (yearly seasonality, per-node univariate) |
