# Comparative Evaluation — Baseline Models

Benchmark suite comparing BR-MTGNN against 11 baseline models on multivariate time-series forecasting.

---

## Dataset

**File:** `data/sm_data.csv`  
**Shape:** 264 timesteps × 190 features (96 RMD + 89 PT + 5 other)  
**Split:** 60% train / 20% validation / 20% test  
**Input window:** 10 timesteps → **Forecast horizon:** 36 timesteps

---

## Models

| Model | Type | Description |
|---|---|---|
| BR-MTGNN | Graph-Temporal | Bayesian Recurrent-MTGNN — the proposed model |
| BMTGNN | Graph-Temporal | Bayesian-Multivariate Temporal Graph Neural Network |
| MTGNN | Graph-Temporal | Multivariate Temporal Graph Neural Network |
| LSTM_M | Recurrent | LSTM, multivariate input |
| LSTM_U | Recurrent | LSTM, univariate (channel-independent) |
| Transformer_M | Attention | Transformer encoder, multivariate input |
| Transformer_U | Attention | Transformer encoder, univariate |
| DCRNN | Graph-Recurrent | Diffusion Convolutional Recurrent Neural Network |
| AGCRN | Graph-Recurrent | Adaptive Graph Convolutional Recurrent Network |
| PatchTST | Patch-Attention | Patch Time Series Transformer (channel-independent) |
| TFT | Attention | Temporal Fusion Transformer (full architecture) |
| TimesFM | Foundation | TimesFM 1.0 pretrained backbone (frozen) + fine-tuned linear head |
| Prophet | Classical-Probabilistic | Facebook Prophet additive model, per-node univariate (yearly seasonality, 95% PI) |

---

## Quick Start

### Install dependencies

```bash
pip install -r requirements.lock.txt
```

### Run all models

```bash
cd B-MTGNN/BR_MTGNN/Comparative_Evaluation
python3 run_all_baselines.py
```

### Common options

```bash
# Adjust epochs
python3 run_all_baselines.py --epochs 50

# Use GPU
python3 run_all_baselines.py --epochs 50 --device cuda:0

# Run models in parallel across two GPUs
python3 run_all_baselines.py --epochs 50 --devices cuda:0,cuda:1 --parallel

# Skip specific models
python3 run_all_baselines.py --epochs 50 --skip-models BMTGNN,MTGNN

# ReduceLROnPlateau schedule
python3 run_all_baselines.py --epochs 100 --lr_schedule plateau --patience 20
```

### Run a single model

```bash
# Advanced baselines (TFT, DCRNN, AGCRN, PatchTST, TimesFM, Prophet)
cd Baselines/Advanced
python3 tft.py              --data ../../data/sm_data.csv --epochs 50 --device cpu
python3 dcrnn.py            --data ../../data/sm_data.csv --epochs 50 --device cpu
python3 agcrn.py            --data ../../data/sm_data.csv --epochs 50 --device cpu
python3 patchtst.py         --data ../../data/sm_data.csv --epochs 50 --device cpu
python3 timesfm.py          --data ../../data/sm_data.csv --epochs 50 --device cpu
python3 prophet_baseline.py --data ../../data/sm_data.csv --seq_out_len 36

# LSTM / Transformer baselines
cd Baselines/LSTM
python3 LSTM_m.py --data ./data/sm_data.csv --epochs 50 --device cpu

# BMTGNN
cd BMTGNN
python3 BMTGNN.py --data ./data/sm_data.csv --epochs 50 --device cpu
```

---

## Output Files

All files are written to `Comparative_Evaluation/` after `run_all_baselines.py` finishes:

| File | Contents |
|---|---|
| `BASELINE_COMPARISON.md` | Markdown comparison table (RSE / RAE / Corr per model) |
| `baseline_comparison_results.csv` | Same table as CSV |
| `baseline_comparison_all_results.csv` | Full table including failed / skipped models |
| `baseline_run_summary.json` | Run config + all metrics as JSON |
| `baseline_run_summary.txt` | Plain-text summary |

### Regenerate table from existing results

```bash
python3 generate_comparison_table.py
```

---

## Metrics

| Metric | Direction | Description |
|---|---|---|
| **RSE** | Lower is better | Relative Squared Error |
| **RAE** | Lower is better | Relative Absolute Error |
| **Corr** | Higher is better | Pearson Correlation (−1 to 1) |
| **Coverage** | Target ≈ 0.95 | Prediction interval coverage (BR-MTGNN only) |

---

## Directory Structure

```
Comparative_Evaluation/
├── README.md
├── requirements.lock.txt
├── run_all_baselines.py          ← Orchestrator: runs all models, writes reports
├── generate_comparison_table.py  ← Standalone report generator
│
├── data/
│   ├── sm_data.csv               ← Shared dataset (264 × 190)
│   ├── graph.csv
│   └── graph_sparse.csv
│
├── BMTGNN/
│   ├── BMTGNN.py
│   ├── net.py, layer.py, trainer.py, util.py
│   └── data/
│
├── BR-MTGNN/
│   ├── train_test.py
│   ├── net.py, layer.py, trainer.py, util.py
│   └── data/
│
└── Baselines/
    ├── seq_common.py             ← Shared training loop for LSTM / Transformer
    ├── LSTM/
    │   ├── LSTM_m.py, LSTM_u.py
    │   └── data/
    ├── Transformer/
    │   ├── transformer_m.py, transformer_u.py
    │   └── data/
    ├── MTGNN/
    │   ├── MTGNN.py
    │   └── data/
    └── Advanced/
        ├── common.py             ← Shared training loop for Advanced baselines
        ├── models.py             ← DCRNN, AGCRN, PatchTST, TFT, TimesFMModel
        ├── train_model.py        ← Dispatcher
        ├── dcrnn.py
        ├── agcrn.py
        ├── patchtst.py
        ├── tft.py
        ├── timesfm.py
        └── prophet_baseline.py  ← Prophet per-node univariate (no --epochs needed)
```

---

## Notes

- **TimesFM** loads pretrained weights from `google/timesfm-1.0-200m-pytorch` (downloaded automatically via `huggingface_hub` on first run and cached locally). The 200M-parameter backbone is frozen; only a lightweight linear head is fine-tuned.
- **TFT** implements the full architecture from Lim et al. (2021): GRN, Variable Selection Networks, shared-V Interpretable Multi-Head Attention, LSTM encoder-decoder, and gated skip connections at every stage.
- All Advanced baselines share the same training loop in `common.py` (L1 loss, Adam, early stopping, blend search on validation).
- **Prophet** fits one additive model per time series (190 series total). It uses yearly seasonality, `changepoint_prior_scale=0.05`, and `interval_width=0.95` for prediction interval coverage. No GPU or `--epochs` argument is needed; Stan inference runs on CPU automatically. The Stan backend is bundled via `cmdstanpy` — no separate Stan installation required. Install with `pip install prophet cmdstanpy`.
