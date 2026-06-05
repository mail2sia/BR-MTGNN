# Quick Start: BR-MTGNN Pipeline

## From Raw Data → Validated → Trained → Forecasts → Publication Outputs

**Goal:** Validate historical PT data with date-based splitting, train BR-MTGNN, generate 36-month forecasts, compare against 11 baselines on the same test window, and build BRGI/CISN decision-support outputs.

**Total time:** ~30 minutes (includes decision map regeneration)  
**Output:** 48 RMD forecast plots + confidence intervals, baseline comparison (11 models), BRGI ranked gaps, CISN spillover value, decision map (325 RMD-PT pairs), 6 manuscript figures, 3 tables

---

## One-Command Full Pipeline

Save the block below as `run_pipeline.sh`, then run it:

```bash
#!/bin/bash
set -e
cd "$(dirname "$0")"   # always run from the repo root

echo "=== Step 0a: Validate Raw Data (Optional) ==="
python gemini_validator.py \
  --input_csv data/data.csv \
  --output_csv data/data_validated.csv \
  --report_json model/Bayesian/validation_report.json

echo "=== Step 0b: Smoothing (Date-Based Split) ==="
python smoothing.py \
  --input_csv data/data.csv \
  --output_csv data/sm_data.csv \
  --alpha 0.10 --beta 0.05 --mode des \
  --no_cache \
  --validation_report model/Bayesian/validation_report.json

echo "=== Step 0c: Graph (Training Data Only) ==="
python make_sparse_graph.py \
  --data_csv data/sm_data.csv \
  --out_csv data/graph_sparse.csv \
  --top_k 8 --window 96 --max_lag 6 --min_corr 0.05 \
  --train_end_date 2019-04

echo "=== Step 1: Validation Training (Date-Based Split) ==="
python train_test.py \
  --device cuda:1 \
  --output_dir model/Bayesian \
  --seq_in_len 10 --seq_out_len 36 \
  --use_tdb_input \
  --epochs 500 --patience 400 \
  --train_end_date 2019-04 \
  --val_end_date 2023-09

echo "=== Step 2: Production Training ==="
python train.py \
  --device cuda:1 \
  --output_dir model/Bayesian \
  --seq_in_len 10 --seq_out_len 36 \
  --use_tdb_input \
  --epochs 500 --lr 1e-5 \
  --lambda_delta 0.02 --lambda_start 0.20 --lambda_horizon 0.05

echo "=== Step 3: Forecast + Plots ==="
python forecast.py \
  --model_path model/Bayesian/o_model.pt \
  --data_csv data/sm_data.csv \
  --nodes_csv data/data.csv \
  --graph_csv data/graph_sparse.csv \
  --output_dir model/Bayesian \
  --use_tdb_input \
  --plot_start_year 2007 \
  --smooth_alpha 0.10

echo "=== Step 4: Extract BR-MTGNN Test Predictions ==="
python extract_brmtgnn_test_predictions.py

echo "=== Step 5: Baseline Comparison (11 Models) ==="
python Comparative_Evaluation/baseline_compare.py \
  --brmtgnn_test_predictions_csv documentation/br_mtgnn_full_calibrated_test_predictions.csv

echo "=== Step 6: Build BRGI+CISN Decision Map ==="
python documentation/brgi_cisn_decision_map.py \
  --brgi_csv model/Bayesian/forecast/brgi/ranked_pairs.csv \
  --cisn_csv model/Bayesian/forecast/cisn/pt_spillover_value.csv \
  --edges_csv model/Bayesian/forecast/cisn/spillover_network_edges.csv \
  --out_dir model/Bayesian/forecast/brgi_cisn_decision_map \
  --top_k 20 --gamma 0.5 --q_threshold 0.75

python documentation/brgi_cisn_nature_composite.py \
  --priority_csv model/Bayesian/forecast/brgi_cisn_decision_map/brgi_cisn_priority_matrix.csv \
  --out_dir model/Bayesian/forecast/brgi_cisn_decision_map \
  --top_k 20

echo "=== Step 7: Manuscript Outputs ==="
python documentation/build_manuscript_outputs.py

echo ""
echo "Pipeline complete!"
echo "  Forecast plots  : model/Bayesian/forecast/plots/"
echo "  Validation audit: model/Bayesian/validation_report.json"
echo "  Tables + Figures: documentation/"
```

```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

> **No GPU?** Replace `--device cuda:1` with `--device cpu` in Steps 1–3.
> **No Gemini key yet?** Hard clinical constraints still run without Gemini. Gemini is required for grounded temporal/anachronism and magnitude review. Get a free key at [aistudio.google.com](https://aistudio.google.com) and add it to `.env`.

---

## Step-by-Step (Run Individually)

### Prerequisites

```bash
cd /home/sahsan03/B-MTGNN/BR_MTGNN

# Install dependencies (once)
pip install -r requirements.txt
pip install google-genai python-dotenv

# Verify Gemini key is in .env
cat .env
# Should show: GEMINI_API_KEY=AIzaSy...

# Verify PyTorch
python -c "import torch; print('PyTorch', torch.__version__, '| CUDA:', torch.cuda.is_available())"
```

---

### Step 0a — Validate Raw Data (Optional Standalone)

Run this if you want to inspect the audit report before committing to smoothing:

```bash
python gemini_validator.py \
  --input_csv data/data.csv \
  --output_csv data/data_validated.csv \
  --report_json model/Bayesian/validation_report.json
```

**Expected output:**
```
[gemini_validator] Loaded data/data.csv: 264 rows × 190 cols
[gemini_validator] Applying hard-constraint clinical blacklist...
[gemini_validator] GLOBAL BLACKLIST ZERO: PT_Clozapine Protocol_NoM (...)
[gemini_validator] Date column: 'Month-Year'
[gemini_validator] Audit complete. X columns corrected, Y duplicates flagged.
```

---

### Step 0b — Smoothing (runs validation automatically, date-based split)

```bash
python smoothing.py \
  --input_csv data/data.csv \
  --output_csv data/sm_data.csv \
  --alpha 0.10 --beta 0.05 --mode des \
  --no_cache \
  --validation_report model/Bayesian/validation_report.json
```

Validation fires before any smoothing. Use `--skip_validation` to bypass entirely. `--no_gemini` skips Gemini-only review, but the hard clinical blacklist still runs.

**Expected output:**
```
[gemini_validator] Applying hard-constraint clinical blacklist...
[gemini_validator] GLOBAL BLACKLIST ZERO: PT_Clozapine Protocol_NoM (...)
[gemini_validator] Audit complete. X columns corrected, 0 duplicates flagged.
Saved data/sm_data.csv with shape (264, 190)
```

**Validation checklist:**
- [ ] `data/sm_data.csv` exists, shape (264, 190)
- [ ] `model/Bayesian/validation_report.json` exists
- [ ] First column is `Month-Year` and no missing-date warning appears

---

### Step 0c — Graph Generation (Training Data Only)

```bash
python make_sparse_graph.py \
  --data_csv data/sm_data.csv \
  --out_csv data/graph_sparse.csv \
  --top_k 8 --window 96 --max_lag 6 --min_corr 0.05 \
  --train_end_date 2019-04
```

**Critical:** `--train_end_date 2019-04` restricts graph adjacency to training period (Jan 2004–Apr 2019, 184 rows), preventing evaluation-period leakage.

**Expected output:**
```
[make_sparse_graph] train_end_date=2019-04: using 184 rows (up to 2019-04) for graph construction.
Saved sparse graph: data/graph_sparse.csv
RMD rows: 48 | PT per RMD: 8 | total undirected edges: 384
Scores saved: data/graph_sparse.scores.csv
Note: adjacency matrix is symmetrized during model training (undirected graph).
```

**Validation checklist:**
- [ ] `data/graph_sparse.csv` exists, shape (48, 10)
- [ ] `data/graph_sparse.scores.csv` exists
- [ ] Output confirms `184 rows` used for training period

---

### Step 1 — Validation Training (Date-Based Split)

```bash
python train_test.py \
  --device cuda:1 \
  --output_dir model/Bayesian \
  --seq_in_len 10 --seq_out_len 36 \
  --use_tdb_input \
  --epochs 500 --patience 400 \
  --train_end_date 2019-04 \
  --val_end_date 2023-09
```

**Critical:** Date-based split eliminates target-period leakage:
- Training: Jan 2004 – Apr 2019 (184 windows, last targets ≤ 2019-04)
- Validation: May 2019 – Sep 2023 (60 windows, 2019-05 ≤ last targets ≤ 2023-09)
- Test: Oct 2023 – Dec 2025 (20 windows, last targets > 2023-09)

**Expected output:**
```
Epoch 500/500: train_loss=X.XXXX, val_loss=Y.YYYY
[train_test] Best val epoch saved: model/Bayesian/model.pt
[util.RMDPTData] Date-based split: train=184, valid=60, test=20 samples
```

**Validation checklist:**
- [ ] `model/Bayesian/model.pt` exists
- [ ] `model/Bayesian/hp.txt` exists (JSON, includes `train_end_date` and `val_end_date`)
- [ ] `model/Bayesian/metadata.json` exists (8 input channels: NoM, NoP, TDB + 5 globals)
- [ ] `model/Bayesian/training_history.csv` shows decreasing loss

---

### Step 2 — Production Training (98% Data)

```bash
python train.py \
  --device cuda:1 \
  --output_dir model/Bayesian \
  --seq_in_len 10 --seq_out_len 36 \
  --use_tdb_input \
  --epochs 500 --lr 1e-5 \
  --lambda_delta 0.02 --lambda_start 0.20 --lambda_horizon 0.05
```

**Expected output:**
```
[train.py] Loaded model/Bayesian/model.pt as warm initialization
[train.py] Production model saved: model/Bayesian/o_model.pt
```

**Validation checklist:**
- [ ] `model/Bayesian/o_model.pt` exists
- [ ] `model/Bayesian/production_train_metadata.json` exists

---

### Step 3 — Forecast + Plots

```bash
python forecast.py \
  --model_path model/Bayesian/o_model.pt \
  --data_csv data/sm_data.csv \
  --nodes_csv data/data.csv \
  --graph_csv data/graph_sparse.csv \
  --output_dir model/Bayesian \
  --use_tdb_input \
  --plot_start_year 2007 \
  --smooth_alpha 0.10 \
  --validation_report model/Bayesian/forecast_validation_report.json
```

Validation runs automatically on `data/data.csv` before any plots are drawn. Hard clinical constraints run even with `--no_gemini`; Gemini adds grounded temporal/anachronism review when enabled.

**Expected output:**
```
[gemini_validator] Audit complete. X columns corrected, 0 duplicates flagged.
[forecast.py] Validated nodes_csv written to data/data.validated.csv
[forecast.py] Plotted 48 RMDs
```

**Validation checklist:**
- [ ] `model/Bayesian/forecast/forecast_36m.csv` exists
- [ ] `model/Bayesian/forecast/plot_values_monthly.csv` exists
- [ ] `model/Bayesian/forecast/plots/` contains 48 PNG files
- [ ] No NaN values: `python -c "import pandas as pd; df = pd.read_csv('model/Bayesian/forecast/forecast_36m.csv'); print('NaNs:', df.isnull().sum().sum())"`

---

### Step 4 — Extract Test Window Predictions

```bash
python extract_brmtgnn_test_predictions.py
```

**Validation checklist:**
- [ ] `documentation/br_mtgnn_full_calibrated_test_predictions.csv` exists
- [ ] Shape: 36 rows × 137 columns (or 138 with date)

---

### Step 5 — Baseline Comparison

```bash
python Comparative_Evaluation/baseline_compare.py \
  --brmtgnn_test_predictions_csv documentation/br_mtgnn_full_calibrated_test_predictions.csv
```

**Validation checklist:**
- [ ] `documentation/Table 1 - Baseline comparison.csv` exists
- [ ] `documentation/Figure 3 - Horizon-block model performance.png` exists
- [ ] All 5 models appear in Table 1

---

### Step 6 — BRGI+CISN Decision Map (325 RMD-PT pairs)

```bash
python documentation/brgi_cisn_decision_map.py \
  --brgi_csv model/Bayesian/forecast/brgi/ranked_pairs.csv \
  --cisn_csv model/Bayesian/forecast/cisn/pt_spillover_value.csv \
  --edges_csv model/Bayesian/forecast/cisn/spillover_network_edges.csv \
  --out_dir model/Bayesian/forecast/brgi_cisn_decision_map \
  --top_k 20 \
  --gamma 0.5 \
  --q_threshold 0.75

python documentation/brgi_cisn_nature_composite.py \
  --priority_csv model/Bayesian/forecast/brgi_cisn_decision_map/brgi_cisn_priority_matrix.csv \
  --out_dir model/Bayesian/forecast/brgi_cisn_decision_map \
  --top_k 20
```

**Validation checklist:**
- [ ] `brgi_cisn_priority_matrix.csv` exists (325 rows)
- [ ] `brgi_cisn_priority_matrix.md` exists (human-readable table)
- [ ] `brgi_cisn_decision_summary.md` exists
- [ ] `Fig_Panel_A_BRGI_CISN_Scatter_Marginals.png` exists (publication-ready)
- [ ] `Fig_Panel_B_TopPriority_ClevelandDot.png` exists
- [ ] Output confirms: 9 Strategic Priority, 73 Targeted Priority, 75 Platform Opportunity, 168 Watch

---

### Step 7 — Manuscript Outputs

```bash
python documentation/build_manuscript_outputs.py
```

**Validation checklist:**
- [ ] 6 PNG/PDF figures in `documentation/`
- [ ] 3 CSV/MD tables in `documentation/`
- [ ] `documentation/README_manuscript_outputs.md` exists
- [ ] `documentation/manuscript_outputs_manifest.json` exists

---

## Customisation Reference

### smoothing.py

```bash
python smoothing.py \
  --input_csv data/data.csv \
  --output_csv data/sm_data.csv \
  --alpha 0.10 \             # level smoothing (0–1)
  --beta 0.05 \              # trend smoothing (0–1)
  --mode des \               # des = Holt's; copy = clean only
  --gemini_api_key KEY \     # override .env
  --no_gemini \              # skip Gemini audit; hard clinical blacklist still runs
  --skip_validation \        # bypass all validation
  --validation_report PATH   # write JSON audit log
```

### forecast.py

```bash
python forecast.py \
  --model_path model/Bayesian/o_model.pt \
  --data_csv data/sm_data.csv \
  --nodes_csv data/data.csv \
  --graph_csv data/graph_sparse.csv \
  --output_dir model/Bayesian \
  --use_tdb_input \
  --plot_start_year 2007 \   # first historical year shown on plots
  --smooth_alpha 0.10 \      # EWMA plot smoothing
  --mc_runs 10 \             # MC-dropout forward passes
  --normalize_scope global \ # global | panel | none
  --max_pts_per_plot 12 \    # max PT lines per RMD plot
  --gemini_api_key KEY \
  --no_gemini \
  --skip_validation \
  --validation_report PATH
```

### make_sparse_graph.py

```bash
python make_sparse_graph.py \
  --data_csv data/sm_data.csv \
  --out_csv data/graph_sparse.csv \
  --top_k 8 \                    # PT edges per RMD
  --window 96 \                  # recent months used for correlation
  --max_lag 6 \                  # max PT lag in months
  --min_corr 0.05 \              # minimum correlation threshold
  --train_end_date 2019-04       # restrict to training period (CRITICAL)
```

---

## Expected Runtimes

| Step | GPU (CUDA) | CPU only |
|------|-----------|---------|
| 0a Gemini validation | ~30 s | ~30 s |
| 0b Smoothing | <1 min | <1 min |
| 0c Graph generation (training period only) | <1 min | <1 min |
| 1 Validation training (500 epochs, date-based split) | ~5 min | ~20 min |
| 2 Production training (500 epochs) | ~3 min | ~12 min |
| 3 Forecast + 48 plots | ~2 min | ~5 min |
| 4 Extract test predictions | <1 min | <1 min |
| 5 Baseline comparison (11 models) | ~5 min | ~10 min |
| 6 BRGI+CISN decision map | ~2 min | ~2 min |
| 7 Manuscript outputs | ~2 min | ~2 min |
| **Total** | **~22 min** | **~55 min** |

---

## Troubleshooting

**Gemini quota exceeded (429)**
The daily free-tier cap is exhausted. The quota resets at midnight Pacific time. Enable billing on the Google Cloud project for uninterrupted use. `--no_gemini` is acceptable for offline/exploratory preprocessing because hard clinical constraints still run, but Gemini-only temporal and magnitude checks are skipped.

**`google-genai` not found**
```bash
pip install google-genai python-dotenv
```

**GEMINI_API_KEY not loaded**
Check that `.env` exists in the repo root:
```bash
cat /home/sahsan03/B-MTGNN/BR_MTGNN/.env
# Expected: GEMINI_API_KEY=AIzaSy...
```

**GPU out of memory**
```bash
python train_test.py --batch_size 8 --device cuda:1
# or switch to CPU:
python train_test.py --device cpu
```

**`model/Bayesian/o_model.pt` not found**
Step 2 (`train.py`) must complete before Step 3. Verify Step 1 saved `model/Bayesian/model.pt` first.

**Forecast shape mismatch**
Ensure `--seq_out_len 36` is consistent across Steps 1, 2, and 3.

**Missing `data/sm_data.csv` or `data/graph_sparse.csv`**
```bash
python smoothing.py
python make_sparse_graph.py
```

---

## Output Reference

| File | Step | Description |
|------|------|-------------|
| `model/Bayesian/validation_report.json` | 0a/0b | Hard-constraint and Gemini validation audit log |
| `data/sm_data.csv` | 0b | Smoothed, validated time series with `Month-Year` first column |
| `data/graph_sparse.csv` | 0c | Undirected RMD–PT adjacency (top-8 per RMD, training data only) |
| `model/Bayesian/model.pt` | 1 | Best validation checkpoint (date-based split) |
| `model/Bayesian/hp.txt` | 1 | Hyperparameters (JSON) with date split fields |
| `model/Bayesian/o_model.pt` | 2 | Production model weights (98% data) |
| `model/Bayesian/forecast/forecast_36m.csv` | 3 | 36-month predictions (long format) |
| `model/Bayesian/forecast/plots/*.png` | 3 | 48 RMD forecast plots with 95% PIs (600 DPI) |
| `documentation/br_mtgnn_full_calibrated_test_predictions.csv` | 4 | Test window predictions (20 months × 137 nodes) |
| `documentation/Table 1 - Baseline comparison.csv` | 5 | 11-model ranking on same test window |
| `documentation/Figure 3*.png` | 5 | Horizon-block model performance |
| `model/Bayesian/forecast/brgi/ranked_pairs.csv` | 6 | BRGI scores (384 RMD-PT pairs) |
| `model/Bayesian/forecast/cisn/pt_spillover_value.csv` | 6 | CISN spillover values (72 PTs) |
| `model/Bayesian/forecast/brgi_cisn_decision_map/brgi_cisn_priority_matrix.csv` | 6 | Merged priority matrix (325 pairs) |
| `model/Bayesian/forecast/brgi_cisn_decision_map/Fig_Panel_A_BRGI_CISN_Scatter_Marginals.png` | 6 | Nature-style Panel A (publication-ready) |
| `model/Bayesian/forecast/brgi_cisn_decision_map/Fig_Panel_B_TopPriority_ClevelandDot.png` | 6 | Nature-style Panel B (publication-ready) |
| `documentation/Figure 1-6.png` | 7 | All manuscript figures |

---

## References

- **Validation methodology:** `gemini_validator.py` docstring
- **Baseline comparison:** `Comparative_Evaluation/BASELINE_COMPARE_README.md`
- **BRGI calculation:** `documentation/BRGI_REPORT.md`
- **CISN spillover network:** `documentation/CISN_ANALYSIS_REPORT.md`
- **Full system reference:** `documentation/PROJECT_COMPLETE_REFERENCE.md`
