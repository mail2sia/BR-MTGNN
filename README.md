# BR-MTGNN: Bayesian Residual-Calibrated Temporal Graph Forecasting for RMD/PT Results

**Bayesian Residual-Calibrated Multivariate Temporal Graph Neural Network (BR-MTGNN)** forecasts rare mental disease (RMD) and pertinent technology (PT) trajectories over a 36-month horizon, then converts the forecasts into BRGI/CISN decision-support outputs for manuscript tables and figures.

Current result set rebuilt on **2026-06-05** (date-based split and decision map regeneration) from the pipeline under `B-MTGNN/BR_MTGNN`.

## Current Outputs

Forecast horizon: **2026-01 to 2028-12**

Core result directories:

| Output | Path | Status |
|---|---|---|
| Forecast data and plots | `model/Bayesian/forecast/` | Built |
| BRGI ranked gaps | `model/Bayesian/forecast/brgi/ranked_pairs.csv` | Built, 384 RMD-PT pairs |
| CISN spillover values | `model/Bayesian/forecast/cisn/` | Built, 72 PTs |
| BRGI+CISN report | `model/Bayesian/forecast/brgi_cisn_report/` | Built |
| Decision map | `model/Bayesian/forecast/brgi_cisn_decision_map/` | Built, 325 merged pairs |
| BRGI+CISN publication panels (A/B) | `model/Bayesian/forecast/brgi_cisn_decision_map/` | Built (`Fig_Panel_A_*`, `Fig_Panel_B_*`) |
| RMD-PT bubble chart | `model/Bayesian/forecast/rmd_pt_bubble/` | Built, 70 PTs plotted |
| RMD-PT heatmap grid | `model/Bayesian/forecast/rmd_pt_visualizations/` | Built, 43 RMDs x 30 PTs |
| Manuscript tables and figures | `documentation/` | Tables 1-3 and Figures 4-6 built |

Note: `documentation/build_manuscript_outputs.py` currently generates Tables 1-3 and Figures 4-6. Figure 1 and Figure 2 are skipped because their helper functions are not defined in that script. Figure 3 is skipped unless `documentation/horizon_block_metrics.csv` is supplied.

## Canonical Test Metrics

Source: `model/Bayesian/metrics/test_metrics.csv`

| Split | Group | RAE | RSE | Corr | Coverage |
|---|---|---:|---:|---:|---:|
| test | Overall | 0.232 | 0.471 | 0.890 | 0.936 |
| test | NoM | 0.232 | 0.471 | 0.890 | 0.936 |
| test | TDB | 0.418 | 0.639 | 0.846 | 0.899 |
| test | PT_NoM | 0.161 | 0.419 | 0.911 | 0.957 |

Metric interpretation:

- `RAE`: relative absolute error, lower is better.
- `RSE`: relative squared error, lower is better.
- `Corr`: Pearson correlation, higher is better.
- `Coverage`: empirical coverage of the 95% prediction interval.

Validation and production metric files are retained for audit only:

- `model/Bayesian/metrics/validation_metrics.csv`
- `model/Bayesian/metrics/production_metrics.csv`

## Baseline Comparison

Source: `documentation/Table 1 - Baseline comparison.csv` and `Comparative_Evaluation/baseline_comparison_all_results.csv`

| Model | RAE | RSE | Corr | Coverage |
|---|---:|---:|---:|---:|
| BR-MTGNN | 0.251 | 0.428 | 0.913 | 0.950 |
| DCRNN | 0.255 | 0.431 | 0.000 | NA |
| PatchTST | 0.355 | 0.674 | 0.145 | NA |
| TFT | 0.356 | 0.642 | 0.190 | NA |
| AGCRN | 0.517 | 0.935 | 0.120 | NA |
| TimesFM | 0.715 | 1.405 | 0.102 | NA |
| LSTM_U | 1.235 | 2.252 | 0.244 | NA |
| Transformer_U | 1.272 | 1.831 | 0.020 | NA |
| LSTM_M | 1.867 | 2.675 | 0.181 | NA |
| Transformer_M | 2.746 | 3.931 | 0.082 | NA |
| Prophet | 4.064 | 7.707 | 0.000 | NA |
| MTGNN | 6.556 | 5.687 | 0.000 | NA |
| BMTGNN | 6.905 | 5.932 | 0.000 | NA |

Only BR-MTGNN reports conformal coverage in the current baseline table.

## BRGI Results

BRGI is the Burden-Readiness Gap Index. In the current rebuilt output it is derived from `model/Bayesian/forecast/gap_monthly.csv` by clipping negative RMD-PT gaps to zero and averaging positive gaps per RMD-PT pair.

Source: `model/Bayesian/forecast/brgi/ranked_pairs.csv`

Rows: **384** RMD-PT pairs

Top BRGI gaps from `documentation/Table 2 - BRGI top gaps.csv`:

| Rank | RMD | PT | BRGI Score |
|---:|---|---|---:|
| 1 | Hallucinogen Persisting Perception Disorder | Memory Retrieval Techniques | 0.6770 |
| 2 | Antisocial Personality Disorder | Memory Retrieval Techniques | 0.6762 |
| 3 | Postpartum Psychosis | Digital Imaging Technologies For Brain Scanning | 0.6626 |
| 4 | Antisocial Personality Disorder | Lithium Therapy | 0.6575 |
| 5 | Factitious Disorder | Cognitive Enhancement Programs | 0.6392 |
| 6 | Factitious Disorder | Blockchain For Data Protection | 0.6304 |
| 7 | Chronic Traumatic Encephalopathy | Blockchain For Data Protection | 0.6282 |
| 8 | Dissociative Fugue | Vocal And Speech Biomarkers | 0.6220 |

## CISN Results

CISN is the Cross-RMD Intervention Spillover Network. It estimates cross-disorder PT portfolio value from time-series RMD similarity and the RMD-PT association matrix.

Source: `model/Bayesian/forecast/cisn/pt_spillover_value.csv`

Summary from `model/Bayesian/forecast/cisn/validation_report.json`:

| Item | Value |
|---|---:|
| RMDs analyzed | 48 |
| PTs ranked | 72 |
| RMD-PT mapped pairs | 384 |
| Similarity source | data-driven time-series similarity |
| Spillover normalization | row |
| Top PT | Electroconvulsive Therapy |
| Top PT spillover value | 6.1359 |
| Top PT linked RMDs | 19 |

Top CISN PTs from `documentation/Table 3 - CISN PT ranking.csv`:

| Rank | PT | Spillover Value | Classification |
|---:|---|---:|---|
| 1 | Electroconvulsive Therapy | 6.1359 | Generalist |
| 2 | Naltrexone | 4.8091 | Generalist |
| 3 | Narrative Exposure Therapy | 3.9825 | Generalist |
| 4 | Stimulant And Wake Promoting Agents | 3.9649 | Generalist |
| 5 | Mood Stabilizers | 3.8725 | Generalist |
| 6 | Animal Assisted Therapy | 3.6829 | Bridge |
| 7 | Hypnotherapy | 3.6409 | Bridge |
| 8 | Metaverse | 3.5368 | Bridge |

## BRGI+CISN Decision Map

Source: `model/Bayesian/forecast/brgi_cisn_decision_map/brgi_cisn_priority_matrix.csv`

Decision-map summary from `model/Bayesian/forecast/brgi_cisn_decision_map/brgi_cisn_decision_manifest.json`:

| Priority Label | Count |
|---|---:|
| Strategic Priority | 9 |
| Targeted Priority | 73 |
| Platform Opportunity | 75 |
| Watch | 168 |

Rows loaded and merged:

| Input | Rows |
|---|---:|
| BRGI rows loaded | 384 |
| CISN rows loaded | 72 |
| Merged decision rows | 325 |

Top decision-map priorities:

| Rank | RMD | PT | BRGI | Spillover Norm | Final Priority | Label |
|---:|---|---|---:|---:|---:|---|
| 1 | Hallucinogen Persisting Perception Disorder | Memory Retrieval Techniques | 0.6771 | 0.0135 | 1.0068 | Targeted Priority |
| 2 | Antisocial Personality Disorder | Lithium Therapy | 0.6575 | 0.0732 | 1.0067 | Targeted Priority |
| 3 | Antisocial Personality Disorder | Memory Retrieval Techniques | 0.6762 | 0.0135 | 1.0055 | Targeted Priority |
| 4 | Selective Mutism | Transcranial Magnetic Stimulation | 0.5358 | 0.5346 | 1.0029 | Strategic Priority |
| 5 | Selective Mutism | Transcranial Direct Current Stimulation | 0.5841 | 0.3011 | 0.9926 | Targeted Priority |
| 6 | Postpartum Psychosis | Digital Imaging Technologies For Brain Scanning | 0.6626 | 0.0223 | 0.9896 | Targeted Priority |
| 7 | Selective Mutism | Cognitive Behavioral Therapy | 0.5398 | 0.4768 | 0.9874 | Strategic Priority |
| 8 | Factitious Disorder | Blockchain For Data Protection | 0.6304 | 0.1194 | 0.9868 | Targeted Priority |

## Manuscript Deliverables

Generated under `documentation/`:

| File | Description |
|---|---|
| `Table 1 - Baseline comparison.csv` / `.md` | Baseline comparison table |
| `Table 2 - BRGI top gaps.csv` / `.md` | Top 20 BRGI RMD-PT gaps |
| `Table 3 - CISN PT ranking.csv` / `.md` | Top 20 PTs by CISN spillover value |
| `Figure 4 - BRGI gap ranking.png` / `.pdf` | BRGI bar chart |
| `Figure 5 - CISN spillover specialist-generalist map.png` / `.pdf` | CISN PT map |
| `Figure 6 - BRGI + CISN decision quadrant.png` / `.pdf` | BRGI+CISN quadrant |
| `README_manuscript_outputs.md` | Generated manuscript-output inventory |
| `manuscript_outputs_manifest.json` | Machine-readable generated-output manifest |

Generated under `model/Bayesian/forecast/`:

| Directory | Key files |
|---|---|
| `brgi/` | `ranked_pairs.csv` |
| `cisn/` | `pt_spillover_value.csv`, `spillover_matrix.csv`, `rmd_pt_association_matrix.csv`, `cisn_adjusted_brgi.csv`, `validation_report.json`, PDF/PNG figures |
| `brgi_cisn_report/` | `priority_matrix.csv`, `pt_spillover_value_classified.csv`, `top_priority_summary.md`, PDF/PNG figures |
| `brgi_cisn_decision_map/` | `brgi_cisn_priority_matrix.csv`, `.md`, `Fig_BRGI_CISN_DecisionMap_Composite.pdf/.png`, `Fig_Panel_A_BRGI_CISN_Scatter_Marginals.pdf/.png`, `Fig_Panel_B_TopPriority_ClevelandDot.pdf/.png`, manifest |
| `rmd_pt_bubble/` | `rmd_pt_bubble_chart.png`, `.pdf`, chart data, quadrant analysis, abbreviation legend |
| `rmd_pt_visualizations/` | `rmd_pt_heatmap_grid.png`, `.pdf`, priority and gap matrices |


## Validation and Clinical Safety Firewall

Preprocessing now uses a two-gate validation system in `gemini_validator.py`:

1. **Hard clinical constraints** run first and do not require Gemini. These rules zero known clinical hallucinations before smoothing, graph construction, training, or plotting.
2. **Gemini grounded audit** runs next when `GEMINI_API_KEY` is available. It handles nuanced temporal anachronism, magnitude plausibility, and duplicate-sequence review.

### Caching Behavior

By default, `smoothing.py` reuses cached audits from `data/validation_cache.json` to avoid redundant API calls. Use `--no_cache` to force a fresh audit:

- **First run or after rule updates:** Use `--no_cache` to ensure hard constraints and Gemini checks apply fresh
- **Subsequent runs (offline/exploratory):** Omit `--no_cache` to reuse cache (faster)
- **Production submission:** Use `--no_cache` to eliminate stale cache artifacts

The validation cache hash includes `_VALIDATION_SCHEMA_VERSION`, so major rule updates automatically invalidate stale cached results.

### Current Hard Constraints

- `PT_Clozapine Protocol_NoM` is globally zeroed because Clozapine is treated as medically inappropriate data noise in this dataset.
- Additional RMD-PT safety-firewall rules cover implausible pairings such as Stendhal Syndrome or Koro with high-intensity psychiatric protocols or recent digital technologies.
- Rules are re-applied with `--no_cache` to ensure alignment with corrected date-based split and scaler fitting (training data only).

`data/sm_data.csv` now preserves `Month-Year` as the first column. Expected shape after smoothing is **264 rows x 190 columns**.

## Temporal Validation Protocol (Date-Based Split)

The pipeline now uses **date-based splitting** instead of index-based splitting to eliminate target-period leakage:

- **Graph adjacency** computed from training period only (Jan 2004 – Apr 2019, 184 rows)
- **Scaler** fitted on training data only (prevents val/test influence on normalization)
- **Training windows:** Jan 2004 – Apr 2019 (184 sliding windows)
- **Validation windows:** May 2019 – Sep 2023 (60 sliding windows)
- **Test windows:** Oct 2023 – Dec 2025 (20 sliding windows)
- **Split criterion:** Each window assigned to train/val/test based on its **last forecast target month**, not by index

This ensures perfect temporal boundaries with zero target-period overlap between splits.

## Rebuild Commands

Run from `B-MTGNN/BR_MTGNN` unless an absolute path is shown.

### 1. Validate Raw Data (Optional Standalone)

```bash
python gemini_validator.py \
  --input_csv data/data.csv \
  --output_csv data/data_validated.csv \
  --report_json model/Bayesian/validation_report.json
```

### 2. Preprocess and Graph (Date-Based Split)

```bash
python smoothing.py \
  --input_csv data/data.csv \
  --output_csv data/sm_data.csv \
  --alpha 0.10 --beta 0.05 --mode des \
  --no_cache \
  --validation_report model/Bayesian/validation_report.json

python make_sparse_graph.py \
  --data_csv data/sm_data.csv \
  --out_csv data/graph_sparse.csv \
  --top_k 8 --window 96 --max_lag 6 --min_corr 0.05 \
  --train_end_date 2019-04
```

**Critical flags explained:**
- `--no_cache`: Force fresh validation audit (don't reuse stale cache). Ensures hard clinical constraints and Gemini checks align with corrected date-based split.
- `--train_end_date 2019-04`: Restrict graph adjacency to training period only (Jan 2004–Apr 2019, 184 rows). Prevents validation/test signal from influencing RMD-PT relationships.

### 3. Validation Training (Date-Based Split)

```bash
python train_test.py \
  --device cuda:1 \
  --output_dir model/Bayesian \
  --seq_in_len 10 \
  --seq_out_len 36 \
  --use_tdb_input \
  --epochs 500 \
  --patience 400 \
  --train_end_date 2019-04 \
  --val_end_date 2023-09
```

**Critical:** Date-based split eliminates target-period leakage:
- Training: Jan 2004 – Apr 2019 (184 sliding windows)
- Validation: May 2019 – Sep 2023 (60 sliding windows)
- Test: Oct 2023 – Dec 2025 (20 sliding windows)

### 4. Production Training

```bash
python train.py \
  --device cuda:1 \
  --output_dir model/Bayesian \
  --seq_in_len 10 \
  --seq_out_len 36 \
  --use_tdb_input \
  --epochs 500 \
  --lr 1e-5 \
  --lambda_delta 0.02 \
  --lambda_start 0.20 \
  --lambda_horizon 0.05
```

Production training uses all 264 months (0.98/0.01 ratio-based split) to estimate final weights.

### 5. Forecast + Plots

```bash
python forecast.py \
  --model_path model/Bayesian/o_model.pt \
  --data_csv data/sm_data.csv \
  --nodes_csv data/data.csv \
  --graph_csv data/graph_sparse.csv \
  --output_dir model/Bayesian \
  --use_tdb_input \
  --plot_start_year 2007 \
  --smooth_alpha 0.10
```

### 6. Baseline Comparison

```bash
python Comparative_Evaluation/run_all_baselines.py
python Comparative_Evaluation/generate_comparison_table.py
```

The rebuilt manuscript adapter is `documentation/baseline_metrics_for_manuscript.csv`; it mirrors the baseline result columns expected by `documentation/build_manuscript_outputs.py`.

### 7. Build BRGI Ranked Pairs From Current Gap Forecast

```bash
python scripts/build_brgi_ranked_pairs.py
```

This helper derives `model/Bayesian/forecast/brgi/ranked_pairs.csv` from `model/Bayesian/forecast/gap_monthly.csv` by averaging `max(Gap_RMD_minus_PT, 0)` for each RMD-PT pair and ranking by `BRGI_Score`.

### 8. Build CISN and BRGI+CISN Reports

```bash
python brgi_spillover.py \
  --forecast_csv model/Bayesian/forecast/data/forecast_NoM_mean.csv \
  --history_csv data/sm_data.csv \
  --nodes_csv data/data.csv \
  --mapping_csv model/Bayesian/forecast/plots_data/selected_pts_per_rmd.csv \
  --brgi_csv model/Bayesian/forecast/brgi/ranked_pairs.csv \
  --out_dir model/Bayesian/forecast/cisn \
  --similarity_source timeseries \
  --gamma 0.5 \
  --top_k 30

python brgi_cisn_report.py \
  --brgi_csv model/Bayesian/forecast/brgi/ranked_pairs.csv \
  --cisn_csv model/Bayesian/forecast/cisn/pt_spillover_value.csv \
  --cisn_brgi_csv model/Bayesian/forecast/cisn/cisn_adjusted_brgi.csv \
  --edges_csv model/Bayesian/forecast/cisn/spillover_network_edges.csv \
  --out_dir model/Bayesian/forecast/brgi_cisn_report \
  --top_k 20 \
  --gamma 0.5

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

`documentation/brgi_cisn_nature_composite.py` generates publication-ready panels:
- **Panel A:** BRGI-vs-Spillover scatter with marginal KDEs, priority-region shading, and external legend (Fig_Panel_A_BRGI_CISN_Scatter_Marginals.pdf/.png)
- **Panel B:** Cleveland dot plot of top priorities (default `top_k=20`) with external legend and no panel label marker (Fig_Panel_B_TopPriority_ClevelandDot.pdf/.png)

Both panels use Nature-style formatting suitable for high-impact journal submission.

### 9. Build RMD-PT Visuals

```bash
python documentation/rmd_pt_bubble_chart.py \
  --brgi_csv model/Bayesian/forecast/brgi_cisn_decision_map/brgi_cisn_priority_matrix.csv \
  --forecast_csv model/Bayesian/forecast/pt_forecast_metadata.csv \
  --output_dir model/Bayesian/forecast/rmd_pt_bubble

python documentation/rmd_pt_heatmap_grid.py
```

### 10. Build Manuscript Tables and Figures

```bash
python documentation/build_manuscript_outputs.py \
  --repo_root . \
  --data_csv data/sm_data.csv \
  --forecast_dir model/Bayesian/forecast \
  --baseline_csv documentation/baseline_metrics_for_manuscript.csv \
  --out_dir documentation \
  --top_k 20 \
  --gamma 0.5
```

## Data and Signal Layout

The input time series use these column conventions:

| Pattern | Meaning | Forecast target |
|---|---|---|
| `RMD_*_NoM` | RMD mention volume | Yes |
| `RMD_*_NoP` | RMD number of patients/deaths | Yes, folded into TDB in default mode |
| `PT_*_NoM` | PT mention volume | Yes |
| `GLOBAL_*`, `WAR_*` | Exogenous/global covariates | No, broadcast as inputs |
| `Month-Year` | Monthly time index | No |

Default training uses `--use_tdb_input`, where the model predicts a combined RMD TDB signal (`NoM + NoP`) and PT NoM trajectories.

## Model and Uncertainty

### Architecture

Key model features:

- Dilated temporal convolution plus graph convolution.
- RMD-PT graph from `data/graph_sparse.csv`.
- Persistence residual: `forecast = last_observed_value + learned_delta`.
- MC-Dropout uncertainty plus conformal residual calibration.
- Chronological train/validation/test split with no random shuffling.

### BR-MTGNN: Technical Definition

**BR-MTGNN** = **Bayesian Residual-Calibrated Multivariate Temporal Graph Neural Network**

The method combines:

1. **Bayesian component:** MC-Dropout (Gal & Ghahramani, 2016) for epistemic uncertainty estimation
   - Multiple forward passes during inference with dropout active
   - Computes mean and std from MC samples

2. **Residual-Calibrated component:** One-time conformal quantile correction
   - Conformal quantile computed from validation prediction **residuals**: `q = quantile(|y_true - y_pred|, 1-α)`
   - Applied uniformly to all test predictions
   - **Not adaptive or online**—quantile fixed after validation

### Prediction Intervals

Conformal calibrated prediction interval formula:

```text
lower = mean - 1.96 * mc_std - q_residual
upper = mean + 1.96 * mc_std + q_residual
```

Where:
- `mean`: Ensemble mean from MC-dropout
- `mc_std`: Ensemble standard deviation from MC-dropout  
- `q_residual`: Conformal quantile computed from validation residuals

Separate calibration quantiles are stored for RMD, PT, and NoP/TDB groups in `model/Bayesian/metadata.json`.

## Scientific Interpretation

BRGI and CISN are decision-support indices. They identify forecasted unmet-need gaps and cross-RMD PT portfolio value. They do not establish clinical efficacy, causality, or treatment transferability. Use the outputs as prioritization and hypothesis-generation tools.

## Troubleshooting

Stale validation cache:

- If validation results seem outdated after rule updates, use `--no_cache` in `smoothing.py` to force a fresh audit.
- The cache includes version hashing (`_VALIDATION_SCHEMA_VERSION`), so major rule changes should invalidate old results automatically.

Missing `nature_style_utils.py`:

- The current pipeline expects `nature_style_utils.py` at repo root for BRGI/CISN plotting helpers.

Missing `model/Bayesian/o_model.pt`:

- Run `train.py` after `train_test.py`; `forecast.py` requires the production checkpoint.

Missing `model/Bayesian/forecast/brgi/ranked_pairs.csv`:

- Rebuild it from `model/Bayesian/forecast/gap_monthly.csv` after `forecast.py` completes.

Missing Figure 3 in `documentation/`:

- Provide `documentation/horizon_block_metrics.csv`, then rerun `documentation/build_manuscript_outputs.py`.

Gemini quota or API errors:

- Set `GEMINI_API_KEY` in `.env` for grounded temporal/anachronism auditing.
- `--no_gemini` still applies the hard clinical blacklist, but skips Gemini-only anachronism and magnitude review; use it only for exploratory or offline preprocessing.
- For production submission, use `--no_cache` with Gemini key to ensure fresh, comprehensive audit aligned with date-based split.

## Citation
```
