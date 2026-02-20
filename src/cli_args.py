import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PyTorch Time series forecasting")
    parser.add_argument(
        "--data",
        type=str,
        default="./data/sm_data_g.csv",
        help="Path to the time-series data file (CSV or TSV). Default uses spike-cleaned smoothed dataset.",
    )
    parser.add_argument(
        "--metrics_json",
        type=str,
        default=None,
        help='Path to a metrics_*.json file whose embedded "args" dict will be applied to this run.',
    )
    parser.add_argument(
        "--use_best_tune",
        action="store_true",
        help="Apply args from a prior metrics_validation.json (uses --best_tune_path).",
    )
    parser.add_argument(
        "--best_tune_path",
        type=str,
        default="runs/tune_20260206-120354_01/metrics_validation.json",
        help="Path to a metrics_validation.json produced by targeted tuning.",
    )
    parser.add_argument(
        "--skip_ckpt_infer",
        action="store_true",
        help="If set, do not infer hyperparameters from checkpoint shapes (sets BMTGNN_SKIP_CKPT_INFER=1).",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Force training mode even when metrics_json has eval_only=true (sets eval_only=False).",
    )
    parser.add_argument(
        "--no_trainer_mode",
        action="store_true",
        help="Disable Trainer path even if trainer_mode is set in JSON/args (forces trainer_mode=False).",
    )
    parser.add_argument(
        "--runlog",
        action="store_true",
        help="Write compact JSONL diagnostics while training/evaluating",
    )
    parser.add_argument("--log_interval", type=int, default=2000, metavar="N", help="report interval")
    parser.add_argument(
        "--show_gpu_titles",
        action="store_true",
        help="Print GPU PIDs with process titles via nvidia-smi + ps (Linux only).",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="model/Bayesian/model.pt",
        help="path to save the final model",
    )
    parser.add_argument(
        "--ensemble_ckpts",
        type=int,
        default=0,
        help="Number of rolling ensemble checkpoints to keep (0 disables).",
    )
    parser.add_argument(
        "--ensemble_every",
        type=int,
        default=20,
        help="Save ensemble checkpoints every N epochs (requires --ensemble_ckpts > 0).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="runs",
        help="Root folder to save metrics/plots for each run",
    )
    parser.add_argument(
        "--run_tag",
        type=str,
        default="",
        help="Optional unique tag for this run; used to place outputs",
    )
    parser.add_argument("--optim", type=str, default="adamw")
    parser.add_argument("--L1Loss", action="store_true", help="use L1 loss in training")
    parser.add_argument(
        "--normalize",
        type=int,
        default=2,
        help="Normalization mode: 3=per-node z-score (recommended for heterogeneous nodes)",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="")
    parser.add_argument("--gcn_true", action="store_true", help="Enable graph convolution layer")
    parser.add_argument(
        "--buildA_true",
        action="store_true",
        help="Enable construction of adaptive adjacency matrix",
    )
    parser.add_argument(
        "--gcn_depth",
        type=int,
        default=2,
        help="graph convolution depth (3 for stronger graph propagation)",
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=95,
        help="number of nodes/variables (auto-detected)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.05,
        help="dropout rate (0.0 recommended for stable training)",
    )
    parser.add_argument(
        "--subgraph_size",
        type=int,
        default=8,
        help="Size of subgraphs (0 to disable). Recommended: 20-40 for large graphs.",
    )
    parser.add_argument("--node_dim", type=int, default=40, help="dim of nodes")
    parser.add_argument("--dilation_exponential", type=int, default=2, help="dilation exponential")
    parser.add_argument(
        "--conv_channels",
        type=int,
        default=16,
        help="convolution channels (96 for optimized config)",
    )
    parser.add_argument(
        "--exclude-names",
        type=str,
        default="",
        help="Comma-separated list of series names to exclude from training/eval",
    )
    parser.add_argument(
        "--residual_channels",
        type=int,
        default=32,
        help="residual channels (96 for optimized config)",
    )
    parser.add_argument(
        "--skip_channels",
        type=int,
        default=64,
        help="skip channels (192 for optimized config)",
    )
    parser.add_argument(
        "--end_channels",
        type=int,
        default=128,
        help="end channels (384 for optimized config)",
    )
    parser.add_argument(
        "--allow_wide_end",
        action="store_true",
        help="Allow 1024 end channels in search space (may increase memory).",
    )
    parser.add_argument("--in_dim", type=int, default=1, help="inputs dimension")
    parser.add_argument(
        "--seq_in_len",
        type=int,
        default=120,
        help="input sequence length (24 months = 2 years context)",
    )
    parser.add_argument(
        "--seq_out_len",
        type=int,
        default=36,
        help="output sequence length (36 months = 3 years forecast)",
    )
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument(
        "--layers",
        type=int,
        default=8,
        help="number of layers (7 for deeper receptive field, improved accuracy)",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="batch size (16 for more stable gradients)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0004,
        help="learning rate (0.0003 for finer gradient updates, higher accuracy)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.00001,
        help="weight decay rate (5e-7 for model flexibility)",
    )

    parser.add_argument(
        "--clip",
        type=float,
        default=5.0,
        help="Gradient norm clip (20.0 for smoother convergence)",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=None,
        help="Alias for --clip (if set, overrides --clip value)",
    )

    parser.add_argument(
        "--propalpha",
        type=float,
        default=0.05,
        help="prop alpha (0.08 for stronger graph influence)",
    )
    parser.add_argument(
        "--tanhalpha",
        type=float,
        default=3,
        help="tanh alpha (4 for sharper graph learning)",
    )

    parser.add_argument("--epochs", type=int, default=100, help="epochs (400 for better convergence)")
    parser.add_argument("--seed", type=int, default=None, help="random seed (overrides fixed_seed)")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision (AMP)")
    parser.add_argument(
        "--cudnn_benchmark",
        action="store_true",
        help="Enable cuDNN benchmark (faster, non-deterministic)",
    )
    parser.add_argument("--mc_runs", type=int, default=10, help="MC-Dropout samples at eval")
    parser.add_argument(
        "--test_eval_mode",
        type=str,
        default="sliding",
        help="Testing eval mode: sliding (default), batch, or both",
    )
    parser.add_argument(
        "--vectorized_mc",
        action="store_true",
        default=False,
        help="Vectorize MC-Dropout by repeating the batch (ON = faster, higher peak memory). OFF = looped passes (safer).",
    )
    parser.add_argument(
        "--no_vectorized_mc",
        action="store_true",
        help="Explicitly disable vectorized MC (overrides --vectorized_mc).",
    )

    parser.add_argument(
        "--trainer_mode",
        action="store_true",
        default=True,
        help="Use the Trainer-based path (skips random search) and wire probabilistic losses if requested.",
    )
    parser.add_argument(
        "--mc_vec_max_elems",
        type=int,
        default=120_000_000,
        help="If vectorized MC would create (mc_runs*B*C*N*T) > this element count, fallback to looped MC automatically.",
    )
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help="Accumulate gradients over N mini-batches before optimizer step (simulated larger batch).",
    )
    parser.add_argument(
        "--fused_optim",
        action="store_true",
        help="Use fused AdamW optimizer (CUDA fused=True) instead of the custom Optim wrapper",
    )
    parser.add_argument(
        "--graph",
        type=str,
        default="./data/graph_square.csv",
        help="Path to the predefined adjacency matrix (square CSV or RMD-PT edge list).",
    )
    parser.add_argument(
        "--persist_mc",
        type=str,
        default="",
        help="Path prefix to persist per-slide MC samples (npz files)",
    )
    parser.add_argument(
        "--persist_mc_slides",
        type=str,
        default="",
        help='Comma-separated slide start indices to persist (e.g. "10,46"). Empty = persist all when --persist_mc supplied',
    )
    parser.add_argument(
        "--debug_rf",
        action="store_true",
        help="Print receptive field and kernel widths per layer and exit",
    )
    parser.add_argument(
        "--ckpt_to_compare",
        type=str,
        default="",
        help="Path to checkpoint file to compare shapes against current model",
    )
    parser.add_argument(
        "--ckpt_compare_csv",
        type=str,
        default="",
        help="If provided, write CSV comparing checkpoint vs model parameter shapes",
    )
    parser.add_argument(
        "--use_cached_hp",
        action="store_true",
        default=False,
        help="Skip hyperparameter search; use cached best hyperparameters from model/Bayesian/hp.txt and existing checkpoint",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="With --use_cached_hp, only evaluate the cached checkpoint without any training/search",
    )
    parser.add_argument(
        "--search_trials",
        type=int,
        default=120,
        help="Number of random hyperparameter samples during search",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=80,
        help="Epoch patience per trial before aborting that trial early (0 disables, 80 recommended)",
    )
    parser.add_argument("--no_plots", action="store_true", help="Disable plotting during evaluation")
    parser.add_argument(
        "--enable_test_plots",
        action="store_true",
        default=True,
        help="Generate validation & testing plots whenever a new global best is found during training (ENABLED by default)",
    )
    parser.add_argument(
        "--no_test_plots",
        dest="enable_test_plots",
        action="store_false",
        help="Disable test plot generation",
    )
    parser.add_argument(
        "--plot_mode",
        type=str,
        default="auto",
        choices=["rmd_only", "all", "top_error", "top_variance", "important", "auto"],
        help="Dynamic node selection for plotting: rmd_only=RMD nodes only, all=plot all nodes, "
        "top_error=top-k nodes by prediction error, top_variance=top-k by variance, "
        "important=combine error+variance+RMD, auto=intelligent selection (default)",
    )
    parser.add_argument(
        "--plot_top_k",
        type=int,
        default=20,
        help="Number of top nodes to plot when using top_error/top_variance/important/auto modes (default: 20)",
    )
    parser.add_argument(
        "--smooth_l1_only",
        action="store_true",
        help="Use only SmoothL1Loss in normalized space (debug)",
    )
    parser.add_argument(
        "--use_weighted_horizon_loss",
        action="store_true",
        help="Use weighted Huber horizon loss (upweights non-zeros & far horizon) instead of composite loss",
    )
    parser.add_argument(
        "--plot_norm_space",
        action="store_true",
        help="Also plot predictions vs truth in normalized (z) space",
    )
    parser.add_argument(
        "--movement_debug",
        action="store_true",
        help="Print movement diagnostics in normalized space (variance, corr)",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.60,
        help="Train ratio for ratio-based chronological split of historical data (2004-2025).",
    )
    parser.add_argument(
        "--valid_ratio",
        type=float,
        default=0.20,
        help="Validation ratio for ratio-based split of historical data (2004-2025). Remainder becomes test set.",
    )
    parser.add_argument(
        "--no_cleanup",
        action="store_true",
        help="Skip automatic cleanup of checkpoints and cache at pipeline start.",
    )
    parser.add_argument(
        "--calibration",
        choices=["none", "val", "test", "both"],
        default="both",
        help=("Per-node linear calibration a*pred+b. " "none=disabled (default); " "val=fit on validation only; " "test=apply cached params on test; " "both=fit on val and apply on test."),
    )
    parser.add_argument(
        "--analysis_log",
        type=str,
        default="./outputs/analysis_v4_safe.jsonl",
        help="Path to a file for detailed JSON analysis logging.",
    )
    parser.add_argument(
        "--y_transform",
        type=str,
        default="none",
        choices=["none", "log1p"],
        help="Target transform to stabilize heavy-tailed sparse series (none recommended).",
    )
    parser.add_argument(
        "--has_header",
        action="store_true",
        help="Input CSV has a header row (use pandas to read). RECOMMENDED: always use --has_header for sm_data_g.csv.",
    )
    parser.add_argument(
        "--drop_first_col",
        action="store_true",
        help="Drop the first column (e.g., date) when loading data.",
    )
    parser.add_argument(
        "--trend_smooth",
        action="store_true",
        help="Enable trend-aware smoothing (Holt trend + smoothed residual).",
    )
    parser.add_argument("--trend_alpha", type=float, default=0.25, help="Holt level smoothing alpha (0-1).")
    parser.add_argument("--trend_beta", type=float, default=0.05, help="Holt trend smoothing beta (0-1).")
    parser.add_argument(
        "--resid_alpha",
        type=float,
        default=0.20,
        help="Residual EMA smoothing alpha (0-1).",
    )
    parser.add_argument(
        "--clip_outliers",
        action="store_true",
        help="Clip extreme outliers per column at specified percentile during data loading.",
    )
    parser.add_argument(
        "--clip_percentile",
        type=float,
        default=95.0,
        help="Percentile threshold for outlier clipping (default: 95.0).",
    )
    parser.add_argument(
        "--aggressive_smooth",
        action="store_true",
        help="Generate and use aggressively smoothed data before training/eval (uses agg_smoothing recipe).",
    )
    parser.add_argument(
        "--aggr_out_prefix",
        type=str,
        default="data/sm_aggr",
        help="Output prefix for aggressive smoothing (writes .csv and .txt).",
    )
    parser.add_argument(
        "--aggr_alpha",
        type=float,
        default=0.01,
        help="Aggressive smoothing alpha for double exponential filter.",
    )
    parser.add_argument(
        "--aggr_beta",
        type=float,
        default=0.01,
        help="Aggressive smoothing beta for double exponential filter.",
    )
    parser.add_argument(
        "--aggr_gauss_sigma",
        type=float,
        default=1.2,
        help="Aggressive smoothing Gaussian sigma (0 disables).",
    )
    parser.add_argument(
        "--aggr_ma_window",
        type=int,
        default=5,
        help="Aggressive smoothing moving-average window (1 disables).",
    )
    parser.add_argument(
        "--aggr_force",
        action="store_true",
        help="Force re-generation of aggressive-smoothed files even if they exist.",
    )
    parser.add_argument(
        "--nonzero_weight",
        type=float,
        default=1.0,
        help="Upweight loss when y_true > 0 (handles sparse signals).",
    )
    parser.add_argument(
        "--rmd_loss_weight",
        type=float,
        default=2.5,
        help="Multiplier for RMD-series loss weight (default: 2.5).",
    )
    parser.add_argument(
        "--pt_loss_weight",
        type=float,
        default=1.0,
        help="Multiplier for PT-series loss weight (default: 1.0).",
    )
    parser.add_argument(
        "--other_loss_weight",
        type=float,
        default=0.5,
        help="Multiplier for other/auxiliary series loss weight (default: 0.5).",
    )
    parser.add_argument(
        "--horizon_gamma",
        type=float,
        default=1.0,
        help="Exponent for horizon weighting; higher => more weight on far horizon.",
    )
    parser.add_argument("--huber_delta", type=float, default=1.0, help="Huber delta in transformed space.")
    parser.add_argument(
        "--robust_metrics",
        action="store_true",
        default=True,
        help="Use variance-masked, epsilon-guarded RRSE/RAE/Correlation/sMAPE computations (safer with constant nodes).",
    )
    parser.add_argument(
        "--residual_head",
        action="store_true",
        default=True,
        help="Train/predict residuals relative to last input level per-sample (original units). RECOMMENDED for better amplitude tracking.",
    )
    parser.add_argument(
        "--dual_channel",
        choices=["none", "diff", "pct"],
        default="pct",
        help="Second input channel: pct (percentage changes, scale-invariant, RECOMMENDED), diff (z[t]-z[t-1]), or none for single-channel",
    )
    parser.add_argument(
        "--pct_clip",
        type=float,
        default=0.0,
        help="Clip absolute value of pct channel to stabilize training (3.0 recommended, 0 disables).",
    )
    parser.add_argument(
        "--conformal",
        action="store_true",
        default=True,
        help="Enable split-conformal adjustment on MC intervals (adds per-node q̂ half-width).",
    )
    parser.add_argument(
        "--conf_alpha",
        type=float,
        default=0.05,
        help="Miscoverage level α for conformal PI (0.05 => ~95%% nominal).",
    )
    parser.add_argument(
        "--auto_tune_dropout",
        action="store_true",
        help="Gently auto-lower dropout when validation CI is too wide.",
    )
    parser.add_argument(
        "--auto_dropout_target",
        type=float,
        default=0.35,
        help="Target median(CI_half / robust_amp) on Validation.",
    )
    parser.add_argument(
        "--auto_dropout_min",
        type=float,
        default=0.02,
        help="Minimum allowed dropout when auto-tuning.",
    )
    parser.add_argument(
        "--auto_dropout_step",
        type=float,
        default=0.10,
        help="Fractional decrease per adjust step, e.g. 0.10 = -10 percent.",
    )
    parser.add_argument(
        "--amp_loss_weight",
        type=float,
        default=0.3,
        help="Weight for amplitude (range) penalty in original units (0.0 disables, recommended for MAE-dominant loss)",
    )
    parser.add_argument(
        "--smooth_plot",
        action="store_true",
        help="Apply light exponential smoothing to plots only",
    )
    parser.add_argument("--smooth_alpha", type=float, default=0.1, help="Smoothing alpha for plots (0-1)")
    parser.add_argument(
        "--smooth_actual_alpha",
        type=float,
        default=None,
        help="Smoothing alpha for actual curve (defaults to max(smooth_alpha, 1.5*smooth_alpha capped at 0.6))",
    )
    parser.add_argument(
        "--smooth_actual_passes",
        type=int,
        default=1,
        help="Number of smoothing passes for actual curve in plots (>=1).",
    )

    parser.add_argument(
        "--steps_per_year",
        type=int,
        default=12,
        help="Steps per year for MM-YYYY plot ticks (12 for monthly data).",
    )
    parser.add_argument(
        "--despike_plots",
        action="store_true",
        default=False,
        help="Remove noise spikes from actual curves in plots (median filter only, no clipping)",
    )
    parser.add_argument(
        "--despike_median_window",
        type=int,
        default=5,
        help="Median filter window size for spike removal in plots",
    )
    parser.add_argument(
        "--plot_trend_smooth",
        action="store_true",
        help="Use trend-aware smoothing for actual curves in plots (Holt trend + residual EMA)",
    )
    parser.add_argument(
        "--plot_trend_alpha",
        type=float,
        default=0.25,
        help="Holt level alpha for plot smoothing (0-1)",
    )
    parser.add_argument(
        "--plot_trend_beta",
        type=float,
        default=0.05,
        help="Holt trend beta for plot smoothing (0-1)",
    )
    parser.add_argument(
        "--plot_resid_alpha",
        type=float,
        default=0.20,
        help="Residual EMA alpha for plot smoothing (0-1)",
    )
    parser.add_argument(
        "--plot_trend_clamp_min",
        type=float,
        default=None,
        help="Clamp minimum before plot trend smoothing (use None to allow negatives)",
    )
    parser.add_argument(
        "--plot_trend_impute",
        action="store_true",
        help="Impute spike picks to trend (robust residual outlier detection)",
    )
    parser.add_argument(
        "--plot_trend_impute_window",
        type=int,
        default=5,
        help="Window size for robust spike detection (odd >=3)",
    )
    parser.add_argument(
        "--plot_trend_impute_k",
        type=float,
        default=3.0,
        help="Outlier threshold in robust stds for plot trend spike impute",
    )
    parser.add_argument(
        "--plot_pi_level",
        type=float,
        default=0.95,
        help="Nominal central coverage for shaded band in validation plots " "(e.g. 0.5, 0.8, 0.95). Affects label only.",
    )
    parser.add_argument(
        "--plot_ci_scale",
        type=float,
        default=1.0,
        help="Scale factor for CI half-width in validation plots (visual only; <1 shrinks band).",
    )
    parser.add_argument(
        "--plot_ci_cap_ratio",
        type=float,
        default=1.0,
        help="Cap CI half-width to ratio * data range in validation plots (visual only).",
    )

    parser.add_argument(
        "--loss_alpha",
        type=float,
        default=1.0,
        help="Weight of L2/MSE term (1.5 for prioritizing closer curve fit)",
    )
    parser.add_argument(
        "--loss_beta",
        type=float,
        default=0.4,
        help="Weight of 1-corr term (0.0 disables correlation penalty which can cause lag)",
    )
    parser.add_argument(
        "--loss_gamma",
        type=float,
        default=0.8,
        help="Weight of sMAPE term (0.3 reduced to balance with MSE)",
    )
    parser.add_argument(
        "--mae_weight",
        type=float,
        default=0.2,
        help="MAE weight (3.0 reduced to prioritize MSE for tighter predictions)",
    )
    parser.add_argument(
        "--movement_loss_weight",
        type=float,
        default=0.3,
        help="Weight for movement loss (L1 on first differences in original units) to hug short-term changes.",
    )
    parser.add_argument(
        "--conf_calibrate",
        action="store_true",
        default=True,
        help="Calibrate dropout CI width on validation for ~95%% coverage",
    )

    parser.add_argument(
        "--scheduler",
        choices=["none", "cosine", "onecycle"],
        default="cosine",
        help="LR scheduler (epoch-wise). Cosine recommended for stable convergence.",
    )
    parser.add_argument(
        "--sched_T0",
        type=int,
        default=10,
        help="CosineWarmRestarts T0 (20 for longer stable learning)",
    )
    parser.add_argument("--sched_Tmult", type=int, default=2, help="CosineWarmRestarts T_mult")
    parser.add_argument("--onecycle_pct", type=float, default=0.3, help="OneCycle warmup pct")

    parser.add_argument(
        "--temporal_attn",
        action="store_true",
        default=True,
        help="Enable lightweight temporal self-attention block (after temporal convs). RECOMMENDED for better pattern capture.",
    )
    parser.add_argument(
        "--attn_heads",
        type=int,
        default=4,
        help="Number of attention heads for temporal MHSA (8 for multi-scale pattern capture).",
    )
    parser.add_argument(
        "--attn_dim",
        type=int,
        default=128,
        help="Projection dimension for temporal MHSA (Q/K/V) - 128 for richer representations.",
    )
    parser.add_argument(
        "--attn_dropout",
        type=float,
        default=0.08,
        help="Dropout inside temporal MHSA block (0.05 for slight regularization).",
    )
    parser.add_argument(
        "--attn_window",
        type=int,
        default=0,
        help="If >0, restrict attention to last W timesteps (local window).",
    )
    parser.add_argument(
        "--attn_math_mode",
        action="store_true",
        help="Force PyTorch to use math (non-flash) scaled dot product attention kernels for stability.",
    )
    parser.add_argument(
        "--attn_bn_chunk",
        type=int,
        default=0,
        help="If >0, chunk the (batch*nodes) axis for temporal attention to save memory.",
    )
    parser.add_argument(
        "--attn_gate_threshold",
        type=int,
        default=1500000,
        help="Auto-disable temporal attention when (mc_runs * batch_size * num_nodes) exceeds this threshold (0 disables gating).",
    )

    parser.add_argument(
        "--temporal_transformer",
        type=int,
        default=1,
        help="Enable Transformer encoder over time (1=enabled, RECOMMENDED).",
    )
    parser.add_argument(
        "--tt_layers",
        type=int,
        default=3,
        help="Number of Transformer encoder layers when enabled (4 for deeper temporal modeling).",
    )

    parser.add_argument(
        "--graph_mix",
        type=float,
        default=0.6,
        help="Blend ratio alpha for predefined vs learned adjacency (0.5 = equal mix, RECOMMENDED).",
    )
    parser.add_argument(
        "--dropedge_p",
        type=float,
        default=0.0,
        help="DropEdge probability during training (0.08 for regularization, 0.0-0.5 typical).",
    )

    parser.add_argument(
        "--quantiles",
        type=str,
        default="0.1,0.5,0.9",
        help='Comma-separated list of quantiles, e.g. "0.1,0.5,0.9" (empty to disable).',
    )
    parser.add_argument(
        "--lambda_quantile",
        type=float,
        default=0.2,
        help="Weight for quantile (pinball) loss.",
    )
    parser.add_argument(
        "--use_gauss",
        type=int,
        default=1,
        help="Enable Gaussian NLL auxiliary head/loss (1=on,0=off).",
    )
    parser.add_argument(
        "--lambda_nll",
        type=float,
        default=0.1,
        help="Weight for Gaussian NLL term when --use_gauss=1.",
    )
    parser.add_argument(
        "--use_nb_head",
        type=int,
        default=0,
        help="Enable Negative Binomial head/loss (1=on,0=off). Optional: --use_zinb 1 adds zero-inflation.",
    )
    parser.add_argument(
        "--use_zinb",
        type=int,
        default=0,
        help="With --use_nb_head: use Zero-Inflated NB (ZINB).",
    )

    parser.add_argument(
        "--graph_normalize",
        choices=["none", "sym", "square", "row"],
        default="none",
        help="Apply normalization to loaded adjacency: sym (D^-1/2 A D^-1/2), square (A@A, then row norm), row (row stochastic), none. Use --strong_rmdpt for optimal defaults.",
    )
    parser.add_argument(
        "--weight_nodes_in_loss",
        action="store_true",
        help="Weight node contributions in loss by inverse robust amplitude (per-node). Enabled by --strong_rmdpt.",
    )
    parser.add_argument(
        "--weight_nodes_in_metrics",
        action="store_true",
        default=True,
        help="Apply same node weights when aggregating metrics (MAE/RMSE/etc). Enabled by --strong_rmdpt.",
    )
    parser.add_argument(
        "--compile",
        type=str,
        choices=["off", "eager", "inductor", "auto"],
        default="off",
        help="Control torch.compile usage: off (disable), eager (backend=eager), inductor (default backend), auto (best effort).",
    )
    parser.add_argument(
        "--nan_debug",
        action="store_true",
        help="Enable verbose NaN/inf diagnostics during evaluation and sliding-window MC.",
    )
    parser.add_argument(
        "--auto_window_adjust",
        action="store_true",
        help="If no training windows exist, automatically shrink seq_in_len (and if needed seq_out_len) until at least --auto_window_min_train windows are available.",
    )
    parser.add_argument(
        "--auto_window_min_train",
        type=int,
        default=8,
        help="Target minimum number of training windows when using --auto_window_adjust (stop early if reached).",
    )
    parser.add_argument(
        "--debug_layout",
        action="store_true",
        help="Print detailed tensor layout transformations for debugging sequence length mismatches.",
    )
    parser.add_argument(
        "--log_gpu_mem",
        action="store_true",
        help="Log CUDA allocated/peak memory after each epoch and before/after evaluation.",
    )
    parser.add_argument(
        "--log_peak_mem",
        action="store_true",
        help="Alias for --log_gpu_mem (deprecated name).",
    )

    parser.add_argument(
        "--strong_rmdpt",
        action="store_true",
        help="Enable a strong default recipe for sparse/heavy-tailed RMD/PT forecasting (better RAE & curve hugging).",
    )
    parser.add_argument(
        "--hugging_mode",
        action="store_true",
        help="Maximize validation curve-hugging with a more aggressive preset (may increase compute).",
    )
    parser.add_argument(
        "--fresh_start",
        action="store_true",
        help="Start fresh training without loading existing checkpoint (random initialization). Note: AUTO-CLEANUP already removes old checkpoints unless --no_cleanup is used.",
    )

    return parser
