#!/usr/bin/env python3
"""
Run All Baselines and Generate Comparative Evaluation Table

This script runs all baseline models (BMTGNN, LSTM, MTGNN, Transformer)
and generates a comprehensive comparison table with performance metrics.

Usage:
    python3 run_all_baselines.py [--epochs 10] [--device cpu] [--skip-models model1,model2]
"""

import os
import sys
import json
import time
import argparse
import subprocess
import math
import pandas as pd
from datetime import datetime
from pathlib import Path
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    import torch
except Exception:
    torch = None

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

class BaselineRunner:
    def __init__(
        self,
        repo_root,
        epochs=1,
        device='cpu',
        devices=None,
        trials=1,
        repeats=1,
        batch_size=16,
        parallel=False,
        lr_schedule='none',
        lr_patience_sched=10,
        lr_min=1e-6,
        lr_decay_factor=0.5,
        patience=50,
    ):
        self.repo_root = Path(repo_root)
        self.comparative_eval_dir = self.repo_root
        self.epochs = epochs
        self.device = device
        self.devices = devices or [device]
        self.trials = trials
        self.repeats = repeats
        self.batch_size = batch_size
        self.parallel = parallel
        self.lr_schedule = lr_schedule
        self.lr_patience_sched = lr_patience_sched
        self.lr_min = lr_min
        self.lr_decay_factor = lr_decay_factor
        self.patience = patience
        self.results = {}
        self.errors = {}
        self.skipped = []

        # Define baselines
        self.baselines = {
            'BMTGNN': {
                'dir': 'BMTGNN',
                'script': 'BMTGNN.py',
                'cmd_base': 'python3 -u BMTGNN.py',
                'data': './data/sm_data.csv',
                'args': f'--data ./data/sm_data.csv --epochs {epochs} --device {device} --seq_in_len 10 --seq_out_len 36 --num_nodes 190 --batch_size {batch_size} '
                        f'--lr_schedule {lr_schedule} --lr_patience_sched {lr_patience_sched} --lr_min {lr_min} '
                        f'--lr_decay_factor {lr_decay_factor} --patience {patience}',
                'metrics_pattern': r'(mae|mape|rmse|rae|rse|corr).*?(\d+\.?\d*)',
            },
            'BR-MTGNN': {
                'dir': 'BR-MTGNN',
                'script': 'train_test.py',
                'cmd_base': 'python3 -u train_test.py',
                'data': 'data/sm_data.csv',
                'args': f'--data_csv data/sm_data.csv --epochs {epochs} --device {device} --seq_in_len 10 --seq_out_len 36 --batch_size {batch_size} '
                        f'--lr_schedule {lr_schedule} --lr_patience_sched {lr_patience_sched} --lr_min {lr_min} '
                        f'--lr_decay_factor {lr_decay_factor} --patience {patience}',
                'metrics_pattern': r'(mae|mape|rmse|rae|rse|corr).*?(\d+\.?\d*)',
            },
            'LSTM_M': {
                'dir': 'Baselines/LSTM',
                'script': 'LSTM_m.py',
                'cmd_base': 'python3 -u LSTM_m.py',
                'data': './data/sm_data.csv',
                'args': f'--data ./data/sm_data.csv --epochs {epochs} --device {device} --seq_in_len 10 --seq_out_len 36 --batch_size {batch_size} '
                        f'--lr_schedule {lr_schedule} --lr_patience_sched {lr_patience_sched} --lr_min {lr_min} '
                        f'--lr_decay_factor {lr_decay_factor} --patience {patience}',
                'metrics_pattern': r'(mae|mape|rmse|rae|rse|corr).*?(\d+\.?\d*)',
            },
            'LSTM_U': {
                'dir': 'Baselines/LSTM',
                'script': 'LSTM_u.py',
                'cmd_base': 'python3 -u LSTM_u.py',
                'data': './data/sm_data.csv',
                'args': f'--data ./data/sm_data.csv --epochs {epochs} --device {device} --seq_in_len 10 --seq_out_len 36 --batch_size {batch_size} '
                        f'--lr_schedule {lr_schedule} --lr_patience_sched {lr_patience_sched} --lr_min {lr_min} '
                        f'--lr_decay_factor {lr_decay_factor} --patience {patience}',
                'metrics_pattern': r'(mae|mape|rmse|rae|rse|corr).*?(\d+\.?\d*)',
            },
            'MTGNN': {
                'dir': 'Baselines/MTGNN',
                'script': 'MTGNN.py',
                'cmd_base': 'python3 -u MTGNN.py',
                'data': './data/sm_data.csv',
                'args': f'--data ./data/sm_data.csv --epochs {epochs} --device {device} --seq_in_len 10 --seq_out_len 36 --num_nodes 190 --batch_size {batch_size} '
                        f'--lr_schedule {lr_schedule} --lr_patience_sched {lr_patience_sched} --lr_min {lr_min} '
                        f'--lr_decay_factor {lr_decay_factor} --patience {patience}',
                'metrics_pattern': r'(mae|mape|rmse|rae|rse|corr).*?(\d+\.?\d*)',
            },
            'Transformer_M': {
                'dir': 'Baselines/Transformer',
                'script': 'transformer_m.py',
                'cmd_base': 'python3 -u transformer_m.py',
                'data': './data/sm_data.csv',
                'args': f'--data ./data/sm_data.csv --epochs {epochs} --device {device} --seq_in_len 10 --seq_out_len 36 --batch_size {batch_size} '
                        f'--lr_schedule {lr_schedule} --lr_patience_sched {lr_patience_sched} --lr_min {lr_min} '
                        f'--lr_decay_factor {lr_decay_factor} --patience {patience}',
                'metrics_pattern': r'(mae|mape|rmse|rae|rse|corr).*?(\d+\.?\d*)',
            },
            'Transformer_U': {
                'dir': 'Baselines/Transformer',
                'script': 'transformer_u.py',
                'cmd_base': 'python3 -u transformer_u.py',
                'data': './data/sm_data.csv',
                'args': f'--data ./data/sm_data.csv --epochs {epochs} --device {device} --seq_in_len 10 --seq_out_len 36 --batch_size {batch_size} '
                        f'--lr_schedule {lr_schedule} --lr_patience_sched {lr_patience_sched} --lr_min {lr_min} '
                        f'--lr_decay_factor {lr_decay_factor} --patience {patience}',
                'metrics_pattern': r'(mae|mape|rmse|rae|rse|corr).*?(\d+\.?\d*)',
            },
            'DCRNN': {
                'dir': 'Baselines/Advanced',
                'script': 'dcrnn.py',
                'cmd_base': 'python3 -u dcrnn.py',
                'data': '../../data/sm_data.csv',
                'args': f'--data ../../data/sm_data.csv --epochs {epochs} --device {device} --seq_in_len 10 --seq_out_len 36 --num_nodes 190 --batch_size {batch_size} '
                        f'--lr_schedule {lr_schedule} --lr_patience_sched {lr_patience_sched} --lr_min {lr_min} '
                        f'--lr_decay_factor {lr_decay_factor} --patience {patience}',
                'metrics_pattern': r'(mae|mape|rmse|rae|rse|corr).*?(\d+\.?\d*)',
            },
            'AGCRN': {
                'dir': 'Baselines/Advanced',
                'script': 'agcrn.py',
                'cmd_base': 'python3 -u agcrn.py',
                'data': '../../data/sm_data.csv',
                'args': f'--data ../../data/sm_data.csv --epochs {epochs} --device {device} --seq_in_len 10 --seq_out_len 36 --num_nodes 190 --batch_size {batch_size} '
                        f'--lr_schedule {lr_schedule} --lr_patience_sched {lr_patience_sched} --lr_min {lr_min} '
                        f'--lr_decay_factor {lr_decay_factor} --patience {patience}',
                'metrics_pattern': r'(mae|mape|rmse|rae|rse|corr).*?(\d+\.?\d*)',
            },
            'PatchTST': {
                'dir': 'Baselines/Advanced',
                'script': 'patchtst.py',
                'cmd_base': 'python3 -u patchtst.py',
                'data': '../../data/sm_data.csv',
                'args': f'--data ../../data/sm_data.csv --epochs {epochs} --device {device} --seq_in_len 10 --seq_out_len 36 --num_nodes 190 --batch_size {batch_size} '
                        f'--lr_schedule {lr_schedule} --lr_patience_sched {lr_patience_sched} --lr_min {lr_min} '
                        f'--lr_decay_factor {lr_decay_factor} --patience {patience}',
                'metrics_pattern': r'(mae|mape|rmse|rae|rse|corr).*?(\d+\.?\d*)',
            },
            'TFT': {
                'dir': 'Baselines/Advanced',
                'script': 'tft.py',
                'cmd_base': 'python3 -u tft.py',
                'data': '../../data/sm_data.csv',
                'args': f'--data ../../data/sm_data.csv --epochs {epochs} --device {device} --seq_in_len 10 --seq_out_len 36 --num_nodes 190 --batch_size {batch_size} '
                        f'--lr_schedule {lr_schedule} --lr_patience_sched {lr_patience_sched} --lr_min {lr_min} '
                        f'--lr_decay_factor {lr_decay_factor} --patience {patience}',
                'metrics_pattern': r'(mae|mape|rmse|rae|rse|corr).*?(\d+\.?\d*)',
            },
            'TimesFM': {
                'dir': 'Baselines/Advanced',
                'script': 'timesfm.py',
                'cmd_base': 'python3 -u timesfm.py',
                'data': '../../data/sm_data.csv',
                'args': f'--data ../../data/sm_data.csv --epochs {epochs} --device {device} --seq_in_len 10 --seq_out_len 36 --num_nodes 190 --batch_size {batch_size} '
                        f'--lr_schedule {lr_schedule} --lr_patience_sched {lr_patience_sched} --lr_min {lr_min} '
                        f'--lr_decay_factor {lr_decay_factor} --patience {patience}',
                'metrics_pattern': r'(mae|mape|rmse|rae|rse|corr).*?(\d+\.?\d*)',
            },
            'Prophet': {
                'dir': 'Baselines/Advanced',
                'script': 'prophet_baseline.py',
                'cmd_base': 'python3 -u prophet_baseline.py',
                'data': '../../data/sm_data.csv',
                'args': f'--data ../../data/sm_data.csv --seq_out_len 36 --num_nodes 190 '
                        f'--changepoint_prior_scale 0.05 --interval_width 0.95',
                'metrics_pattern': r'(mae|mape|rmse|rae|rse|corr|coverage).*?(\d+\.?\d*)',
            },
        }

    @staticmethod
    def _normalize_devices(devices):
        """Return a validated device list; drop invalid CUDA ordinals."""
        if not devices:
            return ['cpu']

        if torch is None:
            # Torch unavailable in runner process, keep user devices as-is.
            return devices

        cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        valid = []
        for raw in devices:
            d = str(raw).strip().lower()
            if d.startswith('cuda:'):
                try:
                    idx = int(d.split(':', 1)[1])
                except (ValueError, IndexError):
                    continue
                if idx < cuda_count:
                    valid.append(f'cuda:{idx}')
            elif d == 'cuda':
                if cuda_count > 0:
                    valid.append('cuda:0')
            elif d == 'cpu':
                valid.append('cpu')
            else:
                # Preserve any custom device strings.
                valid.append(raw)

        if valid:
            return valid

        # Fallback when requested CUDA devices are unavailable.
        return ['cpu']

    @staticmethod
    def _resolve_runtime_device(device):
        """Resolve a single runtime device to a safe value for this machine."""
        d = str(device).strip().lower()
        if d == 'cuda':
            d = 'cuda:0'

        if not d.startswith('cuda:'):
            return d, None

        # If torch cannot be imported here, safest fallback is cuda:0.
        if torch is None:
            if d != 'cuda:0':
                return 'cuda:0', f"torch unavailable for validation; forcing {d} -> cuda:0"
            return d, None

        cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        try:
            idx = int(d.split(':', 1)[1])
        except (ValueError, IndexError):
            return ('cuda:0' if cuda_count > 0 else 'cpu'), f"invalid CUDA spec '{device}', using safe fallback"

        if cuda_count == 0:
            return 'cpu', f"no CUDA devices available; forcing {d} -> cpu"
        if idx >= cuda_count:
            return 'cuda:0', f"CUDA device ordinal out of range ({idx} >= {cuda_count}); forcing {d} -> cuda:0"
        return d, None

    def run_baseline(self, model_name, device=None):
        """Run a single baseline model."""
        device = device or self.device
        device, device_note = self._resolve_runtime_device(device)
        print(f"\n{'='*70}")
        print(f"Running {model_name} on {device}...")
        if device_note:
            print(f"Device adjustment: {device_note}")
        print(f"{'='*70}")

        baseline = self.baselines[model_name]
        model_dir = self.comparative_eval_dir / baseline['dir']

        if not model_dir.exists():
            self.errors[model_name] = f"Directory not found: {model_dir}"
            print(f"❌ {model_name}: Directory not found")
            return False

        if not (model_dir / baseline['script']).exists():
            self.errors[model_name] = f"Script not found: {baseline['script']}"
            print(f"❌ {model_name}: Script not found")
            return False

        # Build command and force selected runtime device regardless of baseline defaults.
        # This avoids stale hardcoded values (e.g., cuda:1) causing invalid device ordinals.
        args = re.sub(r'--device\s+\S+', f'--device {device}', baseline['args'])
        cmd = f"{baseline['cmd_base']} {args}"

        try:
            print(f"Command: {cmd}")
            print("Live progress table:")
            # Drop loss_norm column from the live table; show core metrics only.
            print("epoch\tRSE\tRAE\tCorr")
            print("-" * 70)

            # Run and stream logs live so epoch/trial metrics are visible during training.
            start_time = time.time()
            process = subprocess.Popen(
                cmd,
                shell=True,
                cwd=model_dir,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            output_lines = []
            progress_state = {'trial': None, 'last_key': None, 'epoch': None}
            if process.stdout is not None:
                for raw_line in process.stdout:
                    line = raw_line.rstrip('\n')
                    output_lines.append(line)
                    self.print_realtime_progress_line(line, progress_state)
            return_code = process.wait()
            elapsed_time = time.time() - start_time

            # Parse output for metrics
            output = "\n".join(output_lines)
            self.print_epoch_trial_metrics(output, model_name)
            if return_code != 0:
                self.errors[model_name] = f"Process exited with code {return_code}"
                print(f"❌ {model_name}: Failed with exit code {return_code}")
                print(f"   Output (last 500 chars):\n{output[-500:]}")
                return False

            metrics = self.extract_metrics(output, model_name)

            if metrics:
                metrics.pop('sMAPE', None)
                if 'Corr' in metrics and metrics['Corr'] is not None:
                    try:
                        metrics['Corr'] = max(0.0, float(metrics['Corr']))
                    except (ValueError, TypeError):
                        pass
                metrics['elapsed_time'] = elapsed_time
                metrics['status'] = 'completed'
                self.results[model_name] = metrics
                print(f"✅ {model_name}: Successfully completed ({elapsed_time:.1f}s)")
                print(f"   Metrics: {metrics}")
                return True
            else:
                self.errors[model_name] = "Could not extract metrics from output"
                print(f"⚠️  {model_name}: Completed but metrics not found")
                print(f"   Output (last 500 chars):\n{output[-500:]}")
                return False
        except Exception as e:
            self.errors[model_name] = str(e)
            print(f"❌ {model_name}: Error - {e}")
            return False

    @staticmethod
    def _extract_metric_value(line, metric_name):
        number_pattern = r'([-+]?(?:nan|inf|[0-9]*\.?[0-9]+(?:e[-+]?[0-9]+)?))'
        # Prefer exact-word matches first (e.g., 'loss', 'rse').
        match = re.search(rf'\b{metric_name}\b\s*[:=]?\s*{number_pattern}', line, re.IGNORECASE)
        if not match:
            # Fallback: accept tokens that include the metric name, e.g. 'train_loss', 'train-loss'.
            match2 = re.search(rf'\w*{metric_name}\w*\s*[:=]?\s*{number_pattern}', line, re.IGNORECASE)
            if not match2:
                return None
            try:
                return float(match2.group(1))
            except ValueError:
                return None
        try:
            return float(match.group(1))
        except ValueError:
            return None

    @staticmethod
    def _extract_epoch_only(line):
        line_l = line.lower()
        m = re.search(r'end of epoch\s+(\d+)', line_l)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                return None
        m = re.search(r'\bepoch\s*[:=]?\s*(\d+)\b', line_l)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                return None
        return None

    def _extract_progress_update(self, line, state):
        line_s = line.strip()
        line_l = line.lower()

        # Track trial-like progress hints used across baseline scripts.
        trial_match = re.search(r'\btrial\s*[:=]?\s*(\d+)\b', line_l)
        if trial_match:
            try:
                state['trial'] = int(trial_match.group(1))
            except ValueError:
                pass
        else:
            # BMTGNN prints hyperparameter trial as a standalone "Iter: <q>" line.
            # Avoid confusing this with batch-level lines like "iter: 20 | loss: ...".
            hp_iter_match = re.match(r'^\s*iter\s*[:=]?\s*(\d+)\s*$', line_s, re.IGNORECASE)
            if hp_iter_match:
                try:
                    state['trial'] = int(hp_iter_match.group(1))
                except ValueError:
                    pass

        # Unified per-epoch progress row:
        # epoch, loss, RSE, RAE, Corr
        epoch = self._extract_epoch_only(line)
        if epoch is not None:
            state['epoch'] = epoch

        loss = self._extract_metric_value(line, 'loss')
        if loss is None:
            loss = self._extract_metric_value(line, 'valid_loss')

        rse = self._extract_metric_value(line, 'rse')
        rae = self._extract_metric_value(line, 'rae')
        corr = self._extract_metric_value(line, 'corr')
        if epoch is None and loss is None and rse is None and rae is None and corr is None:
            return None

        epoch = state.get('epoch')
        # Keep display clean: emit only epoch-level rows that contain both RSE and RAE.
        if epoch is None or rse is None or rae is None:
            return None

        # Suppress repeated rows for the same epoch. Some baselines print more than one
        # progress line per epoch; we only want a single live table row per epoch.
        last_printed_epoch = state.get('last_printed_epoch')
        if last_printed_epoch == epoch:
            return None

        return {
            'trial': state.get('trial'),
            'epoch': epoch,
            'rse': rse,
            'rae': rae,
            'corr': corr,
            'loss': loss,
        }

    def print_realtime_progress_line(self, line, state):
        """Print normalized training progress rows from a raw log line."""
        row = self._extract_progress_update(line, state)
        if not row:
            return

        key = (
            row['trial'],
            row['epoch'],
            round(row['rse'], 10) if row['rse'] is not None else None,
            round(row['rae'], 10) if row['rae'] is not None else None,
            round(row['corr'], 10) if row['corr'] is not None else None,
        )
        if key == state.get('last_key'):
            return
        state['last_key'] = key
        state['last_printed_epoch'] = row['epoch']

        epoch_txt = str(row['epoch']) if row['epoch'] is not None else "-"
        # Print numeric columns with up to 4 digits after the decimal point.
        rse_txt = f"{row['rse']:.4f}" if row['rse'] is not None else "-"
        rae_txt = f"{row['rae']:.4f}" if row['rae'] is not None else "-"
        corr_txt = f"{row['corr']:.4f}" if row['corr'] is not None else "-"
        print(f"{epoch_txt}\t{rse_txt}\t{rae_txt}\t{corr_txt}")

    def print_epoch_trial_metrics(self, output, model_name):
        """Print per-trial/per-epoch RSE/RAE extracted from model logs."""
        rows = self.extract_epoch_trial_metrics(output)
        if not rows:
            return

        print(f"\n{model_name} epoch/trial metrics (RSE/RAE):")
        print("trial\tepoch\tRSE\tRAE")
        for row in rows:
            trial_txt = str(row['trial']) if row['trial'] is not None else "-"
            print(f"{trial_txt}\t{row['epoch']}\t{row['rse']:.6f}\t{row['rae']:.6f}")

    def extract_epoch_trial_metrics(self, output):
        """Best-effort extraction of trial/epoch RSE/RAE progress from mixed logs."""
        number_pattern = r'([-+]?(?:nan|inf|[0-9]*\.?[0-9]+(?:e[-+]?[0-9]+)?))'
        trial = None
        rows = []
        seen = set()

        trial_patterns = [
            re.compile(r'\btrial\s*[:=]?\s*(\d+)\b', re.IGNORECASE),
            re.compile(r'\biter\s*[:=]?\s*(\d+)\b', re.IGNORECASE),
        ]
        epoch_rse_rae_patterns = [
            re.compile(
                rf'\bepoch\s*[:=]?\s*(\d+).*?\brse\b\s*[:=]?\s*{number_pattern}.*?\brae\b\s*[:=]?\s*{number_pattern}',
                re.IGNORECASE
            ),
            re.compile(
                rf'\bepoch\s*[:=]?\s*(\d+).*?\brae\b\s*[:=]?\s*{number_pattern}.*?\brse\b\s*[:=]?\s*{number_pattern}',
                re.IGNORECASE
            ),
            # MTGNN-style: "...| end of epoch 12 | ... | valid rse 0.1234 | valid rae 0.4567"
            re.compile(
                rf'end of epoch\s+(\d+).*?\brse\b\s*[:=]?\s*{number_pattern}.*?\brae\b\s*[:=]?\s*{number_pattern}',
                re.IGNORECASE
            ),
        ]

        for line in output.splitlines():
            for pat in trial_patterns:
                m_trial = pat.search(line)
                if m_trial:
                    try:
                        trial = int(m_trial.group(1))
                    except ValueError:
                        pass
                    break

            for pat in epoch_rse_rae_patterns:
                m = pat.search(line)
                if not m:
                    continue
                try:
                    epoch = int(m.group(1))
                    n2 = float(m.group(2))
                    n3 = float(m.group(3))
                except ValueError:
                    continue

                # Determine whether pattern captured rse->rae or rae->rse
                if 'rae' in line.lower() and line.lower().find('rae') < line.lower().find('rse'):
                    rae, rse = n2, n3
                else:
                    rse, rae = n2, n3

                key = (trial, epoch, round(rse, 10), round(rae, 10))
                if key in seen:
                    continue
                seen.add(key)
                rows.append({'trial': trial, 'epoch': epoch, 'rse': rse, 'rae': rae})
                break

        rows.sort(key=lambda x: ((x['trial'] if x['trial'] is not None else 10**9), x['epoch']))
        return rows

    def extract_metrics(self, output, model_name):
        """Extract metrics from model output."""
        metrics = {}
        number_pattern = r'([-+]?(?:nan|inf|[0-9]*\.?[0-9]+(?:e[-+]?[0-9]+)?))'

        # Special handling for BR-MTGNN table format: "Group RAE RSE Corr Coverage"
        if model_name == 'BR-MTGNN':
            # Look for "Overall" row in table format
            match = re.search(r'Overall\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.-]+)\s+([0-9.]+)', output)
            if match:
                try:
                    metrics['RAE'] = float(match.group(1))
                    metrics['RSE'] = float(match.group(2))
                    metrics['Corr'] = float(match.group(3))
                    metrics['Coverage'] = float(match.group(4))
                    return metrics
                except ValueError:
                    pass

        final_test_match = re.search(
            rf'final test rse\s+{number_pattern}\s+\|\s+test rae\s+{number_pattern}\s+\|\s+test corr\s+{number_pattern}\s+\|\s+test smape\s+{number_pattern}',
            output,
            re.IGNORECASE
        )
        if final_test_match:
            try:
                metrics['RSE'] = float(final_test_match.group(1))
                metrics['RAE'] = float(final_test_match.group(2))
                metrics['Corr'] = float(final_test_match.group(3))
                return metrics
            except ValueError:
                pass

        # LSTM/Transformer scripts print:
        # test    rse    rae
        # mean    1.2345 0.6789
        table_match = re.search(
            rf'test\s+rse\s+rae\s*(?:\r?\n|$).*?mean\s+{number_pattern}\s+{number_pattern}',
            output,
            re.IGNORECASE | re.DOTALL
        )
        if table_match:
            try:
                metrics['RSE'] = float(table_match.group(1))
                metrics['RAE'] = float(table_match.group(2))
            except ValueError:
                pass

        # MTGNN/BMTGNN scripts print:
        # test    rse    rae    corr    s-mape
        # mean    1.2345 0.6789 0.1234  0.4567
        test_table_match = re.search(
            rf'test\s+rse\s+rae\s+corr\s+s-?mape.*?mean\s+{number_pattern}\s+{number_pattern}\s+{number_pattern}\s+{number_pattern}',
            output,
            re.IGNORECASE | re.DOTALL
        )
        if test_table_match:
            try:
                metrics['RSE'] = float(test_table_match.group(1))
                metrics['RAE'] = float(test_table_match.group(2))
                metrics['Corr'] = float(test_table_match.group(3))
            except ValueError:
                pass

        # Look for common metrics in output
        # Uses repo's canonical metrics: RAE, RSE, Corr, Coverage
        metric_patterns = {
            'RAE': [rf'RAE[:=\s]*{number_pattern}', rf'rae[:=\s]*{number_pattern}'],
            'RSE': [rf'RSE[:=\s]*{number_pattern}', rf'rse[:=\s]*{number_pattern}'],
            'Corr': [rf'Corr[:=\s]*{number_pattern}', rf'corr[:=\s]*{number_pattern}', rf'correlation[:=\s]*{number_pattern}'],
            'Coverage': [rf'Coverage[:=\s]*{number_pattern}', rf'coverage[:=\s]*{number_pattern}'],
        }

        for metric_name, patterns in metric_patterns.items():
            if metric_name in metrics:
                continue
            for pattern in patterns:
                match = re.search(pattern, output, re.IGNORECASE)
                if match:
                    try:
                        metrics[metric_name] = float(match.group(1))
                    except ValueError:
                        pass
                    break

        # If Corr was not captured from summary lines, use the latest epoch progress line.
        if 'Corr' not in metrics:
            corr_candidates = []
            for line in output.splitlines():
                m = re.search(r'\bcorr\b\s*[:=]?\s*([-+]?(?:nan|inf|[0-9]*\.?[0-9]+(?:e[-+]?[0-9]+)?))', line, re.IGNORECASE)
                if not m:
                    continue
                try:
                    v = float(m.group(1))
                except ValueError:
                    continue
                if math.isfinite(v):
                    corr_candidates.append(v)
            if corr_candidates:
                metrics['Corr'] = corr_candidates[-1]

        return metrics if metrics else None

    def run_all(self, skip_models=None):
        """Run all baselines."""
        if skip_models is None:
            skip_models = []

        print(f"\n{'='*70}")
        print(f"Starting Baseline Comparison Evaluation")
        print(f"{'='*70}")
        print(f"Epochs: {self.epochs}")
        print(f"Devices: {', '.join(self.devices)}")
        print(f"Trials per random-search model: {self.trials}")
        print(f"Repeats per model: {self.repeats}")
        print(f"Batch size: {self.batch_size}")
        print(f"Parallel: {self.parallel}")
        print(f"Timeout: None (free GPU available)")
        print(f"Models to run: {len(self.baselines) - len(skip_models)}")

        model_names = []
        for model_name in self.baselines.keys():
            if model_name not in skip_models:
                model_names.append(model_name)
            else:
                print(f"⊘ Skipping {model_name}")

        if self.parallel and len(self.devices) > 1:
            with ThreadPoolExecutor(max_workers=len(self.devices)) as executor:
                futures = {
                    executor.submit(self.run_baseline, model_name, self.devices[i % len(self.devices)]): model_name
                    for i, model_name in enumerate(model_names)
                }
                for future in as_completed(futures):
                    future.result()
        else:
            for i, model_name in enumerate(model_names):
                self.run_baseline(model_name, self.devices[i % len(self.devices)])

        return self.generate_report()

    def generate_report(self):
        """Generate comparison report."""
        print(f"\n{'='*70}")
        print("COMPARATIVE EVALUATION RESULTS")
        print(f"{'='*70}\n")

        all_df = self._build_all_results_dataframe()
        print("All Models Summary:")
        print(all_df.to_string())
        self._save_all_results(all_df)

        # Create results dataframe
        if self.results:
            df = pd.DataFrame.from_dict(self.results, orient='index')

            # Ensure requested final columns exist (fill with NaN if absent)
            required_cols = ['RSE', 'RAE', 'Corr']
            for c in required_cols:
                if c not in df.columns:
                    df[c] = pd.NA

            # Sort by multiple metrics and print
            print("\nResults Table:")
            # Print only the standard comparison columns for readability
            print(df[required_cols].to_string())

            # Save to CSV with exact column order: Model | RSE | RAE | Corr
            output_file = self.comparative_eval_dir / 'baseline_comparison_results.csv'
            final_df = df[required_cols].copy()
            final_df.insert(0, 'Model', final_df.index)
            final_df = final_df.reset_index(drop=True)
            final_df.to_csv(output_file, index=False)
            print(f"\n✅ Results saved to: {output_file}")

            # Generate markdown table
            self.generate_markdown_table(df)
        else:
            print("❌ No results to report")

        # Print errors
        if self.errors:
            print(f"\n{'='*70}")
            print("ERRORS ENCOUNTERED")
            print(f"{'='*70}")
            for model_name, error in self.errors.items():
                print(f"❌ {model_name}: {error}")

        return self.results

    def _build_all_results_dataframe(self):
        """Build one summary table including successes and failures."""
        metric_names = set()
        for metrics in self.results.values():
            metric_names.update(metrics.keys())

        ordered_metrics = [m for m in ['RAE', 'RSE', 'Corr', 'Coverage', 'elapsed_time'] if m in metric_names]
        ordered_metrics += sorted(metric_names - set(ordered_metrics) - {'status'})

        rows = {}
        for model_name in self.baselines.keys():
            if model_name in self.results:
                row = dict(self.results[model_name])
                row['status'] = row.get('status', 'completed')
                row['error'] = ''
            elif model_name in self.errors:
                row = {metric: None for metric in ordered_metrics}
                row['status'] = 'failed'
                row['error'] = self.errors[model_name]
            else:
                row = {metric: None for metric in ordered_metrics}
                row['status'] = 'not_run_or_skipped'
                row['error'] = ''
            rows[model_name] = row

        columns = ordered_metrics + ['status', 'error']
        return pd.DataFrame.from_dict(rows, orient='index').reindex(columns=columns)

    def _save_all_results(self, all_df):
        """Save complete run summary in CSV, JSON, and TXT formats."""
        def json_safe(value):
            if isinstance(value, dict):
                return {key: json_safe(item) for key, item in value.items()}
            if isinstance(value, list):
                return [json_safe(item) for item in value]
            if isinstance(value, float) and not math.isfinite(value):
                return None
            return value

        generated_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        csv_file = self.comparative_eval_dir / 'baseline_comparison_all_results.csv'
        json_file = self.comparative_eval_dir / 'baseline_run_summary.json'
        txt_file = self.comparative_eval_dir / 'baseline_run_summary.txt'

        all_df.to_csv(csv_file)
        clean_df = all_df.astype(object).where(pd.notnull(all_df), None)
        summary = {
            'generated_at': generated_at,
            'configuration': {
                'epochs': self.epochs,
                'devices': self.devices,
                'trials': self.trials,
                'repeats': self.repeats,
                'batch_size': self.batch_size,
                'parallel': self.parallel,
            },
            'counts': {
                'total': len(self.baselines),
                'completed': len(self.results),
                'failed': len(self.errors),
            },
            'models': json_safe(clean_df.to_dict(orient='index')),
        }

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, allow_nan=False)

        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("Baseline Run Summary\n")
            f.write("====================\n\n")
            f.write(f"Generated: {generated_at}\n")
            f.write(f"Epochs: {self.epochs}\n")
            f.write(f"Devices: {', '.join(self.devices)}\n")
            f.write(f"Trials: {self.trials}\n")
            f.write(f"Repeats: {self.repeats}\n")
            f.write(f"Batch size: {self.batch_size}\n")
            f.write(f"Parallel: {self.parallel}\n\n")
            f.write(clean_df.to_string())
            f.write("\n")

        print(f"\n✅ All-model CSV saved to: {csv_file}")
        print(f"✅ Run summary JSON saved to: {json_file}")
        print(f"✅ Run summary TXT saved to: {txt_file}")

    def _dataframe_to_markdown(self, df, columns=None):
        """Convert dataframe to markdown table without tabulate dependency."""
        if columns:
            df = df[columns]

        markdown = "| " + " | ".join(df.columns) + " |\n"
        markdown += "|" + "|".join(["-" * (len(col) + 2) for col in df.columns]) + "|\n"

        for idx, row in df.iterrows():
            values = [str(idx)] + [str(round(v, 4)) if isinstance(v, (int, float)) else str(v) for v in row.values]
            markdown += "| " + " | ".join(values) + " |\n"

        return markdown

    def generate_markdown_table(self, df):
        """Generate markdown comparison table."""
        md_file = self.comparative_eval_dir / 'BASELINE_COMPARISON.md'

        with open(md_file, 'w') as f:
            f.write("# Baseline Comparison Results\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Configuration:**\n")
            f.write(f"- Epochs: {self.epochs}\n")
            f.write(f"- Devices: {', '.join(self.devices)}\n")
            f.write(f"- Trials: {self.trials}\n")
            f.write(f"- Repeats: {self.repeats}\n")
            f.write(f"- Batch size: {self.batch_size}\n")
            f.write(f"- Data: 264 timesteps × 190 features\n\n")

            f.write("## Results Table\n\n")
            f.write(self._dataframe_to_markdown(df))
            f.write("\n\n")

            f.write("## Metric Definitions\n\n")
            f.write("### Primary Metrics\n")
            f.write("- **RAE**: Relative Absolute Error (lower is better)\n")
            f.write("- **RSE**: Relative Squared Error (lower is better)\n")
            f.write("- **Corr**: Pearson Correlation (higher is better, -1 to 1)\n")
            f.write("- **Coverage**: Prediction interval coverage (target: ~0.95 for 95%)\n")
            f.write("- **elapsed_time**: Execution time in seconds\n\n")

            f.write("## Model Descriptions\n\n")
            f.write("| Model | Type | Description |\n")
            f.write("|-------|------|-------------|\n")
            f.write("| BMTGNN | Graph-Temporal | Bayesian-Multivariate Temporal Graph Neural Network |\n")
            f.write("| BR-MTGNN | Graph-Temporal | Bayesian Recurrent-Multivariate Temporal Graph Neural Network |\n")
            f.write("| LSTM_M | Recurrent | LSTM with Multivariate input |\n")
            f.write("| LSTM_U | Recurrent | LSTM with Univariate input |\n")
            f.write("| MTGNN | Graph-Temporal | Multivariate Temporal Graph Neural Network |\n")
            f.write("| Transformer_M | Attention | Transformer with Multivariate input |\n")
            f.write("| Transformer_U | Attention | Transformer with Univariate input |\n")
            f.write("| DCRNN | Graph-Recurrent | Diffusion Convolutional Recurrent Neural Network |\n")
            f.write("| AGCRN | Graph-Recurrent | Adaptive Graph Convolutional Recurrent Network |\n")
            f.write("| PatchTST | Patch-Attention | Patch Time Series Transformer (channel-independent) |\n")
            f.write("| TFT | Attention | Temporal Fusion Transformer |\n")
            f.write("| TimesFM | Foundation | TimesFM 1.0 pretrained backbone (frozen) + fine-tuned linear head |\n")
            f.write("| Prophet | Classical-Probabilistic | Facebook Prophet additive model (yearly seasonality, per-node univariate) |\n")
        print(f"✅ Markdown table saved to: {md_file}")

def main():
    parser = argparse.ArgumentParser(
        description='Run all baseline models and generate comparison table'
    )
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use when --devices is not set: cpu, cuda, cuda:0, or cuda:1 (default: cuda)')
    parser.add_argument('--devices', type=str, default='',
                        help='Comma-separated devices for model-level parallelism, e.g. cuda:0,cuda:1')
    parser.add_argument('--parallel', action='store_true',
                        help='Run independent baselines concurrently across --devices')
    parser.add_argument('--trials', type=int, default=1,
                        help='Random-search trials per model (default: 1; old behavior was 30)')
    parser.add_argument('--repeats', type=int, default=1,
                        help='Full repeated runs per model (default: 1; old behavior was 5)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for all trainable baseline models (default: 16)')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience for models that support it (default: 50)')
    parser.add_argument('--lr_schedule', type=str, default='none',
                        choices=['none', 'plateau', 'cosine'],
                        help='Learning rate schedule for models that support it (default: none)')
    parser.add_argument('--lr_patience_sched', type=int, default=10,
                        help='Patience for ReduceLROnPlateau (default: 10)')
    parser.add_argument('--lr_min', type=float, default=1e-6,
                        help='Minimum learning rate for schedulers (default: 1e-6)')
    parser.add_argument('--lr_decay_factor', type=float, default=0.5,
                        help='LR decay factor for ReduceLROnPlateau (default: 0.5)')
    parser.add_argument('--skip-models', type=str, default='',
                        help='Comma-separated list of models to skip')

    args = parser.parse_args()

    # Get repo root (Comparative_Evaluation directory)
    repo_root = Path(__file__).parent

    # Parse skip models
    skip_models = [m.strip() for m in args.skip_models.split(',') if m.strip()]
    requested_devices = [d.strip() for d in args.devices.split(',') if d.strip()] or [args.device]
    devices = BaselineRunner._normalize_devices(requested_devices)
    if devices != requested_devices:
        print(f"Requested devices: {', '.join(requested_devices)}")
        print(f"Validated devices: {', '.join(devices)}")

    # Create runner and run all baselines
    runner = BaselineRunner(
        repo_root,
        epochs=args.epochs,
        device=devices[0],
        devices=devices,
        trials=args.trials,
        repeats=args.repeats,
        batch_size=args.batch_size,
        parallel=args.parallel,
        lr_schedule=args.lr_schedule,
        lr_patience_sched=args.lr_patience_sched,
        lr_min=args.lr_min,
        lr_decay_factor=args.lr_decay_factor,
        patience=args.patience,
    )

    results = runner.run_all(skip_models=skip_models)

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total models: {len(runner.baselines)}")
    print(f"Completed: {len(results)}")
    print(f"Failed: {len(runner.errors)}")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
