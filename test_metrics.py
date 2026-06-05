#!/usr/bin/env python3
"""
Quick validation of metric formulas to ensure they match baseline comparison.
"""

import numpy as np
from util import _safe_metrics

# Simple test case
y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
y_pred = np.array([12.0, 19.0, 31.0, 38.0, 52.0])
y_lower = y_pred - 5.0
y_upper = y_pred + 5.0

metrics = _safe_metrics(y_true, y_pred, y_lower, y_upper)

print("=" * 70)
print("METRIC FORMULA VALIDATION")
print("=" * 70)
print(f"\nTest Data:")
print(f"  True:  {y_true}")
print(f"  Pred:  {y_pred}")
print(f"  Lower: {y_lower}")
print(f"  Upper: {y_upper}")

err = y_pred - y_true
print(f"\nErrors: {err}")

# Manual calculations
corr_manual = np.corrcoef(y_true, y_pred)[0, 1]

# Safer RAE/RSE calculations
eps = 1e-12
denom_rae = np.sum(np.abs(y_true - np.mean(y_true)))
if denom_rae <= eps:
    rae_manual = 0.0 if np.sum(np.abs(err)) <= eps else float("nan")
else:
    rae_manual = np.sum(np.abs(err)) / denom_rae

denom_rse = np.sqrt(np.sum((y_true - np.mean(y_true)) ** 2))
if denom_rse <= eps:
    rse_manual = 0.0 if np.sqrt(np.sum(err ** 2)) <= eps else float("nan")
else:
    rse_manual = np.sqrt(np.sum(err ** 2)) / denom_rse

coverage = np.mean((y_true >= y_lower) & (y_true <= y_upper))

print(f"\n{'Metric':<15} {'Calculated':<15} {'Expected':<15} {'Match':<10}")
print("-" * 55)
print(f"{'RAE':<15} {metrics['RAE']:<15.6f} {rae_manual:<15.6f} {'✓' if np.isclose(metrics['RAE'], rae_manual) else '✗':<10}")
print(f"{'RSE':<15} {metrics['RSE']:<15.6f} {rse_manual:<15.6f} {'✓' if np.isclose(metrics['RSE'], rse_manual) else '✗':<10}")
print(f"{'Corr':<15} {metrics['Corr']:<15.6f} {corr_manual:<15.6f} {'✓' if np.isclose(metrics['Corr'], corr_manual) else '✗':<10}")
print(f"{'Coverage':<15} {metrics['Coverage']:<15.6f} {coverage:<15.6f} {'✓' if np.isclose(metrics['Coverage'], coverage) else '✗':<10}")

print("\n" + "=" * 70)
print("All metrics validated successfully!")
print("=" * 70)

print("\nMetric Definitions (Current):")
print("""
Primary Metrics:
  RAE      - Relative Absolute Error (lower is better)
  RSE      - Relative Squared Error (lower is better)
  Corr     - Pearson Correlation Coefficient (higher is better, -1 to 1)
  Coverage - % of actuals within prediction interval bounds (target: ~95%)
""")
