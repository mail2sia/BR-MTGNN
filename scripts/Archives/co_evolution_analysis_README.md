# Co-evolution Analysis (RQ5)

This script generates the missing RQ5 components:
- learned adjacency "attention" map + top edges
- node influence scores (in/out/total strength)
- leading/lagging metrics (lag-correlation + optional Granger p-values)
- early-warning gap metrics (RMD rising while PT lags)
- rolling network-level dynamics (avg RMD-PT correlation)

## Run

```powershell
python scripts/co_evolution_analysis.py --data data/sm_data_g.csv --nodes data/nodes.csv --rmd-pt-map data/Archives/RMD_PT_map.csv --graph data/graph_topk_k12.csv --checkpoint model/Bayesian/model.pt --forecast model/Bayesian/forecast/forecast_2026_2028.csv --output-dir model/Bayesian/forecast/co_evolution --trend-window 24 --max-lag 12
```

## Outputs

- `attention_heatmap.png` / `attention_heatmap.pdf`
- `attention_edges_all.csv` / `attention_edges_topk.csv`
- `node_influence_strength.csv`
- `rmd_pt_lead_lag_metrics.csv`
- `early_warning_gaps.csv`
- `network_dynamics.csv`

## Summary Report

Create top-10 tables and a composite figure:

```powershell
python scripts/co_evolution_report.py --input-dir model/Bayesian/forecast/co_evolution --topk 10
```

Outputs:
- `early_warning_gaps_top10.csv`
- `node_influence_top10.csv`
- `co_evolution_summary.png`
- `co_evolution_summary.pdf`
