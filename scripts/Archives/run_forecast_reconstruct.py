import os, sys, torch, importlib.util, inspect
from datetime import datetime
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

spec = importlib.util.spec_from_file_location('forecast_module', os.path.join(os.path.dirname(__file__), 'forecast.py'))
if spec is None or getattr(spec, 'loader', None) is None:
    raise RuntimeError('Unable to create module spec or loader for scripts/forecast.py')
forecast = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
# load module without executing as __main__
spec.loader.exec_module(forecast)  # type: ignore[attr-defined]

ck_path = Path('model') / 'Bayesian' / 'model.pt'
print('Loading checkpoint', ck_path)
ck = torch.load(str(ck_path), map_location='cpu')
sd = ck.get('state_dict', ck)

# Infer architecture params from state_dict shapes
conv_channels = sd['residual_convs.0.weight'].shape[1]
residual_channels = sd['residual_convs.0.weight'].shape[0]
skip_channels = sd['skip_convs.0.weight'].shape[0]
end_channels = sd['end_conv_1.weight'].shape[0]
out_dim = sd['end_conv_2.weight'].shape[0]
seq_length = sd['skip_convs.0.weight'].shape[3]
# layers = number of residual_convs entries
layers = len({int(k.split('.')[1]) for k in sd.keys() if k.startswith('residual_convs.')})
# node embedding
num_nodes = sd['gc.emb1.weight'].shape[0]
node_dim = sd['gc.emb1.weight'].shape[1]

print('Inferred params:')
print('  num_nodes=', num_nodes)
print('  node_dim=', node_dim)
print('  seq_length=', seq_length)
print('  conv_channels=', conv_channels)
print('  residual_channels=', residual_channels)
print('  skip_channels=', skip_channels)
print('  end_channels=', end_channels)
print('  out_dim=', out_dim)
print('  layers=', layers)

# choose sensible defaults for remaining args
gcn_depth = 2
dropout = 0.03
dilation_exponential = 2
propalpha = 0.05
tanhalpha = 3.0
attn_dim = 64
attn_heads = 8
tt_layers = 2

# construct model using forecast.gtnet
print('Constructing model...')
model = forecast.gtnet(
    gcn_true=True,
    buildA_true=True,
    gcn_depth=gcn_depth,
    num_nodes=num_nodes,
    device='cpu',
    predefined_A=None,
    static_feat=None,
    dropout=dropout,
    subgraph_size=20,
    node_dim=node_dim,
    dilation_exponential=dilation_exponential,
    conv_channels=conv_channels,
    residual_channels=residual_channels,
    skip_channels=skip_channels,
    end_channels=end_channels,
    seq_length=seq_length,
    in_dim=2,
    out_dim=out_dim,
    layers=layers,
    propalpha=propalpha,
    tanhalpha=tanhalpha,
    layer_norm_affline=True,
    temporal_attn=False,
    attn_dim=attn_dim,
    attn_heads=attn_heads,
    attn_dropout=0.1,
    temporal_transformer=True,
    tt_layers=tt_layers
)

print('Loading state_dict...')
# load state dict; use mapping to current model
state_dict = sd
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print('Missing keys:', len(missing))
print('Unexpected keys:', len(unexpected))

model.eval()
print('Model ready')

# Now call forecast pipeline functions from forecast module
# Load scaler
scaler_path = Path('model') / 'Bayesian' / 'y_scaler.pt'
if scaler_path.exists():
    scaler = forecast.load_scaler(str(scaler_path))
else:
    print('Scaler not found at', scaler_path)
    scaler = None

# Load historical data (use default data path expected by forecast.py)
data_path = Path('data') / 'sm_data_g.csv'
if not data_path.exists():
    print('Historical data not found at', data_path)
    sys.exit(1)

# Use dual_channel='pct' to match training dual-channel input (levels + pct change)
input_tensor, column_names, last_date, historical_df, last_values, in_dim, start_date = forecast.load_historical_data(
    str(data_path), seq_length=seq_length, scaler=scaler, dual_channel='pct', pct_clip=0.0, y_transform=None, expected_nodes=num_nodes
)

# generate forecast
forecast_mean, confidence, variance, quantile_bounds = forecast.generate_forecast(
    model=model,
    input_tensor=input_tensor,
    device='cpu',
    num_runs=10,
    pi_level=0.95,
    band_scale=1.0,
    ci_mode='quantile'
)

# inverse transform
forecast_original = forecast.inverse_transform_forecast(forecast_mean, scaler, column_names=column_names, start_date=None, clamp_floor=0.0, baseline=last_values, y_transform=None)

# filter RMD/PT
filtered_forecast, filtered_cols = forecast.filter_rmd_pt_variables(forecast_original, column_names)

# save
out_csv = Path('model') / 'Bayesian' / 'forecast' / 'forecast_2026_2028.csv'
forecast.save_forecast_to_csv(filtered_forecast, filtered_cols, start_date='2026-01-01', output_path=out_csv)
print('Saved forecast to', out_csv)

# attempt plotting (will use matplotlib Agg if headless)
try:
    # Convert to tensors with shapes expected by plot_forecast:
    # historical: [time, nodes], forecast: [horizon, nodes], confidence: [horizon, nodes]
    hist_t = torch.FloatTensor(historical_df.values)
    fore_t = torch.FloatTensor(filtered_forecast.T)
    conf_t = torch.FloatTensor(confidence.T)
    index_map = {name: i for i, name in enumerate(column_names)}
    start_date_dt = datetime(2026, 1, 1)
    forecast.plot_forecast(
        hist_t,
        fore_t,
        conf_t,
        filtered_cols[0],
        [],
        index_map,
        column_names,
        alarming=False,
        band_levels=None,
        pi_level=0.95,
        start_date=start_date_dt,
    )
except Exception as e:
    print('Plot failed:', e)

print('Done')
