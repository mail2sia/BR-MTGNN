import torch

from src.net import gtnet


def _strip_module_prefix(state_dict):
    if any(key.startswith('module.') for key in state_dict.keys()):
        return {key.replace('module.', '', 1): value for key, value in state_dict.items()}
    return state_dict


def _infer_from_state_dict(state_dict):
    conv_channels = state_dict['residual_convs.0.weight'].shape[1]
    residual_channels = state_dict['residual_convs.0.weight'].shape[0]
    skip_channels = state_dict['skip_convs.0.weight'].shape[0]
    end_channels = state_dict['end_conv_1.weight'].shape[0]
    out_dim = state_dict['end_conv_2.weight'].shape[0]
    seq_length = state_dict['skip_convs.0.weight'].shape[3]
    layers = len({int(key.split('.')[1]) for key in state_dict.keys() if key.startswith('residual_convs.')})
    num_nodes = state_dict['gc.emb1.weight'].shape[0]
    node_dim = state_dict['gc.emb1.weight'].shape[1]
    in_dim = state_dict['start_conv.weight'].shape[1]
    gcn_true = any(key.startswith('gconv1.') for key in state_dict.keys())
    temporal_attn = any(key.startswith('temporal_attn.') for key in state_dict.keys())
    temporal_transformer = any(key.startswith('temporal_enc.') for key in state_dict.keys())
    quantiles = []
    if '_taus' in state_dict and hasattr(state_dict['_taus'], 'tolist'):
        try:
            quantiles = list(state_dict['_taus'].tolist())
        except Exception:
            quantiles = []
    nb_head = any(key.startswith('end_conv_nb.') for key in state_dict.keys())
    zinb = any(key.startswith('end_conv_zi.') for key in state_dict.keys())
    gauss_head = any(key.startswith('end_conv_gauss.') for key in state_dict.keys())
    return {
        'conv_channels': conv_channels,
        'residual_channels': residual_channels,
        'skip_channels': skip_channels,
        'end_channels': end_channels,
        'out_dim': out_dim,
        'seq_length': seq_length,
        'layers': layers,
        'num_nodes': num_nodes,
        'node_dim': node_dim,
        'in_dim': in_dim,
        'gcn_true': gcn_true,
        'temporal_attn': temporal_attn,
        'temporal_transformer': temporal_transformer,
        'quantiles': quantiles,
        'nb_head': nb_head,
        'zinb': zinb,
        'gauss_head': gauss_head,
    }


def _infer_attn_params(state_dict, default_dim=64, default_heads=8):
    attn_dim = default_dim
    attn_heads = default_heads

    qkv_key = 'temporal_attn.qkv.weight'
    inproj_key = 'temporal_attn.mha.in_proj_weight'
    if qkv_key in state_dict and hasattr(state_dict[qkv_key], 'shape'):
        attn_dim = int(state_dict[qkv_key].shape[0])
    elif inproj_key in state_dict and hasattr(state_dict[inproj_key], 'shape'):
        attn_dim = int(state_dict[inproj_key].shape[0] // 3)

    for cand in (8, 4, 2):
        if attn_dim % cand == 0:
            attn_heads = cand
            break

    return attn_dim, attn_heads


def _infer_tt_layers(state_dict, default_layers=2):
    layers = set()
    for key in state_dict.keys():
        if key.startswith('temporal_enc.enc.layers.'):
            parts = key.split('.')
            if len(parts) > 3 and parts[3].isdigit():
                layers.add(int(parts[3]))
    return (max(layers) + 1) if layers else default_layers


def build_model_from_checkpoint(checkpoint, device):
    state_dict = _strip_module_prefix(checkpoint['state_dict'])
    meta = _infer_from_state_dict(state_dict)
    attn_dim, attn_heads = _infer_attn_params(state_dict)
    tt_layers = _infer_tt_layers(state_dict, default_layers=2)
    layer_norm_affline = any(key.startswith('norm.') for key in state_dict.keys())

    hparams = checkpoint.get('hparams')
    hp = None
    if isinstance(hparams, (list, tuple)) and len(hparams) >= 13:
        hp = {
            'gcn_depth': int(hparams[0]),
            'conv_channels': int(hparams[2]),
            'residual_channels': int(hparams[3]),
            'skip_channels': int(hparams[4]),
            'end_channels': int(hparams[5]),
            'subgraph_size': int(hparams[6]),
            'dropout': float(hparams[7]),
            'dilation_exponential': int(hparams[8]),
            'node_dim': int(hparams[9]),
            'propalpha': float(hparams[10]),
            'tanhalpha': float(hparams[11]),
            'layers': int(hparams[12]),
        }

    if hp and hp['layers'] != meta['layers']:
        print('Hparams layers mismatch; using checkpoint-inferred layers', meta['layers'])

    model = gtnet(
        gcn_true=meta['gcn_true'],
        buildA_true=meta['gcn_true'],
        gcn_depth=hp['gcn_depth'] if hp else 2,
        num_nodes=meta['num_nodes'],
        device=device,
        predefined_A=None,
        static_feat=None,
        dropout=hp['dropout'] if hp else 0.03,
        subgraph_size=hp['subgraph_size'] if hp else 20,
        node_dim=meta['node_dim'],
        dilation_exponential=hp['dilation_exponential'] if hp else 2,
        conv_channels=meta['conv_channels'],
        residual_channels=meta['residual_channels'],
        skip_channels=meta['skip_channels'],
        end_channels=meta['end_channels'],
        seq_length=meta['seq_length'],
        in_dim=meta['in_dim'],
        out_dim=meta['out_dim'],
        layers=meta['layers'] if not hp else (meta['layers'] if hp['layers'] != meta['layers'] else hp['layers']),
        propalpha=hp['propalpha'] if hp else 0.05,
        tanhalpha=hp['tanhalpha'] if hp else 3.0,
        layer_norm_affline=layer_norm_affline,
        temporal_attn=meta['temporal_attn'],
        attn_dim=attn_dim,
        attn_heads=attn_heads,
        attn_dropout=0.1,
        temporal_transformer=meta['temporal_transformer'],
        tt_layers=tt_layers,
        quantiles=meta['quantiles'],
        nb_head=meta['nb_head'],
        zinb=meta['zinb'],
        gauss_head=meta['gauss_head'],
    )
    model = model.to(device)

    return model, meta['seq_length'], meta['in_dim'], state_dict


def filter_state_dict_for_model(state_dict, model):
    model_state = model.state_dict()
    filtered = {}
    unexpected = []
    shape_mismatch = []
    for key, value in state_dict.items():
        if key not in model_state:
            unexpected.append(key)
            continue
        if model_state[key].shape != value.shape:
            shape_mismatch.append((key, tuple(value.shape), tuple(model_state[key].shape)))
            continue
        filtered[key] = value

    missing = [key for key in model_state.keys() if key not in filtered]
    unexpected = [key for key in unexpected if key != 'predefined_A']
    return filtered, missing, unexpected, shape_mismatch
