import torch,collections
p='model/Bayesian/model.pt'
ck=torch.load(p,map_location='cpu')
sd=ck.get('state_dict', ck.get('model_state_dict', ck))
# collect patterns
layers=set()
filter_in=None
filter_out=None
num_nodes=None
skip_out=None
end_out=None
residual_candidate=None
for k,v in sd.items():
    if k=='predefined_A' and hasattr(v,'size'):
        s=tuple(v.size())
        if len(s)>=1:
            num_nodes=s[0]
    if 'filter_convs.' in k and k.endswith('.weight') and hasattr(v,'size'):
        s=tuple(v.size())
        # pattern filter_convs.{layer}.tconv.{i}.weight -> first index is out_channels
        parts=k.split('.')
        try:
            layer=int(parts[1])
            layers.add(layer)
        except:
            pass
        if filter_in is None:
            if len(s)>=2:
                filter_in=s[1]
            filter_out=s[0]
    if 'gate_convs.' in k and k.endswith('.weight') and hasattr(v,'size'):
        s=tuple(v.size())
        if residual_candidate is None and len(s)>=2:
            # gate convs often mirror filter convs
            residual_candidate=s[0]
    if k.endswith('skip_conv.weight') and hasattr(v,'size'):
        s=tuple(v.size())
        skip_out=s[0]
    if k.endswith('end_conv.weight') and hasattr(v,'size'):
        s=tuple(v.size())
        end_out=s[0]
# heuristics
num_layers = max(layers)+1 if layers else None
inferred = {
    'num_nodes': num_nodes,
    'conv_channels': filter_in,
    'filter_out_channels_sample': filter_out,
    'residual_channels_candidate': residual_candidate,
    'skip_channels': skip_out,
    'end_channels': end_out,
    'layers': num_layers
}
print('Inferred hparams:')
for k,v in inferred.items():
    print(f"{k}: {v}")
open('model/Bayesian/inferred_hparams.txt','w').write('\n'.join([f"{k}: {v}" for k,v in inferred.items()]))
print('\nWrote model/Bayesian/inferred_hparams.txt')
