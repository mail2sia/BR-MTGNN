import torch,sys
ck='model/Bayesian/model.pt'
print('Loading',ck)
ckp=torch.load(ck,map_location='cpu')
print('Type:',type(ckp))
if isinstance(ckp,dict):
    print('Top-level keys:',list(ckp.keys()))
    if 'model_args' in ckp:
        print('\nmodel_args:')
        print(ckp['model_args'])
    if 'state_dict' in ckp:
        sd=ckp['state_dict']
    else:
        sd=ckp
else:
    sd=ckp
print('\nState-dict sample keys and shapes:')
for i,(k,v) in enumerate(sd.items()):
    print(i,k, getattr(v,'shape',None))
    if i>=30:
        break
print('\nTotal params in state-dict:', len(sd))
