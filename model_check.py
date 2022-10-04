import gdown
import os
import torch
import omegaconf

if not os.path.exists('./checkpoint_best.pt'):
    gdown.download(url='https://drive.google.com/u/0/uc?id=1XJJ6A3xEMLjRbaO4z9FI-RP84fA4pvep&export=download', output='./checkpoint_best.zip') ## reference model
    os.system('unzip checkpoint_best.zip')
    gdown.download(url='https://drive.google.com/uc?id=1jCf7y2p2rVg_v1U0c2U3s_6Fzsxneua8', output='./checkpoint_best_new.pt') ## ours model

ref_model = torch.load('./checkpoint_best.pt')
our_model = torch.load('./checkpoint_best_new.pt')

ref_model_weight = ref_model['model']
our_model_weight = our_model['model']

differ_count = 0

for ref_n, our_n in zip(ref_model_weight, our_model_weight):
    if ref_n != our_n:
        print('model weight name differenct!')
        differ_count += 1
    if not torch.equal(ref_model_weight[ref_n], our_model_weight[our_n]):
        print('model weight manipulated!')
        differ_count += 1

print('')
print('-'*50)
print(f'our model and ref model differ count = {differ_count}')
print('-'*50)
print('')
