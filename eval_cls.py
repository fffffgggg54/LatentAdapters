import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import timm
import timm.data
import timm.optim

import tqdm

from PIL import Image

import matplotlib.pyplot as plt
import seaborn as sns

import math
from datetime import datetime

import os

from adapter import Adapter
import losses

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16



def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print("Created Directory : ", dir)
    return dir

model_names = [
    'caformer_b36.sail_in22k_ft_in1k',
    'convformer_b36.sail_in22k_ft_in1k',
    'vit_base_patch16_224.augreg_in21k_ft_in1k',
    'vit_base_patch16_clip_224.openai_ft_in1k',
    'eva02_base_patch14_448.mim_in22k_ft_in22k_in1k',
    #'vit_small_patch16_224.dino',
    #'eva02_base_patch14_448.mim_in22k_ft_in22k',
    'vit_large_patch16_dinov3.lvd1689m',
    #'vit_base_patch16_224.mae',
    #'vit_base_patch16_siglip_224.v2_webli',
    'vit_so400m_patch14_siglip_gap_224.pali2_10b_pt',
]
# TODO unnecesary, figure out a way to get embed dim/head without instantiating and loading weights for the whole model
print("building models...")
models = [timm.create_model(model_name, pretrained=True, num_classes=1000).eval() for model_name in tqdm.tqdm(model_names)]
model_dims = [model.num_features for model in models]
@torch.compile()
def fw_enc(model, x):
    x = model.forward_features(x)
    return model.forward_head(x, pre_logits=True)

def fw_head(model, x):
    model.head.to(device)
    if hasattr(model.head, 'fc'):
        # metaformer
        return model.head.fc(x)
    else:
        # vit
        return model.head(x)

# TODO flexible paths
print("loading train embeds...")
embeds_train = [
    torch.load(
        f'embeds/embeds_in1k_train_{model.default_cfg['architecture']}.{model.default_cfg['tag']}.pt',
        map_location='cpu'
    ).to(torch.bfloat16) for model in tqdm.tqdm(models)
]
labels_train = torch.load('embeds/labels_in1k_train.pt', map_location='cpu')

embeds_val = [
    torch.load(
        f'embeds/embeds_in1k_val_{model.default_cfg['architecture']}.{model.default_cfg['tag']}.pt',
        map_location='cpu'
    ) for model in models
]

labels_val = torch.load('embeds/labels_in1k_val.pt', map_location='cpu')


class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, embedsList):
        self.embedsList = embedsList
    def __len__(self):
        return len(self.embedsList[0])
    def __getitem__(self, idx):
        return tuple([embed[idx] for embed in self.embedsList])

adapter = Adapter([x.replace('.', '_') for x in model_names], model_dims)
adapter.load_state_dict(torch.load('adapters/adapter_20251014_weights_only.pt', weights_only=True))
#adapter = torch.load('adapters/adapter_20251014-192543_epoch_99.pt', weights_only=False)
#torch.save(adapter.state_dict(), 'adapters/adapter_20251014_weights_only.pt')
adapter = adapter.to(device)

print(adapter)
print(model_names)
print([x[0].shape for x in embeds_train])


bs_val = 2000
# no translation, own head
correct_default = torch.zeros(len(models), device=device)
# translate to own/other embedding space
# own on diagonal, other elsewhere
# in_model, out_model
correct_adapted = torch.zeros(len(models), len(models), device=device)
# translate self -> other -> self
# self, other
correct_cycle = torch.zeros(len(models), len(models), device=device)
# no translation, other head
#correct_naive = 0
total = 0

# N * [B, d_n]
embeds_val = [embed.to(device) for embed in embeds_val]
# N * [B]
labels_val = labels_val.to(device)

for embeds, labels in zip(zip(*[torch.split(x, bs_val, 0) for x in embeds_val]), torch.split(labels_val, bs_val, 0)):
    with torch.no_grad():
        adapted_embeds = adapter.adapt_all_to_all(embeds)
        self_cycle_embeds = adapter.fw_self_cycle_embeds(adapted_embeds)

        correct_default += torch.stack([(labels == torch.argmax(fw_head(model.float(), embed.float()), dim=1)).sum() for model, embed in zip(models, embeds)], dim=0)

        # stride on out_model, stack on dim 1 for in_model, out_model
        correct_adapted += torch.stack([(labels.unsqueeze(0) == torch.argmax(fw_head(model.float(), embed.float()), dim=-1)).sum(-1) for model, embed in zip(models, adapted_embeds)], dim=1)

        # stride on self, stack on dim 0 for self, other
        correct_cycle += torch.stack([(labels.unsqueeze(0) == torch.argmax(fw_head(model.float(), embed.float()), dim=-1)).sum(-1) for model, embed in zip(models, self_cycle_embeds)], dim=0)

        total += len(labels)
print(f'default: {correct_default/total}, adapted: {correct_adapted/total}, cycle: {correct_cycle/total}')