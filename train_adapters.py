import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

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

#out_dir = "outputs/basic_discriminator1.0_latent1.0_MSE_expansion_AllAnchors_JointAddition/"
out_dir = "outputs/scratch_mmID_2048_100epoch_discriminator0.0_latent1.0_MSE_JointTraining/"
expand = False
separate_expand = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print("Created Directory : ", dir)
    return dir
'''
base_adapter_models = [
    'caformer_b36.sail_in22k_ft_in1k',
    'convformer_b36.sail_in22k_ft_in1k',#
    'vit_base_patch16_224.augreg_in21k_ft_in1k',#
    'vit_base_patch16_clip_224.openai_ft_in1k',#
    'eva02_base_patch14_448.mim_in22k_ft_in22k_in1k',
    'vit_large_patch16_dinov3.lvd1689m',
    'vit_so400m_patch14_siglip_gap_224.pali2_10b_pt',#

    'convnext_base.fb_in1k',#
    'beit3_large_patch16_224.in22k_ft_in1k',#
    'convnextv2_base.fcmae_ft_in1k',#
    'aimv2_large_patch14_224.apple_pt',
    'convnext_base.clip_laion2b_augreg_ft_in12k_in1k',#
]
'''
base_adapter_models = [
    'caformer_b36.sail_in22k_ft_in1k',

    'vit_large_patch16_dinov3.lvd1689m',
    'vit_huge_patch14_gap_224.in22k_ijepa',

    'eva02_base_patch14_448.mim_in22k_ft_in22k_in1k',
    'vit_so400m_patch14_siglip_gap_224.pali2_10b_pt',
    'aimv2_large_patch14_224.apple_pt',
    
    'vit_pe_core_gigantic_patch14_448.fb',
    'text_pe_core_text',
    'text_qwen3_embedding_4b_bf16',

    'convformer_b36.sail_in22k_ft_in1k',
    'vit_base_patch16_224.augreg_in21k_ft_in1k',
    'vit_base_patch16_clip_224.openai_ft_in1k',
    'convnext_base.fb_in1k',
    'beit3_large_patch16_224.in22k_ft_in1k',
]

if expand:
    models_to_add = [

        #'convnext_base.fb_in1k',
        #'beit3_large_patch16_224.in22k_ft_in1k',
        #'convnextv2_base.fcmae_ft_in1k',
        #'aimv2_large_patch14_224.apple_pt',
        #'convnext_base.clip_laion2b_augreg_ft_in12k_in1k',

        'convformer_b36.sail_in22k_ft_in1k',
        'vit_base_patch16_224.augreg_in21k_ft_in1k',
        'vit_base_patch16_clip_224.openai_ft_in1k',
        'convnext_base.fb_in1k',
        'beit3_large_patch16_224.in22k_ft_in1k',
    ]
    model_names = [*base_adapter_models, *models_to_add]
    if separate_expand:
        zsl_idxs = (torch.arange(len(models_to_add)) + len(base_adapter_models)).tolist()
else:
    model_names = base_adapter_models



# TODO unnecesary, figure out a way to get embed dim/head without instantiating and loading weights for the whole model
#print("building models...")
#models = [timm.create_model(model_name, pretrained=True, num_classes=1000).eval() for model_name in tqdm.tqdm(model_names)]


'''
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
'''

# TODO flexible paths
print("loading train embeds...")
embeds_train = [
    torch.load(
        f'embeds/embeds_cc12m_{model}.pt',
        map_location='cpu'
    ).to(torch.bfloat16) for model in tqdm.tqdm(model_names)
]
model_dims = [embed.shape[1] for embed in embeds_train]
'''
embeds_val = [
    torch.load(
        f'embeds/embeds_in1k_val_{model}.pt',
        map_location='cpu'
    ) for model in model_names
]
'''

class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, embedsList):
        self.embedsList = embedsList
    def __len__(self):
        return len(self.embedsList[0])
    def __getitem__(self, idx):
        return tuple([embed[idx] for embed in self.embedsList])

adapter_hidden_dim = 2048

if expand:
    adapter = Adapter([x.replace('.', '_') for x in base_adapter_models], model_dims[:len(base_adapter_models)], hidden_dim = adapter_hidden_dim,)
    #adapter.load_state_dict(torch.load(out_dir + "adapter_epoch_99.pt", weights_only=True, map_location='cpu'))
    adapter.middle_model.load_state_dict(torch.load(out_dir + "adapter_middle_model_epoch_39.pt", weights_only=True, map_location='cpu'))
    for model in base_adapter_models:
        adapter.load_state_dict_for_one_model(
            model.replace('.', '_'), 
            torch.load(out_dir + f"adapter_{model}_epoch_39.pt", weights_only=True, map_location='cpu')
        )
    adapter.expand([x.replace('.', '_') for x in models_to_add], model_dims[len(base_adapter_models):])
else:
    adapter = Adapter([x.replace('.', '_') for x in model_names], model_dims, hidden_dim = adapter_hidden_dim,)


adapter = adapter.to(device)
print(adapter)
print(model_names)
print([x[0].shape for x in embeds_train])
discriminator = nn.Sequential(
        nn.Linear(adapter.hidden_dim, 512),
        nn.GELU(),
        nn.Linear(512, len(model_names))
    ).to(device)

# TODO these should arguments/cfg file
num_epochs = 100
lr = 1e-4
dc_lr = 3e-5
bs_train = 2**10
grad_accum_iters = 1
bs_val = 250

embeds_ds = EmbeddingDataset(embeds_train)
loader = timm.data.loader.MultiEpochsDataLoader(embeds_ds, batch_size=bs_train, num_workers=16, shuffle=True, pin_memory=True, persistent_workers=True, prefetch_factor=3)
#loader = list(zip(*[torch.split(x, bs_train, 0) for x in embeds_train]))

if expand:
    params_to_optimize = adapter.get_params_for_model_list([x.replace('.', '_') for x in models_to_add])
else:
    params_to_optimize = adapter.parameters()

optimizer = timm.optim.Adan(params_to_optimize, lr=lr, weight_decay=2e-2)
scaler = torch.amp.GradScaler('cuda')
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=lr,
    #steps_per_epoch=math.ceil(len(embeds_train[0]) / bs_train) * num_epochs,
    steps_per_epoch=len(loader),
    epochs=num_epochs,
    pct_start=0.1
)

opt_dc = timm.optim.Adan(discriminator.parameters(), lr=lr, weight_decay=2e-2)
scaler_dc = torch.amp.GradScaler('cuda')
scheduler_dc = optim.lr_scheduler.OneCycleLR(
    opt_dc,
    max_lr=dc_lr,
    #steps_per_epoch=math.ceil(len(embeds_train[0]) / bs_train) * num_epochs,
    steps_per_epoch=len(loader),
    epochs=num_epochs,
    pct_start=0.1
)

#embeds_val = [embed.to(device) for embed in embeds_val]
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

aug_strength = 0.8
for epoch in range(num_epochs):
    loss_train = 0
    loss_dc_train = 0
    dc_correct_train = 0
    adapter.train()
    for i, embeds in enumerate(tqdm.tqdm(loader)):
        embeds = [embed.to(device, non_blocking=True) for embed in embeds]
        optimizer.zero_grad()
        opt_dc.zero_grad()
        with torch.amp.autocast(device_type="cuda", dtype=autocast_dtype):
            loss, loss_dc, loss_dc_pred, dc_acc = losses.pairwise_adapter_loss_with_discriminator(
                adapter,
                discriminator,
                [(embed + (torch.randn_like(embed) * aug_strength * embed.std(dim=0)).detach()).float() for embed in embeds],
                zsl_idxs = zsl_idxs if separate_expand else None
            )
            scaler.scale(loss).backward()
            scaler_dc.scale(loss_dc).backward()
            if((i+1) % grad_accum_iters == 0):
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                scaler_dc.unscale_(opt_dc)
                scaler_dc.step(opt_dc)
                scaler_dc.update()
                scheduler_dc.step()

            loss_train = loss_train + loss.detach() * bs_train
            loss_dc_train = loss_dc_train + loss_dc.detach() * bs_train
            dc_correct_train = dc_correct_train + dc_acc * bs_train

    '''
    loss_val = 0
    loss_dc_val = 0
    dc_correct_val = 0
    adapter.eval()
    for embeds in zip(*[torch.split(x, bs_val, 0) for x in embeds_val]):
        with torch.no_grad():
            loss, loss_dc, loss_dc_pred, dc_acc = losses.pairwise_adapter_loss_with_discriminator(
                adapter,
                discriminator,
                [embed.float() for embed in embeds],
                zsl_idxs = zsl_idxs if separate_expand else None
            )
            loss_val = loss_val + loss * bs_val
            loss_dc_val = loss_dc_val + loss_dc * bs_val
            dc_correct_val = dc_correct_val + dc_acc * bs_val
    '''
    loss_train = loss_train / len(embeds_train[0])
    loss_dc_train = loss_dc_train / len(embeds_train[0])
    dc_acc_train = 100 * dc_correct_train / len(embeds_train[0])
    #loss_val = loss_val / len(embeds_val[0])
    #loss_dc_val = loss_dc_val / len(embeds_val[0])
    #dc_acc_val = 100 * dc_correct_val / len(embeds_val[0])
    print(f'epoch {epoch}: train loss: {loss_train.item()}, train dc loss: {loss_dc_train.item()}, train dc acc: {["{:.4f}".format(x) for x in dc_acc_train]}, ', end="")
    #print(f'val loss: {loss_val.item()}, val dc loss: {loss_dc_val.item()}, val dc acc: {["{:.4f}".format(x) for x in dc_acc_val]}')


    # TODO proper output directory handling
    create_dir(out_dir)
    
    torch.save(adapter.middle_model.state_dict(), out_dir + f'adapter_middle_model_epoch_{epoch}.pt')
    if(epoch > 0):
        os.remove(out_dir + f'adapter_middle_model_epoch_{epoch-1}.pt')
    for model in model_names:
        torch.save(adapter.get_state_dict_for_one_model(model.replace('.', '_')), out_dir + f'adapter_{model}_epoch_{epoch}.pt')
        if(epoch > 0):
            os.remove(out_dir + f'adapter_{model}_epoch_{epoch-1}.pt')
