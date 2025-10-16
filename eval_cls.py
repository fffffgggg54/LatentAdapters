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

def plot_heatmap(data, labels, title, cbar_label, x_label = "Original backbone of classifier head", y_label = "Backbone model", out_file = "plot.png"):
    plt.figure(figsize=(16, 10))
    heatmap = sns.heatmap(data,
                        xticklabels=labels,
                        yticklabels=labels if len(data) == len(labels) else False,
                        cmap='RdBu',
                        center=0,
                        annot=True,
                        fmt='.2%',
                        square=True,
                        cbar_kws={'label': cbar_label})

    plt.title(title, fontsize=20, pad=20)
    plt.xlabel(x_label, fontsize=16, labelpad=15)
    plt.ylabel(y_label, fontsize=16, labelpad=15)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(out_file)

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

@torch.compile()
def fw_enc(model, x):
    x = model.forward_features(x)
    return model.forward_head(x, pre_logits=True)

def fw_head(model, x):
    # case probe
    if isinstance(model, nn.Linear):
        model.to(device)
        return model(x)
    else:
        model.head.to(device)
        if hasattr(model.head, 'fc'):
            # metaformer
            return model.head.fc(x)
        else:
            # vit
            return model.head(x)


class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, embedsList):
        self.embedsList = embedsList
    def __len__(self):
        return len(self.embedsList[0])
    def __getitem__(self, idx):
        return tuple([embed[idx] for embed in self.embedsList])

def count_correct(models, adapter, embeds_val, labels_val):
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
            # output shape of N * [N, B, d_out]
            # out_model, in_model, batch_idx, dim
            adapted_embeds = adapter.adapt_all_to_all(embeds)

            # N * [N, B, d_in]
            # in_model, cycle_model, batch_idx, dim 
            self_cycle_embeds = adapter.fw_self_cycle_embeds(adapted_embeds)

            correct_default += torch.stack([(labels == torch.argmax(fw_head(model.float(), embed.float()), dim=1)).sum() for model, embed in zip(models, embeds)], dim=0)

            # stride on out_model, stack on dim 1 for in_model, out_model
            correct_adapted += torch.stack([(labels.unsqueeze(0) == torch.argmax(fw_head(model.float(), embed.float()), dim=-1)).sum(-1) for model, embed in zip(models, adapted_embeds)], dim=1)

            # stride on self, stack on dim 0 for self, other
            correct_cycle += torch.stack([(labels.unsqueeze(0) == torch.argmax(fw_head(model.float(), embed.float()), dim=-1)).sum(-1) for model, embed in zip(models, self_cycle_embeds)], dim=0)

            total += len(labels)
    print(f'default: {correct_default/total}, adapted: {correct_adapted/total}, cycle: {correct_cycle/total}')
    return correct_default, correct_adapted, correct_cycle, total

def eval_cls_with_stock_heads(models, adapter, embeds_val, labels_val):
    correct_default, correct_adapted, correct_cycle, total = count_correct(models, adapter, embeds_val, labels_val)

    top1_adapted = correct_adapted/total
    top1_default = correct_default/total
    top1_cycle = correct_cycle/total

    adapted_vs_self_adapt_backbone = top1_adapted-top1_adapted.diag().unsqueeze(1)
    adapted_vs_self_adapt_head = top1_adapted-top1_adapted.diag().unsqueeze(0)
    adapted_vs_no_adapt_backbone = top1_adapted-top1_default.unsqueeze(1)
    adapted_vs_no_adapt_head = top1_adapted-top1_default.unsqueeze(0)

    
    cycle_vs_self_adapt = top1_cycle-top1_adapted.diag().unsqueeze(1)
    cycle_vs_self_cycle = top1_cycle-top1_cycle.diag().unsqueeze(1)
    cycle_vs_no_adapt = top1_cycle-top1_default.unsqueeze(1)


    plot_heatmap(
        top1_adapted.cpu().numpy(), 
        model_names, 
        "Top-1 Accuracy adapting to stock heads", 
        "Top-1 Accuracy", 
        out_file = "outputs/cls_adapter_with_stock_head.png"
    )

    plot_heatmap(
        adapted_vs_self_adapt_backbone.cpu().numpy(), 
        model_names, 
        "Change in Top-1 Accuracy of Backbone vs. Adapting to Self", 
        "Change in Top-1 Accuracy", 
        out_file = "outputs/cls_adapter_with_stock_head_vs_backbone_adapted_to_self.png"
    )

    plot_heatmap(
        adapted_vs_self_adapt_head.cpu().numpy(), 
        model_names, 
        "Change in Top-1 Accuracy of Head vs. Adapting to Self", 
        "Change in Top-1 Accuracy", 
        out_file = "outputs/cls_adapter_with_stock_head_vs_head_adapted_to_self.png"
    )

    plot_heatmap(
        adapted_vs_no_adapt_backbone.cpu().numpy(), 
        model_names, 
        "Change in Top-1 Accuracy of Backbone vs. No Adapter", 
        "Change in Top-1 Accuracy", 
        out_file = "outputs/cls_adapter_with_stock_head_vs_backbone_without_adapter.png"
    )

    plot_heatmap(
        adapted_vs_no_adapt_head.cpu().numpy(), 
        model_names, 
        "Change in Top-1 Accuracy of Head vs. No Adapter", 
        "Change in Top-1 Accuracy", 
        out_file = "outputs/cls_adapter_with_stock_head_vs_head_without_adapter.png"
    )

    plot_heatmap(
        top1_cycle.cpu().numpy(), 
        model_names, 
        "Cycle Top-1 Accuracy", 
        "Top-1 Accuracy",
        x_label = "Cycle model",
        out_file = "outputs/cls_adapter_with_stock_head_cycle.png"
    )

    plot_heatmap(
        cycle_vs_self_adapt.cpu().numpy(), 
        model_names, 
        "Change in Cycle Top-1 Accuracy vs Adapting to Self", 
        "Change in Top-1 Accuracy",
        x_label = "Cycle model",
        out_file = "outputs/cls_adapter_with_stock_head_cycle_vs_adapting_to_self.png"
    )

    plot_heatmap(
        cycle_vs_self_cycle.cpu().numpy(), 
        model_names, 
        "Change in Cycle Top-1 Accuracy vs Self Cycle", 
        "Change in Top-1 Accuracy",
        x_label = "Cycle model",
        out_file = "outputs/cls_adapter_with_stock_head_cycle_vs_self_cycle.png"
    )

    plot_heatmap(
        cycle_vs_no_adapt.cpu().numpy(), 
        model_names, 
        "Change in Cycle Top-1 Accuracy vs No Adapter", 
        "Change in Top-1 Accuracy",
        x_label = "Cycle model",
        out_file = "outputs/cls_adapter_with_stock_head_cycle_vs_no_adapt.png"
    )

def train_probe_on_embeddings(embeds_train, labels_train, embeds_val, labels_val, lr=0.003, aug_strength = 0.6, epochs = 100, bs=None):
    emb_ds_train = EmbeddingDataset([labels_train, *embeds_train])
    emb_ds_val = EmbeddingDataset([labels_val, *embeds_val])
    bs_train = bs or 2**14
    loader_train = timm.data.loader.MultiEpochsDataLoader(emb_ds_train, batch_size=bs_train, num_workers=16, shuffle=True, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    loader_val = timm.data.loader.MultiEpochsDataLoader(emb_ds_val, batch_size=bs_train, num_workers=16, shuffle=False, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    # FIXME hardcoded class count
    probes = nn.ModuleList([nn.Linear(embed.shape[1], 1000) for embed in embeds_train]).to(device)
    optimizer = timm.optim.Adan(probes.parameters(), lr=lr, weight_decay=1e-5)
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(loader_train), epochs=epochs, pct_start=0.1)
    for epoch in range(epochs):
        correct_train = 0
        total_train = 0
        correct_val = 0
        total_val = 0
        for embedBatch in loader_train:
            with torch.autocast(device.type, dtype=autocast_dtype):
                embedBatch = [embed.to(device, non_blocking=True) for embed in embedBatch]
                labelBatch = embedBatch[0]
                embedBatch = embedBatch[1:]
                # noise aug
                with torch.no_grad():
                    embedBatch = [embed + torch.randn_like(embed) * aug_strength * embed.std(dim=0) for embed in embedBatch]
                optimizer.zero_grad()
                outputs = [probe(embed).float() for probe, embed in zip(probes, embedBatch)]
                loss = torch.stack([criterion(output, labelBatch) for output in outputs])
                loss = loss.mean()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                correct_train = correct_train + torch.stack([(labelBatch == torch.argmax(output, dim=1)).sum() for output in outputs], dim=0)
                total_train += len(labelBatch)
        for embedBatch in loader_val:
            with torch.autocast(device.type, dtype=autocast_dtype):
                with torch.inference_mode():
                    embedBatch = [embed.to(device, non_blocking=True) for embed in embedBatch]
                    labelBatch = embedBatch[0]
                    embedBatch = embedBatch[1:]
                    outputs = [probe(embed).float() for probe, embed in zip(probes, embedBatch)]
                    correct_val = correct_val + torch.stack([(labelBatch == torch.argmax(output, dim=1)).sum() for output in outputs], dim=0)
                    total_val += len(labelBatch)

        print(f'Epoch {epoch+1}: Train Accuracy: {correct_train/total_train}, Val Accuracy: {correct_val/total_val}')
    return probes, correct_val/total_val

def eval_cls_with_embedding_probes(probes, adapter, embeds_val, labels_val):
    correct_default, correct_adapted, correct_cycle, total = count_correct(probes, adapter, embeds_val, labels_val)

    top1_adapted = correct_adapted/total
    top1_default = correct_default/total
    top1_cycle = correct_cycle/total
    loss_vs_self_backbone = top1_adapted-top1_adapted.diag().unsqueeze(1)
    loss_vs_self_head = top1_adapted-top1_adapted.diag().unsqueeze(0)
    loss_vs_default_backbone = top1_adapted-top1_default.unsqueeze(1)
    loss_vs_default_head = top1_adapted-top1_default.unsqueeze(0)


    plot_heatmap(
        top1_adapted.cpu().numpy(), 
        model_names, 
        "Top-1 Accuracy adapting to embedding probes", 
        "Top-1 Accuracy",
        x_label = "Backbone of linear probe",
        out_file = "outputs/cls_adapter_with_embedding_probes.png"
    )

def compute_latents(embeds, adapter, bs=1000):
    embeds_ds = EmbeddingDataset(embeds)
    loader_train = torch.utils.data.DataLoader(embeds_ds, batch_size=bs, num_workers=8, shuffle=False)
    all_latents = []
    for embeds in tqdm.tqdm(loader):
        embeds = [embed.to(device, non_blocking=True) for embed in embeds]
        with torch.amp.autocast(device_type="cuda", dtype=autocast_dtype):
            with torch.autograd.grad_mode.inference_mode():
                # [N, B, d_h]
                latents = adapter.fw_all_embeds_to_latent(embeds)
                all_latents.append(latents.to('cpu', non_blocking=True))
    with torch.autograd.grad_mode.inference_mode():
        latents = torch.cat(all_latents, dim=1)
    return latents

if __name__ == '__main__':
    # TODO unnecesary, figure out a way to get embed dim/head without instantiating and loading weights for the whole model
    print("building models...")
    models = [timm.create_model(model_name, pretrained=True, num_classes=1000).eval() for model_name in tqdm.tqdm(model_names)]
    model_dims = [model.num_features for model in models]

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
    adapter = Adapter([x.replace('.', '_') for x in model_names], model_dims)
    adapter.load_state_dict(torch.load('adapters/adapter_latent_mse_no_discriminator_20251015-111701_epoch_99.pt', weights_only=True))
    #adapter.load_state_dict(torch.load('adapters/adapter_20251014_weights_only.pt', weights_only=True))
    #adapter = torch.load('adapters/adapter_20251014-192543_epoch_99.pt', weights_only=False)
    #torch.save(adapter.state_dict(), 'adapters/adapter_20251014_weights_only.pt')
    adapter = adapter.to(device)

    print(adapter)
    print(model_names)
    print([x[0].shape for x in embeds_train])

    eval_cls_with_stock_heads(models, adapter, embeds_val, labels_val)

    model_embedding_probes, _ = train_probe_on_embeddings(
        embeds_train, 
        labels_train, 
        embeds_val, 
        labels_val, 
        lr=0.003, 
        aug_strength = 0.6, 
        epochs = 20, 
        bs=None
    )