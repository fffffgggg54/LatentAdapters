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

import io
import csv

out_dir = "outputs/basic_mmID_8192_discriminator1.0_latent1.0_MSE_JointTraining_NoExpansion/"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# Gemini-2.5-pro, format data as image metadata
def create_csv_string(tensor, names):
    """
    Generates a CSV-formatted string from an nxn tensor and a list of n names.

    Args:
        tensor: An 1xn or nxn list of lists (the tensor).
        names: A list of n strings (the names).

    Returns:
        A string in CSV format.
    """
    output = io.StringIO()
    writer = csv.writer(output)

    # Write the header row
    header = ["Model Name"] + names
    writer.writerow(header)
    torch.set_printoptions(sci_mode=False)
    # Write the data rows
    for i, row in enumerate(tensor):
        if len(tensor) == len(names):
            row = [names[i]] + row.tolist()
        elif len(tensor) == 1:
            row = [""] + row.tolist()
        else: break
        writer.writerow(row)

    return output.getvalue()


def plot_heatmap(
    data, 
    labels, 
    title, 
    cbar_label, 
    x_label = "Original backbone of classifier head", 
    y_label = "Backbone model", 
    out_file = "plot.png"
):
    plt.figure(figsize=(16, 14))
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
    data_csv = create_csv_string(data, labels)
    plt.savefig(out_file, metadata = {'Plot data': data_csv})
    print(data_csv)

def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print("Created Directory : ", dir)
    return dir

# FIXME inconsistent checkpoints?
base_adapter_models = [
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


models_to_add = [
    'convnext_base.fb_in1k',
    'beit3_large_patch16_224.in22k_ft_in1k',
    'convnextv2_base.fcmae_ft_in1k',
    'aimv2_large_patch14_224.apple_pt',
    'convnext_base.clip_laion2b_augreg_ft_in12k_in1k',
]
model_names = [*base_adapter_models, *models_to_add]
#model_names = base_adapter_models


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


def train_probe_on_embeddings(
    embeds_train, 
    labels_train, 
    embeds_val, 
    labels_val, 
    use_latents=False, 
    adapter=None, 
    lr=0.003, 
    aug_strength = 0.6, 
    epochs = 100, 
    bs=None, 
    grad_accum_iters=1,
    shared=False
):
    emb_ds_train = EmbeddingDataset([labels_train, *embeds_train])
    emb_ds_val = EmbeddingDataset([labels_val, *embeds_val])
    bs_train = bs or 2**14
    bs_train = bs_train // grad_accum_iters
    loader_train = timm.data.loader.MultiEpochsDataLoader(emb_ds_train, batch_size=bs_train, num_workers=10, shuffle=True, pin_memory=True, persistent_workers=True, prefetch_factor=1)
    loader_val = timm.data.loader.MultiEpochsDataLoader(emb_ds_val, batch_size=bs_train, num_workers=10, shuffle=False, pin_memory=True, persistent_workers=True, prefetch_factor=1)
    # FIXME hardcoded class count
    if shared:
        probes = nn.Linear(adapter.hidden_dim, 1000).to(device)
    else:
        if use_latents:
            probes = nn.ModuleList([nn.Linear(adapter.hidden_dim, 1000) for embed in embeds_train]).to(device)
        else:
            # N * [B, d_model]
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
        for i, embedBatch in enumerate(loader_train):
            with torch.autocast(device.type, dtype=autocast_dtype):
                embedBatch = [embed.to(device, non_blocking=True) for embed in embedBatch]
                labelBatch = embedBatch[0]
                embedBatch = embedBatch[1:]
                if use_latents:
                    embedBatch = adapter.fw_all_embeds_to_latent(embedBatch)
                # noise aug
                with torch.no_grad():
                    embedBatch = [embed + torch.randn_like(embed) * aug_strength * embed.std(dim=0) for embed in embedBatch]
                optimizer.zero_grad()
                if shared:
                    outputs = probes(torch.stack(embedBatch, dim=0))
                else:
                    outputs = [probe(embed).float() for probe, embed in zip(probes, embedBatch)]
                loss = torch.stack([criterion(output, labelBatch) for output in outputs])
                loss = loss.mean()
                scaler.scale(loss).backward()
                if((i+1) % grad_accum_iters == 0):
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
                    if use_latents:
                        embedBatch = adapter.fw_all_embeds_to_latent(embedBatch)
                    if shared:
                        #outputs = probes(torch.stack(embedBatch, dim=0))
                        outputs = probes(embedBatch)
                        # FIXME why not stack here??
                    else:
                        outputs = [probe(embed).float() for probe, embed in zip(probes, embedBatch)]
                    correct_val = correct_val + torch.stack([(labelBatch == torch.argmax(output, dim=1)).sum() for output in outputs], dim=0)
                    total_val += len(labelBatch)

        print(f'Epoch {epoch+1}: Train Accuracy: {correct_train/total_train}, Val Accuracy: {correct_val/total_val}')
    return probes, correct_val/total_val

def count_correct(models, adapter, embeds_val, labels_val, shared=False, use_latents=False):
    # FIXME old and wrong inputs description!!
    # case stock heads: models is list of models, shared=False, embeds=list of embeds, latents=None
    # case embedding probes: models is modulelist of probes, shared=False, embeds=list of embeds, latents=None
    # case latent probes: models is modulelist of probes, shared=False, embeds=None, latents=list of latents
    # case shared latent probe: models is latent probe, shared=True, embeds=None, latents=list of latents

    bs_val = 100
    # no translation, own head
    correct_default = torch.zeros(len(adapter.model_names), device=device)
    # translate to own/other embedding space
    # own on diagonal, other elsewhere
    # in_model, out_model
    correct_adapted = torch.zeros(len(adapter.model_names), len(adapter.model_names), device=device)
    # translate self -> other -> self
    # self, other
    correct_cycle = torch.zeros(len(adapter.model_names), len(adapter.model_names), device=device)

    total = 0

    # N * [B, d_model]
    embeds_val = [embed.to(device) for embed in embeds_val]
    # [B]
    labels_val = labels_val.to(device)

    for embeds, labels in zip(zip(*[torch.split(x, bs_val, 0) for x in embeds_val]), torch.split(labels_val, bs_val, 0)):
        with torch.no_grad():
            latents = adapter.fw_all_embeds_to_latent(embeds)
            adapted_embeds = adapter.fw_latent_to_all_embeds(latents)
            cycle_latents = adapter.fw_cycle_latents(adapted_embeds)
            if use_latents:
                if shared:
                    # FIXME probably should be default instead of adapted
                    # stride on out_model, stack on dim 1 for in_model, out_model
                    correct_adapted += (labels.unsqueeze(0) == torch.argmax(models(latents), dim=2)).sum(1).unsqueeze(1)

                    # stride on self, stack on dim 0 for self, other
                    correct_cycle += torch.stack([(labels == torch.argmax(models(embed), dim=2)).sum(1) for embed in cycle_latents], dim=0)
                else:
                    outputs = [model(latent).float() for model, latent in zip(models, latents)]
                    correct_default += torch.stack([(labels == torch.argmax(output, dim=1)).sum() for output in outputs], dim=0)

                    # stride on out_model, stack on dim 1 for in_model, out_model
                    correct_adapted += torch.stack([(labels == torch.argmax(model(latents), dim=2)).sum(1) for model in models], dim=1)

                    # stride on self, stack on dim 0 for self, other
                    correct_cycle += torch.stack([(labels == torch.argmax(model(latent), dim=2)).sum(1) for model, latent in zip(models, cycle_latents)], dim=0)
                
            else:
                # N * [N, B, d_in]
                # in_model, cycle_model, batch_idx, dim 
                self_cycle_embeds = adapter.fw_cycle_latents_to_self_cycle_embeds(cycle_latents)

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
        out_file = out_dir + "cls_adapter_with_stock_head.png"
    )

    plot_heatmap(
        adapted_vs_self_adapt_backbone.cpu().numpy(), 
        model_names, 
        "Change in Top-1 Accuracy of Backbone vs. Adapting to Self", 
        "Change in Top-1 Accuracy", 
        out_file = out_dir + "cls_adapter_with_stock_head_vs_backbone_adapted_to_self.png"
    )

    plot_heatmap(
        adapted_vs_self_adapt_head.cpu().numpy(), 
        model_names, 
        "Change in Top-1 Accuracy of Head vs. Adapting to Self", 
        "Change in Top-1 Accuracy", 
        out_file = out_dir + "cls_adapter_with_stock_head_vs_head_adapted_to_self.png"
    )

    plot_heatmap(
        adapted_vs_no_adapt_backbone.cpu().numpy(), 
        model_names, 
        "Change in Top-1 Accuracy of Backbone vs. No Adapter", 
        "Change in Top-1 Accuracy", 
        out_file = out_dir + "cls_adapter_with_stock_head_vs_backbone_without_adapter.png"
    )

    plot_heatmap(
        adapted_vs_no_adapt_head.cpu().numpy(), 
        model_names, 
        "Change in Top-1 Accuracy of Head vs. No Adapter", 
        "Change in Top-1 Accuracy", 
        out_file = out_dir + "cls_adapter_with_stock_head_vs_head_without_adapter.png"
    )

    plot_heatmap(
        top1_cycle.cpu().numpy(), 
        model_names, 
        "Cycle Top-1 Accuracy", 
        "Top-1 Accuracy",
        x_label = "Cycle model",
        out_file = out_dir + "cls_adapter_with_stock_head_cycle.png"
    )

    plot_heatmap(
        cycle_vs_self_adapt.cpu().numpy(), 
        model_names, 
        "Change in Cycle Top-1 Accuracy vs Adapting to Self", 
        "Change in Top-1 Accuracy",
        x_label = "Cycle model",
        out_file = out_dir + "cls_adapter_with_stock_head_cycle_vs_adapting_to_self.png"
    )

    plot_heatmap(
        cycle_vs_self_cycle.cpu().numpy(), 
        model_names, 
        "Change in Cycle Top-1 Accuracy vs Self Cycle", 
        "Change in Top-1 Accuracy",
        x_label = "Cycle model",
        out_file = out_dir + "cls_adapter_with_stock_head_cycle_vs_self_cycle.png"
    )

    plot_heatmap(
        cycle_vs_no_adapt.cpu().numpy(), 
        model_names, 
        "Change in Cycle Top-1 Accuracy vs No Adapter", 
        "Change in Top-1 Accuracy",
        x_label = "Cycle model",
        out_file = out_dir + "cls_adapter_with_stock_head_cycle_vs_no_adapt.png"
    )

    return correct_default, correct_adapted, correct_cycle, total

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
        x_label = "Backbone of embedding probe",
        out_file = out_dir + "cls_adapter_with_embedding_probes.png"
    )

    return correct_default, correct_adapted, correct_cycle, total

def eval_cls_with_latent_probes(probes, adapter, embeds_val, labels_val):
    correct_default, correct_adapted, correct_cycle, total = count_correct(probes, adapter, embeds_val, labels_val, use_latents=True)

    top1_adapted = correct_adapted/total
    top1_default = correct_default/total
    top1_cycle = correct_cycle/total

    
    plot_heatmap(
        top1_adapted.cpu().numpy(), 
        model_names, 
        "Top-1 Accuracy Adapting to Latent Linear Probes", 
        "Top-1 Accuracy",
        x_label = "Backbone of latent probe",
        out_file = out_dir + "cls_adapter_with_latent_probes.png"
    )

    return correct_default, correct_adapted, correct_cycle, total

def eval_cls_with_shared_latent_probe(probe, adapter, embeds_val, labels_val, top1_adapted_probes):
    correct_default, correct_adapted, correct_cycle, total = count_correct(probe, adapter, embeds_val, labels_val, use_latents=True, shared=True)

    top1_adapted = correct_adapted/total
    top1_default = correct_default/total
    top1_cycle = correct_cycle/total

    plot_heatmap(
        top1_adapted[:,0].unsqueeze(0).cpu().numpy(),
        model_names,
        "Top-1 Accuracy of Shared Latent Linear Probe", 
        "Top-1 Accuracy",
        y_label = None,
        out_file = out_dir + "cls_adapter_with_shared_latent_probe.png"
    )

    plot_heatmap(
        (top1_adapted[:,0].unsqueeze(1) - top1_adapted_probes).cpu().numpy(),
        model_names,
        "Top-1 Accuracy Change from Non-shared Latent Probes to Shared Latent Probe",
        "Top-1 Accuracy",
        x_label = "Backbone of latent probe",
        out_file = out_dir + "cls_adapter_with_shared_latent_probe_vs_separate_latent_probes.png"
    )

    return correct_default, correct_adapted, correct_cycle, total

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
    labels_train = torch.load('labels_in1k_train.pt', map_location='cpu')

    embeds_val = [
        torch.load(
            f'embeds/embeds_in1k_val_{model.default_cfg['architecture']}.{model.default_cfg['tag']}.pt',
            map_location='cpu'
        ) for model in models
    ]

    labels_val = torch.load('labels_in1k_val.pt', map_location='cpu')
    adapter = Adapter([x.replace('.', '_') for x in model_names], model_dims)
    #adapter.load_state_dict(torch.load('adapters/adapter_latent_mse_no_discriminator_20251015-111701_epoch_99.pt', weights_only=True))
    #adapter.load_state_dict(torch.load('adapters/adapter_20251014_weights_only.pt', weights_only=True))
    #adapter.load_state_dict(torch.load(out_dir + "adapter_epoch_99.pt", weights_only=True))
    #adapter.expand([x.replace('.', '_') for x in models_to_add], model_dims[len(base_adapter_models):])
    for model in model_names:
        adapter.load_state_dict_for_one_model(
            model.replace('.', '_'), 
            torch.load(out_dir + f"adapter_{model}_epoch_39.pt", weights_only=True, map_location='cpu')
        )
    adapter.middle_model.load_state_dict(torch.load(out_dir + "adapter_middle_model_epoch_39.pt", weights_only=True, map_location='cpu'))
    adapter = adapter.to(device)
    #adapter = torch.load('adapters/adapter_20251014-192543_epoch_99.pt', weights_only=False)
    #torch.save(adapter.state_dict(), 'adapters/adapter_20251014_weights_only.pt')
    adapter = adapter.to(device)

    print(adapter)
    print(model_names)
    print([x[0].shape for x in embeds_train])

    # in1k only
    print("\nstock head eval\n")
    eval_cls_with_stock_heads(models, adapter, embeds_val, labels_val)

    # for each probe dataset
    # TODO everything after here should be in a foreach loop for sets of embeds/labels for each probe dataset
    # TODO clip-based zero shot also goes here
    # TODO early-fusion VLM zero shot also goes here
    print("\ntrain and evaluate embedding probes\n")
    # TODO impl model selection and hparam sweeps for probes like oai clip/other probe papers
    model_embedding_probes, _ = train_probe_on_embeddings(
        embeds_train, 
        labels_train, 
        embeds_val, 
        labels_val, 
        lr=0.003, 
        aug_strength = 0.6, 
        epochs = 20, 
        bs=2**14, 
        grad_accum_iters = 2,
    )
    eval_cls_with_embedding_probes(model_embedding_probes, adapter, embeds_val, labels_val)

    # for each probe dataset
    print("\ntrain and evaluate non-shared latent probes")
    latent_probes, _ = train_probe_on_embeddings(
        embeds_train, 
        labels_train, 
        embeds_val, 
        labels_val, 
        use_latents=True,
        adapter=adapter,
        lr=0.003, 
        aug_strength = 0.6, 
        epochs = 20, 
        bs=2**14, 
        grad_accum_iters = 2,
    )
    latent_probe_results = eval_cls_with_latent_probes(latent_probes, adapter, embeds_val, labels_val)
    # FIXME there has to be a better way to do this. Can we run all evals, then plot later with all outputs?
    top1_adapted_probes = latent_probe_results[1]/latent_probe_results[3]

    # for each probe dataset
    print("\ntrain and evaluate shared latent probe")
    shared_latent_probe, _ = train_probe_on_embeddings(
        embeds_train, 
        labels_train, 
        embeds_val, 
        labels_val, 
        use_latents=True,
        adapter=adapter,
        lr=0.003, 
        aug_strength = 0.6, 
        epochs = 20, 
        bs=2**14, 
        grad_accum_iters = 2,
        shared=True
    )
    eval_cls_with_shared_latent_probe(shared_latent_probe, adapter, embeds_val, labels_val, top1_adapted_probes)