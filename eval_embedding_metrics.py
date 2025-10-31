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

import torchmetrics

out_dir = "outputs/scratch_mmID_2048_100epoch_discriminator0.0_latent1.0_MSE_JointTraining/"
adapter_hidden_dim = 2048
epoch = 99

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
    out_file = "plot.png",
    fmt='.2%'
):
    plt.figure(figsize=(16, 14))
    heatmap = sns.heatmap(data,
                        xticklabels=labels,
                        yticklabels=labels if len(data) == len(labels) else False,
                        cmap='RdBu',
                        center=0,
                        annot=True,
                        fmt=fmt,
                        square=True,
                        cbar_kws={'label': cbar_label})

    plt.title(title, fontsize=20, pad=20)
    plt.xlabel(x_label, fontsize=16, labelpad=15)
    plt.ylabel(y_label, fontsize=16, labelpad=15)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    data_csv = create_csv_string(data, labels)
    plt.savefig(out_file, metadata = {'Plot data': data_csv})
    #print(data_csv)

def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print("Created Directory : ", dir)
    return dir

model_names = [
    'caformer_b36.sail_in22k_ft_in1k',

    'vit_large_patch16_dinov3.lvd1689m',
    'vit_huge_patch14_gap_224.in22k_ijepa',

    'eva02_base_patch14_448.mim_in22k_ft_in22k_in1k',
    'vit_so400m_patch14_siglip_gap_224.pali2_10b_pt',
    'aimv2_large_patch14_224.apple_pt',
    
    'vit_pe_core_gigantic_patch14_448.fb',

    'convformer_b36.sail_in22k_ft_in1k',
    'vit_base_patch16_224.augreg_in21k_ft_in1k',
    'vit_base_patch16_clip_224.openai_ft_in1k',
    'convnext_base.fb_in1k',
    'beit3_large_patch16_224.in22k_ft_in1k',
]

def compute_embedding_metrics(adapter, embeds_val, metrics):
    bs_val = 100

    results_latents = {k: torch.zeros(len(embeds_val), len(embeds_val)) for k in metrics.keys()}
    results_embeds = {k: torch.zeros(len(embeds_val), len(embeds_val)) for k in metrics.keys()}

    metrics_latents = metrics.clone()
    metrics_embeds = metrics.clone()

    embeds_val = [embed.to(device).float() for embed in embeds_val]

    # TODO symmetric matrices?
    for in_model_idx in range(len(embeds_val)):
        for out_model_idx in range(len(embeds_val)):
            metrics_latents.reset()
            metrics_embeds.reset()
            for embeds in zip(*[torch.split(embeds_val[in_model_idx], bs_val, 0), torch.split(embeds_val[out_model_idx], bs_val, 0)]):
                in_latents = adapter.fw_one_embed_to_latent(embeds[0], adapter.model_names[in_model_idx])
                out_latents = adapter.fw_one_embed_to_latent(embeds[1], adapter.model_names[out_model_idx])
                metrics_latents.update(in_latents, out_latents)
                adapted_embeds = adapter.fw_latent_to_one_embed(in_latents, adapter.model_names[out_model_idx])
                metrics_embeds.update(adapted_embeds, embeds[1])
            
            pair_results_latents = metrics_latents.compute()
            pair_results_embeds = metrics_embeds.compute()
            for key in metrics.keys():
                results_latents[key][in_model_idx][out_model_idx] = pair_results_latents[key]
                results_embeds[key][in_model_idx][out_model_idx] = pair_results_embeds[key]

    print(results_latents)
    print(results_embeds)
    return results_latents, results_embeds

if __name__ == '__main__':
    # TODO flexible paths
    print("loading train embeds...")

    embeds_val = [
        torch.load(
            f'embeds/embeds_in1k_val_{model}.pt',
            map_location='cpu'
        ) for model in model_names
    ]

    model_dims = [embed.shape[1] for embed in embeds_val]

    adapter = Adapter([x.replace('.', '_') for x in model_names], model_dims, hidden_dim = adapter_hidden_dim)
    #adapter.load_state_dict(torch.load('adapters/adapter_latent_mse_no_discriminator_20251015-111701_epoch_99.pt', weights_only=True))
    #adapter.load_state_dict(torch.load('adapters/adapter_20251014_weights_only.pt', weights_only=True))
    #adapter.load_state_dict(torch.load(out_dir + "adapter_epoch_99.pt", weights_only=True))
    #adapter.expand([x.replace('.', '_') for x in models_to_add], model_dims[len(base_adapter_models):])
    for model in model_names:
        adapter.load_state_dict_for_one_model(
            model.replace('.', '_'), 
            torch.load(out_dir + f"adapter_{model}_epoch_{epoch}.pt", weights_only=True, map_location='cpu')
        )
    adapter.middle_model.load_state_dict(torch.load(out_dir + f"adapter_middle_model_epoch_{epoch}.pt", weights_only=True, map_location='cpu'))
    adapter = adapter.to(device).float()
    #adapter = torch.load('adapters/adapter_20251014-192543_epoch_99.pt', weights_only=False)
    #torch.save(adapter.state_dict(), 'adapters/adapter_20251014_weights_only.pt')
    adapter = adapter.to(device)

    print(adapter)
    print(model_names)
    print([x[0].shape for x in embeds_val])

    metrics_to_track = {
        "Uniform R-squared": torchmetrics.R2Score(multioutput='uniform_average'),
        "Weighted R-squared": torchmetrics.R2Score(multioutput='variance_weighted'),
        "Uniform Explained Variance": torchmetrics.ExplainedVariance(multioutput='uniform_average'),
        "Weighted Explained Variance": torchmetrics.ExplainedVariance(multioutput='variance_weighted'),
        #"Cosine Similarity": torchmetrics.CosineSimilarity(reduction = 'mean')
    }

    metric_tracker = torchmetrics.MetricCollection(metrics_to_track).to(device)

    results_latents, results_embeds = compute_embedding_metrics(adapter, embeds_val, metric_tracker)

    for key in metric_tracker.keys():
        
        plot_heatmap(
            results_latents[key],
            model_names,
            "Pairwise " + key + " of Latents",
            key,
            x_label="Backbone 2",
            y_label="Backbone 1",
            out_file=out_dir + "Pairwise " + key + " of Latents.png",
            fmt=".3"
        )
        
        plot_heatmap(
            results_embeds[key],
            model_names,
            "Pairwise " + key + " of Adapted Embeds",
            key,
            x_label="Target model",
            y_label="Backbone model",
            out_file=out_dir + "Pairwise " + key + " of Adapted Embeds.png",
            fmt=".3"
        )
        

