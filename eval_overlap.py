# kNN purity and silhouette coefficient
import os

# adjust as needed
os.environ["OPENBLAS_NUM_THREADS"] = "64" 
os.environ["MKL_NUM_THREADS"] = "64" 
os.environ["OMP_NUM_THREADS"] = "64" 

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


from adapter import Adapter
import losses

import io
import csv

import seaborn as sns
import numpy as np
import colormaps as cmaps
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_samples
from itertools import combinations


out_dir = "outputs/final_mmID_4096_10epoch_discriminator1.0_latent1.0_MSE_JointTraining/"
adapter_hidden_dim = 4096
epoch = 9

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print("Created Directory : ", dir)
    return dir
'''
model_names = [
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
'''
    
'''
model_names = [
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
model_names = [
    'caformer_b36.sail_in22k_ft_in1k',

    'vit_large_patch16_dinov3.lvd1689m',

    'eva02_base_patch14_448.mim_in22k_ft_in22k_in1k',
    'aimv2_large_patch14_224.apple_pt',

    'convformer_b36.sail_in22k_ft_in1k',
    'vit_base_patch16_224.augreg_in21k_ft_in1k',
    'vit_base_patch16_clip_224.openai_ft_in1k',
    'convnext_base.fb_in1k',
    'beit3_large_patch16_224.in22k_ft_in1k',

    'vit_pe_core_gigantic_patch14_448.fb',
    'text_pe_core_text',

    'text_qwen3_embedding_4b_bf16',
    
    'resnet152.tv2_in1k',
    'mobilenetv4_hybrid_large.e600_r384_in1k',
    'mobilenetv4_conv_large.e600_r384_in1k',
    'maxvit_base_tf_384.in21k_ft_in1k',
    'swin_large_patch4_window12_384.ms_in22k_ft_in1k',
    'regnety_080.ra3_in1k',

]

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
    plt.close()
    #print(data_csv)

def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print("Created Directory : ", dir)
    return dir

# Gemini-3-pro, modified
def compute_knn_purity(X, y, tgt_cls, k=15):
    """
    Calculates the average fraction of k-nearest neighbors that share 
    the same label as the query point.
    """
    n_samples = X.shape[0]
    # Handle edge case where subset is smaller than k
    effective_k = min(k, n_samples - 1)
    if effective_k < 1:
        return 0.0

    # Fit Nearest Neighbors
    # We look for k+1 neighbors because the closest neighbor is the point itself
    nbrs = NearestNeighbors(n_neighbors=effective_k + 1).fit(X)
    indices = nbrs.kneighbors(X, return_distance=False)

    # Exclude the first column (the point itself)
    neighbor_indices = indices[:, 1:]
    
    # Get labels of neighbors
    neighbor_labels = y[neighbor_indices]

    if tgt_cls is not None:
        # Filter rows where the query point belongs to tgt_cls
        mask_tgt = (y == tgt_cls)
        if not np.any(mask_tgt):
            return 0.0
            
        relevant_neighbor_labels = neighbor_labels[mask_tgt]
        
        # Check matches
        matches = (relevant_neighbor_labels == tgt_cls)
        
        # Purity = mean match rate per point, then mean across points
        return matches.mean(axis=1).mean()
    else:
        purity_per_point = np.mean(neighbor_labels == y[:, np.newaxis], axis=1)
        return purity_per_point


# Gemini-3-pro, modified
def pairwise_overlap_metrics(X, y, k=15):
    """
    Computes KNN Purity and Silhouette Score for every pair of classes.
    Input:
        X: [N, d] numpy array
        y: [N, 1] or [N] numpy array
    """
    # Ensure y is 1D array
    y = np.array(y).ravel()
    classes = np.unique(y)
    results_pairwise = {k: torch.zeros(len(classes), len(classes)) for k in {'purity', 'silhouette'}}
    results_per_class = {k: torch.zeros(1, len(classes)) for k in {'purity', 'silhouette'}}
    
    print(f"Computing metrics for {len(classes)} classes (Pairwise)...")

    for c2 in range(len(embeds_val)):
        for c1 in range(len(embeds_val)):
            print(f"({c2}, {c1})")
            # Create mask for only the two current classes
            mask = (y == c1) | (y == c2)
            
            X_pair = X[mask]
            y_pair = y[mask]
            
            # 1. Compute Silhouette Score (Requires at least 2 distinct labels)
            # returns -1 (wrong cluster) to +1 (dense, well separated)
            sil_score = silhouette_samples(X_pair, y_pair)[y_pair == c1].mean() if c1 != c2 else 1.0
            
            # 2. Compute KNN Purity
            # returns 0 to 1
            purity_score = compute_knn_purity(X_pair, y_pair, c1, k=k)
            
            results_pairwise['silhouette'][c1][c2] = sil_score
            results_pairwise['purity'][c1][c2] = purity_score

    sil_score = silhouette_samples(X, y)
    purity_score = compute_knn_purity(X, y, None, k=k)
    for tgt_cls in classes:
        results_per_class['silhouette'][0][tgt_cls] = sil_score[y == tgt_cls].mean()
        results_per_class['purity'][0][tgt_cls] = purity_score[y == tgt_cls].mean()

    return results_pairwise, results_per_class

def plot_pairwise_overlap_metrics(coordinates, labels, model_names, tag, k=15):
    results_pairwise, results_per_class = pairwise_overlap_metrics(coordinates, labels, k=k)

    plot_heatmap(
        results_pairwise['silhouette'],
        model_names,
        "Pairwise Silhouette Coefficient of " + tag,
        "Silhouette Coefficient",
        x_label="Distractor Backbone",
        y_label="Focus Backbone",
        out_file=out_dir + "Pairwise Silhouette Coefficient of " + tag + ".png",
        fmt=".3"
    )

    plot_heatmap(
        results_pairwise['purity'],
        model_names,
        "Pairwise kNN Purity of " + tag,
        "kNN Purity",
        x_label="Distractor Backbone",
        y_label="Focus Backbone",
        out_file=out_dir + "Pairwise kNN Purity of " + tag + ".png",
        fmt=".3"
    )

    plot_heatmap(
        results_per_class['silhouette'],
        model_names,
        "Per-model Silhouette Coefficient of " + tag,
        "Silhouette Coefficient",
        x_label="Model",
        y_label="",
        out_file=out_dir + "Per-model Silhouette Coefficient of " + tag + ".png",
        fmt=".3"
    )

    plot_heatmap(
        results_per_class['purity'],
        model_names,
        "Per-model kNN Purity of " + tag,
        "kNN Purity",
        x_label="Model",
        y_label="",
        out_file=out_dir + "Per-model kNN Purity of " + tag + ".png",
        fmt=".3"
    )

def plot_adapted_embeds(adapter, embeds_val, out_model, ds, pca_dim = None):
    print(f"running embeds overlap plot: adapting to {out_model}")
    bs_val = 1000

    #embeds_val = [embed.to(device).float() for embed in embeds_val]
    all_latents = []
    for embeds in zip(*[torch.split(x, bs_val, 0) for x in embeds_val]):
        with torch.inference_mode():
            embeds = [embed.to(device, non_blocking=True).float() for embed in embeds]
            latents = adapter.fw_all_embeds_to_latent(embeds)
            latents = adapter.fw_latent_to_one_embed(latents, out_model.replace('.', '_'))
            all_latents.append(latents)
    all_latents = torch.cat(all_latents, dim=1).to('cpu', non_blocking=True)
    K, B, D = all_latents.shape
    src_models = torch.arange(K).unsqueeze(-1).repeat(1, B).reshape(-1).numpy()

    metric_input = all_latents.reshape(-1, D)
    if pca_dim is not None:
        print(f'running PCA (dim={pca_dim})')
        metric_input = PCA(n_components = pca_dim,).fit_transform(metric_input)
        print('Done')

    plot_pairwise_overlap_metrics(metric_input, src_models, model_names, f"Embeds Adapted to {out_model} on {ds}" + (f" (PCA dim {pca_dim})" if pca_dim else ""))
    
def plot_latents_with_pca(adapter, embeds_val, ds, pca_dim):
    print(f"running latents overlap plot with pca dim {pca_dim}")
    bs_val = 1000

    #embeds_val = [embed.to(device).float() for embed in embeds_val]
    all_latents = []
    for embeds in zip(*[torch.split(x, bs_val, 0) for x in embeds_val]):
        with torch.inference_mode():
            embeds = [embed.to(device, non_blocking=True).float() for embed in embeds]
            latents = adapter.fw_all_embeds_to_latent(embeds)
            all_latents.append(latents)
    all_latents = torch.cat(all_latents, dim=1).to('cpu', non_blocking=True)
    K, B, D = all_latents.shape
    src_models = torch.arange(K).unsqueeze(-1).repeat(1, B).reshape(-1).numpy()

    print(f'running PCA (dim={pca_dim})')
    metric_input = PCA(n_components = pca_dim,).fit_transform(all_latents.reshape(-1, D))
    print('Done')

    plot_pairwise_overlap_metrics(metric_input, src_models, model_names, f"Latents on {ds}" + (f" (PCA dim {pca_dim})" if pca_dim else ""))

if __name__ == '__main__':
    # TODO flexible paths
    print("loading train embeds...")

    embeds_val = [
        torch.load(
            #f'embeds/embeds_in1k_val_{model}.pt',
            f'embeds/embeds_mscoco_captions2017_test_{model}.pt',
            map_location='cpu'
        ) for model in tqdm.tqdm(model_names)
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

    #plot_latents(adapter, embeds_val)

    #ds = "ImageNet-1k"
    ds = 'COCO Captions 2017'
    
    for model in model_names:
        plot_adapted_embeds(adapter, embeds_val, model, ds, pca_dim = 256)
    
    plot_latents_with_pca(adapter, embeds_val, ds, 256)

