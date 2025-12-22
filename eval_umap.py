# Umap clustering of adapted embeds/latents

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

import umap
import umap.plot

import seaborn as sns
import numpy as np
import colormaps as cmaps
import pandas as pd

from sklearn.decomposition import PCA

out_dir = "outputs/scratch_mmID_2048_100epoch_discriminator0.0_latent10.0_MSE_JointTraining/"
adapter_hidden_dim = 2048
epoch = 99

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

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
    'text_pe_core_text',
    'text_qwen3_embedding_4b_bf16',

    'convformer_b36.sail_in22k_ft_in1k',
    'vit_base_patch16_224.augreg_in21k_ft_in1k',
    'vit_base_patch16_clip_224.openai_ft_in1k',
    'convnext_base.fb_in1k',
    'beit3_large_patch16_224.in22k_ft_in1k',
]
    
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
def scatter_plot_with_legend(coordinates, labels, class_names, tag):
    """
    Generates a scatter plot with a legend from numpy arrays and a class name dictionary using Seaborn.

    Args:
        coordinates (np.ndarray): A [N, 2] numpy array of (x, y) coordinates.
        labels (np.ndarray): A [N] numpy array of class labels.
        class_names (dict): A dictionary mapping class labels to class names.
        out_dir (str): The output directory to save the plot.
    """
    # Create a pandas DataFrame from the numpy arrays
    df = pd.DataFrame({
        'X Coordinate': coordinates[:, 0],
        'Y Coordinate': coordinates[:, 1],
        'Class': [class_names.get(label, f"Class {label}") for label in labels]
    })

    # Set the figure size and DPI
    plt.figure(figsize=(30, 30), dpi=500)

    # Create the scatter plot using seaborn
    sns.scatterplot(
        x='X Coordinate',
        y='Y Coordinate',
        hue='Class',
        data=df.sample(frac=1).reset_index(drop=True),
        #cmap=cmaps.cet_g_bw,  # Using a grayscale-compatible colormap
        size=0.5,
        legend='full',
        marker="+"
    )

    plt.title(f'Scatter Plot of Data Points by Class ({tag})')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    data_csv = df.to_csv(index=False)

    # Save the figure
    #plt.savefig(out_dir + f'in1k_val_full_densmap_clustering_{tag}.png', metadata = {'Plot data': data_csv})
    plt.savefig(out_dir + f'mscoco_captions2017_test_densmap_clustering_{tag}.png', metadata = {'Plot data': data_csv})
    #plt.show()


def plot_adapted_embeds(adapter, embeds_val, out_model, pca_dim = None):
    print(f"running embeds umap plot: adapting to {out_model}")
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

    umap_input = all_latents.reshape(-1, D)
    if pca_dim is not None:
        umap_input = PCA(n_components = pca_dim,).fit_transform(umap_input)

    mapper = umap.UMAP(n_neighbors=15, densmap=True, verbose=True, force_approximation_algorithm=False).fit(umap_input)
    #plt.figure(figsize=(8,8))
    #ax = umap.plot.points(mapper, labels=src_models, width=20000, height=20000)
    #plt.show()
    #ax.figure.savefig(out_dir + "in1k_val_10k_densmap_clustering.png")
    coordinates = mapper.transform(umap_input)
    scatter_plot_with_legend(coordinates, src_models, {i:x for i, x in enumerate(model_names)}, out_model + (f"_pca_dim_{pca_dim}" if pca_dim else ""))
    
def plot_latents_with_pca(adapter, embeds_val, pca_dim):
    print(f"running latents umap plot with pca dim {pca_dim}")
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

    pca_repr = PCA(n_components = pca_dim,).fit_transform(all_latents.reshape(-1, D))

    mapper = umap.UMAP(n_neighbors=15, densmap=True, verbose=True, force_approximation_algorithm=False).fit(pca_repr)

    coordinates = mapper.transform(pca_repr)
    scatter_plot_with_legend(coordinates, src_models, {i:x for i, x in enumerate(model_names)}, f"latents_pca_dim_{pca_dim}")




if __name__ == '__main__':
    # TODO flexible paths
    print("loading train embeds...")

    embeds_val = [
        torch.load(
            #f'embeds/embeds_in1k_val_{model}.pt',
            f'embeds/embeds_mscoco_captions2017_test_{model}.pt',
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

    #plot_latents(adapter, embeds_val)
    
    for model in model_names:
        plot_adapted_embeds(adapter, embeds_val, model, pca_dim = 256)
    
    plot_latents_with_pca(adapter, embeds_val, 256)

