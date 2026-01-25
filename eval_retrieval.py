import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv
import io

from adapter import Adapter

out_dir = "outputs/final_mmGELU_4096_10epoch_discriminator1.0_latent1.0_MSE_JointTraining/"
adapter_hidden_dim = 4096
epoch = 9

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
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

    'vit_large_patch16_dinov3.lvd1689m',
    'vit_huge_patch14_gap_224.in22k_ijepa',

    'eva02_base_patch14_448.mim_in22k_ft_in22k_in1k',
    'vit_so400m_patch14_siglip_gap_224.pali2_10b_pt',
    'aimv2_large_patch14_224.apple_pt',

    'convformer_b36.sail_in22k_ft_in1k',
    'vit_base_patch16_224.augreg_in21k_ft_in1k',
    'vit_base_patch16_clip_224.openai_ft_in1k',
    'convnext_base.fb_in1k',
    'beit3_large_patch16_224.in22k_ft_in1k',

    'vit_pe_core_gigantic_patch14_448.fb',
    'text_pe_core_text',

    'text_qwen3_embedding_4b_bf16',
    'text_e5_large_v2',
    'text_bert_large_uncased',
    'text_roberta_large',
    'text_gte_en_mlm_large',
    'text_gte_large_en_v1.5',
    'text_gte_modernbert_base',
    'text_gte_multilingual_base',

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

def compute_retrieval_metrics(query_embeds, key_embeds, k_values=[1, 5, 10], do_norm = True):
    if do_norm:
        query_embeds = F.normalize(query_embeds, p=2, dim=1)
        key_embeds = F.normalize(key_embeds, p=2, dim=1)

    similarity = query_embeds @ key_embeds.T
    
    # We assume the i-th query matches the i-th key
    n = similarity.shape[0]
    gt_indices = torch.arange(n, device=similarity.device)
        
    gt_scores = similarity[torch.arange(n), torch.arange(n)]

    higher_score_mask = similarity > gt_scores.unsqueeze(1) 
    
    gt_ranks = higher_score_mask.sum(dim=1)
    
    gt_ranks_1based = gt_ranks.float() + 1
    
    metrics = {}
    for k in k_values:
        recall = (gt_ranks < k).float().mean().item()
        metrics[f'R@{k}'] = recall
        
    metrics['mean_rank'] = gt_ranks_1based.mean().item()
    metrics['median_rank'] = gt_ranks_1based.median().item()
    
    return metrics

def print_stats(matrix, metric_name, labels):
    matrix = matrix * 100
    n = len(matrix)
    
    diagonal = np.diag(matrix)
    
    mask = ~np.eye(n, dtype=bool)
    off_diag = matrix[mask]
    
    print(f"{metric_name} Stats")
    print(f"Same-model (diagonal):  Mean = {diagonal.mean():.2f}%")
    print(f"Cross-model (off-diag): Mean = {off_diag.mean():.2f}%  Std = {off_diag.std():.2f}%")
    print(f"Overall average:        Mean = {matrix.mean():.2f}%")
    
    triu_indices = np.triu_indices(n, k=1)
    triu_values = matrix[triu_indices]
    
    best_idx = np.argmax(triu_values)
    worst_idx = np.argmin(triu_values)
    
    best_i, best_j = triu_indices[0][best_idx], triu_indices[1][best_idx]
    worst_i, worst_j = triu_indices[0][worst_idx], triu_indices[1][worst_idx]
    
    print(f"\nBest pair:  {labels[best_i]:8s} ↔ {labels[best_j]:8s}  {matrix[best_i, best_j]:.2f}%")
    print(f"Worst pair: {labels[worst_i]:8s} ↔ {labels[worst_j]:8s}  {matrix[worst_i, worst_j]:.2f}%")

def main():
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
    
    for model in model_names:
        adapter.load_state_dict_for_one_model(
            model.replace('.', '_'), 
            torch.load(out_dir + f"adapter_{model}_epoch_{epoch}.pt", weights_only=True, map_location='cpu')
        )
    adapter.middle_model.load_state_dict(torch.load(out_dir + f"adapter_middle_model_epoch_{epoch}.pt", weights_only=True, map_location='cpu'))
    adapter = adapter.to(device).float()
    adapter = adapter.eval()
    
    print(adapter)
    print(model_names)
    print([x[0].shape for x in embeds_val])
    
    bs_val = 1000
    
    print("Running retrieval on latents")
    all_latents = []
    for embeds in zip(*[torch.split(x, bs_val, 0) for x in embeds_val]):
        with torch.inference_mode():
            embeds = [embed.to(device, non_blocking=True).float() for embed in embeds]
            latents = adapter.fw_all_embeds_to_latent(embeds)
            all_latents.append(latents)
    all_latents = torch.cat(all_latents, dim=1)
    all_latents_normalized = F.normalize(all_latents, p=2, dim=-1)
    #all_latents_normalized = all_latents_normalized.to('cpu', non_blocking=True)
    
    n_models = len(model_names)
    r1_matrix = np.zeros((n_models, n_models))
    r5_matrix = np.zeros((n_models, n_models))
    r10_matrix = np.zeros((n_models, n_models))
    
    with tqdm(total=n_models**2, desc="Evaluating") as pbar:
        for i, embeds in enumerate(all_latents_normalized):
            for j, queries in enumerate(all_latents_normalized):
                metrics = compute_retrieval_metrics(queries, embeds, k_values=[1, 5, 10], do_norm=False)
                
                r1_matrix[i, j] = metrics['R@1']
                r5_matrix[i, j] = metrics['R@5']
                r10_matrix[i, j] = metrics['R@10']
                
                pbar.update(1)
    
    
    print_stats(r1_matrix, "R@1", model_names)
    print_stats(r5_matrix, "R@5", model_names)
    print_stats(r10_matrix, "R@10", model_names)
    
    plot_heatmap(
        r1_matrix,
        model_names,
        "Pairwise Retrieval of Latents (R@1, mscoco caption2017)",
        "Recall@1",
        x_label="Embed model",
        y_label="Query model",
        out_file=out_dir + "Pairwise Retrieval of Latents (R@1, mscoco caption2017).png",
        fmt=".2%"
    )
    
    plot_heatmap(
        r5_matrix,
        model_names,
        "Pairwise Retrieval of Latents (R@5, mscoco caption2017)",
        "Recall@5",
        x_label="Embed model",
        y_label="Query model",
        out_file=out_dir + "Pairwise Retrieval of Latents (R@5, mscoco caption2017).png",
        fmt=".2%"
    )
    
    plot_heatmap(
        r10_matrix,
        model_names,
        "Pairwise Retrieval of Latents (R@10, mscoco caption2017)",
        "Recall@10",
        x_label="Embed model",
        y_label="Query model",
        out_file=out_dir + "Pairwise Retrieval of Latents (R@10, mscoco caption2017).png",
        fmt=".2%"
    )
    
    print(f"\nKey findings:")
    mask = ~np.eye(n_models, dtype=bool)
    print(f"  Average cross-model R@1:  {r1_matrix[mask].mean() * 100:.2f}%")
    print(f"  Average cross-model R@5:  {r5_matrix[mask].mean() * 100:.2f}%")
    print(f"  Average cross-model R@10: {r10_matrix[mask].mean() * 100:.2f}%")
    
    print("Running retrieval on embeds")
    with torch.inference_mode():
        # [N, N, B, D]
        # out_model, in_model, batch_idx, dim
        all_adapted_embeds = adapter.fw_latent_to_all_embeds(all_latents)
        all_adapted_embeds_normalized = [F.normalize(x, p=2, dim=-1) for x in all_adapted_embeds]
        #all_adapted_embeds_normalized = all_adapted_embeds_normalized.to('cpu', non_blocking=True)
        
        embeds_val_normalized = [F.normalize(x.to(device, non_blocking=True).float(), p=2, dim=-1) for x in embeds_val]
    
    
    
    print("Adapt queries")
    
    n_models = len(model_names)
    r1_matrix = np.zeros((n_models, n_models))
    r5_matrix = np.zeros((n_models, n_models))
    r10_matrix = np.zeros((n_models, n_models))
    
    with tqdm(total=n_models**2, desc="Evaluating") as pbar:
        for i, (embeds, current_model_queries) in enumerate(zip(embeds_val_normalized, all_adapted_embeds_normalized)):
            for j, queries in enumerate(current_model_queries):
                metrics = compute_retrieval_metrics(queries, embeds, k_values=[1, 5, 10], do_norm=False)
                
                r1_matrix[i, j] = metrics['R@1']
                r5_matrix[i, j] = metrics['R@5']
                r10_matrix[i, j] = metrics['R@10']
                
                pbar.update(1)
    
    
    print_stats(r1_matrix, "R@1", model_names)
    print_stats(r5_matrix, "R@5", model_names)
    print_stats(r10_matrix, "R@10", model_names)
    
    plot_heatmap(
        r1_matrix,
        model_names,
        "Pairwise Retrieval with Adapted Queries (R@1, mscoco caption2017)",
        "Recall@1",
        x_label="Embed model",
        y_label="Query model",
        out_file=out_dir + "Pairwise Retrieval with Adapted Queries (R@1, mscoco caption2017).png",
        fmt=".2%"
    )
    
    plot_heatmap(
        r5_matrix,
        model_names,
        "Pairwise Retrieval with Adapted Queries (R@5, mscoco caption2017)",
        "Recall@5",
        x_label="Embed model",
        y_label="Query model",
        out_file=out_dir + "Pairwise Retrieval with Adapted Queries (R@5, mscoco caption2017).png",
        fmt=".2%"
    )
    
    plot_heatmap(
        r10_matrix,
        model_names,
        "Pairwise Retrieval with Adapted Queries (R@10, mscoco caption2017)",
        "Recall@10",
        x_label="Embed model",
        y_label="Query model",
        out_file=out_dir + "Pairwise Retrieval with Adapted Queries (R@10, mscoco caption2017).png",
        fmt=".2%"
    )
    
    print(f"\nKey findings:")
    mask = ~np.eye(n_models, dtype=bool)
    print(f"  Average cross-model R@1:  {r1_matrix[mask].mean() * 100:.2f}%")
    print(f"  Average cross-model R@5:  {r5_matrix[mask].mean() * 100:.2f}%")
    print(f"  Average cross-model R@10: {r10_matrix[mask].mean() * 100:.2f}%")
    
    print("Adapt embeds")
    
    n_models = len(model_names)
    r1_matrix = np.zeros((n_models, n_models))
    r5_matrix = np.zeros((n_models, n_models))
    r10_matrix = np.zeros((n_models, n_models))
    
    with tqdm(total=n_models**2, desc="Evaluating") as pbar:
        for j, (queries, current_model_embeds) in enumerate(zip(embeds_val_normalized, all_adapted_embeds_normalized)):
            for i, embeds in enumerate(current_model_embeds):
                metrics = compute_retrieval_metrics(queries, embeds, k_values=[1, 5, 10], do_norm=False)
                
                r1_matrix[i, j] = metrics['R@1']
                r5_matrix[i, j] = metrics['R@5']
                r10_matrix[i, j] = metrics['R@10']
                
                pbar.update(1)
    
    
    print_stats(r1_matrix, "R@1", model_names)
    print_stats(r5_matrix, "R@5", model_names)
    print_stats(r10_matrix, "R@10", model_names)
    
    plot_heatmap(
        r1_matrix,
        model_names,
        "Pairwise Retrieval with Adapted Embeds (R@1, mscoco caption2017)",
        "Recall@1",
        x_label="Embed model",
        y_label="Query model",
        out_file=out_dir + "Pairwise Retrieval with Adapted Embeds (R@1, mscoco caption2017).png",
        fmt=".2%"
    )
    
    plot_heatmap(
        r5_matrix,
        model_names,
        "Pairwise Retrieval with Adapted Embeds (R@5, mscoco caption2017)",
        "Recall@5",
        x_label="Embed model",
        y_label="Query model",
        out_file=out_dir + "Pairwise Retrieval with Adapted Embeds (R@5, mscoco caption2017).png",
        fmt=".2%"
    )
    
    plot_heatmap(
        r10_matrix,
        model_names,
        "Pairwise Retrieval with Adapted Embeds (R@10, mscoco caption2017)",
        "Recall@10",
        x_label="Embed model",
        y_label="Query model",
        out_file=out_dir + "Pairwise Retrieval with Adapted Embeds (R@10, mscoco caption2017).png",
        fmt=".2%"
    )
    
    print(f"\nKey findings:")
    mask = ~np.eye(n_models, dtype=bool)
    print(f"  Average cross-model R@1:  {r1_matrix[mask].mean() * 100:.2f}%")
    print(f"  Average cross-model R@5:  {r5_matrix[mask].mean() * 100:.2f}%")
    print(f"  Average cross-model R@10: {r10_matrix[mask].mean() * 100:.2f}%")
    
    

if __name__ == "__main__":
    main()
