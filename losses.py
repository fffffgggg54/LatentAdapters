import torch
import torch.nn.functional as F

def MSE(pred, targ, weight = 1):
    return (weight * (pred - targ) ** 2).mean()

# normalize each dimension to compensate for differing dimension scales
def norm_MSE_signal_power(pred, targ, weight = 1):
    return ((weight * (pred - targ) ** 2).mean(dim=(0,1)) / (targ**2).mean(dim=(0,1))).mean()

def norm_MSE_variance(pred, targ, weight = 1):
    return ((weight * (pred - targ) ** 2).mean(dim=(0,1)) / targ.var(dim=(0,1))).mean()

# TODO embeds should be a dict to adapt a subset of adapters by key
def pairwise_adapter_loss_with_discriminator(adapter, discriminator, embeds, self_mse=1, pw_mse=1, cycle_mse=1, dc_ce=1, loss_fn = MSE):
    # input shape of N * [B, d_in]
    # in_model, batch_idx, dim

    # [N, B, d_hidden]
    # in_model, batch_idx, dim
    latents = adapter.fw_all_embeds_to_latent(embeds)
    #print(latents.shape)

    # N * [N, B, d_out]
    # out_model, in_model, batch_idx, dim
    adapted_embeds = adapter.fw_latent_to_all_embeds(latents)

    # N * [N, B, d_in]
    # in_model, cycle_model, batch_idx, dim   
    self_cycle_embeds = adapter.fw_self_cycle_embeds(adapted_embeds)

    # for each embed corresponding to model i in embeds, calculate the mse between embed and all embeds adapted into model i's space
    # stride by output adapter (same dims for all input adapters), NOT input adapter (different dims from a single input adapter)
    loss_reconstruction = torch.stack([
        loss_fn(
            adapted_embed, # [N, B, d_out]
            embed.unsqueeze(0), # [1, B, d_out]
            weight=(pw_mse * torch.ones(len(embeds), device=embed.device)).scatter_(0, torch.tensor(i, device=embed.device), self_mse).reshape(len(embeds), 1, 1) # [N, 1, 1]
        ) for i, (adapted_embed, embed) in enumerate(zip(adapted_embeds, embeds))
    ]).mean()


    # 0 on diagonal, cycle consistency everywhere else
    # TODO try not masking diagonal
    loss_cycle_consistency = torch.stack([
        loss_fn(
            self_cycle_embed, # [N, B, d_out]
            embed.unsqueeze(0), # [1, B, d_out]
            weight=(cycle_mse * torch.ones(len(embeds), device=embed.device)).scatter_(0, torch.tensor(i, device=embed.device), 0).reshape(len(embeds), 1, 1) # [N, 1, 1]
        ) for i, (self_cycle_embed, embed) in enumerate(zip(self_cycle_embeds, embeds))
    ]).mean()

    # No distance metric loss from the universal geometry paper, use mse since we have paired embeddings are paired
    # TODO vectorize hidden loss, test hidden loss vs discriminator loss vs both
    #loss_pairwise_hidden_mse = F.mse_loss(repA, repB)

    # discriminator infer
    loss_dc_pred = 0
    for latent in latents:
        # [B, N]
        discriminator_outputs_infer = discriminator(latent)
        loss_dc_pred = loss_dc_pred + F.cross_entropy(
                discriminator_outputs_infer, 
                (1/len(latents) * torch.ones_like(discriminator_outputs_infer)) # target: all classes have equal probability
            ) / len(latents)

    loss_dc_train = 0
    for i, latent in enumerate(latents.detach()):
        # [B, N]
        discriminator_outputs_train = discriminator(latent)
        loss_dc_train = loss_dc_train + F.cross_entropy(
                discriminator_outputs_train, 
                (i * torch.ones(len(latent), device=latents.device)).long() # target class is model idx
            ) / len(latents)

    # TODO optimize this to piggyback off of the either train or pred loss
    with torch.no_grad():
        # [N, B, 1]
        discriminator_targets = torch.arange(len(latents), device=latents.device).reshape(len(latents), 1).expand(len(latents), latents.shape[1])
        discriminator_accuracy = (discriminator(latents).argmax(dim=-1) == discriminator_targets).float().mean(dim=1)

    loss = loss_reconstruction + loss_cycle_consistency + loss_dc_pred * dc_ce#+ loss_pairwise_hidden_mse
    return loss, loss_dc_train, loss_dc_pred.detach(), discriminator_accuracy