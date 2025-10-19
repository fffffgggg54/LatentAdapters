import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRA(nn.Module):
    def __init__(self, in_dim, out_dim, rank=16):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.A = nn.Parameter(torch.randn(in_dim, rank) * (1/in_dim))
        self.B = nn.Parameter(torch.zeros(rank, out_dim))

    def extra_repr(self):
        return f'in_features={self.in_dim}, out_features={self.out_dim}, rank={self.rank}'

    def forward(self, x):
        return x @ self.A @ self.B

def zero(x):
    return 0


class MLP(nn.Module):
    def __init__(self, dim, expansion_factor=4, act_fn=nn.GELU):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * expansion_factor)
        self.fc2 = nn.Linear(dim * expansion_factor, dim)
        self.expansion_factor = expansion_factor
        #self.res_scale = nn.Parameter(torch.ones(dim))
        self.act_fn = act_fn()
    
    def forward(self, x, fc1_lora = zero, fc2_lora = zero):
        h = self.act_fn(self.fc1(x) + fc1_lora(x))
        return self.fc2(h) + fc2_lora(h) #+ x * self.res_scale


def filter_state_dict(state_dict, search_key):
    filtered_dict = {}

    # Iterate over the items in the state dictionary
    for key, value in state_dict.items():
        # Check if the key starts with the search key followed by a dot
        if key.startswith(search_key + '.'):
            # Extract the suffix after the dot
            suffix = key.split('.')[-1]
            filtered_dict[suffix] = value
    return filtered_dict

# TODO dict inputs instead of list

class Adapter(nn.Module):
    def __init__(
        self, 
        model_names: list, 
        model_dims: list, 
        hidden_dim = 2048, 
        middle_model = MLP
    ):
        super().__init__()
        self.model_names = model_names
        assert len(model_names) == len(model_dims), f"model_names should have the same length as model_dims, got {len(model_names)} and {len(model_dims)}"
        self.model_dims = model_dims
        self.name_to_dim = {k: v for k, v in zip(model_names, model_dims)}

        self.hidden_dim = hidden_dim
        self.middle_model = middle_model(self.hidden_dim)
        
        self.expansion_factor = self.middle_model.expansion_factor
        self.encoders = nn.ModuleDict({name: nn.Linear(dim, self.hidden_dim) for name, dim in zip(model_names, model_dims)})
        self.fc1_loras = nn.ModuleDict({name: LoRA(self.hidden_dim, self.hidden_dim*self.expansion_factor) for name in model_names})
        self.fc2_loras = nn.ModuleDict({name: LoRA(self.hidden_dim*self.expansion_factor, self.hidden_dim) for name in model_names})
        self.decoders = nn.ModuleDict({name: nn.Linear(self.hidden_dim, dim) for name, dim in zip(model_names, model_dims)})

        #discriminator = nn.Sequential(nn.Linear(hidden_dim, 1024), nn.GELU(), nn.Linear(1024, len(models)))

    def expand(self, model_names: list, model_dims: list):
        # FIXME no safety checks, can be descructive
        self.encoders.update({name: nn.Linear(dim, self.hidden_dim) for name, dim in zip(model_names, model_dims)})
        self.fc1_loras.update({name: LoRA(self.hidden_dim, self.hidden_dim*self.expansion_factor) for name in model_names})
        self.fc2_loras.update({name: LoRA(self.hidden_dim*self.expansion_factor, self.hidden_dim) for name in model_names})
        self.decoders.update({name: nn.Linear(self.hidden_dim, dim) for name, dim in zip(model_names, model_dims)})

        self.name_to_dim.update({k: v for k, v in zip(model_names, model_dims)})
        self.model_dims.extend(model_dims)
        self.model_names.extend(model_names)
    
    def get_params_for_one_model(self, model_name):
        return [
            *self.encoders[model_name].parameters(),
            *self.fc1_loras[model_name].parameters(),
            *self.fc2_loras[model_name].parameters(),
            *self.decoders[model_name].parameters()
        ]
    
    def get_state_dict_for_one_model(self, model_name):
        return nn.ModuleDict({
            'encoder': self.encoders[model_name],
            'fc1_lora': self.fc1_loras[model_name],
            'fc2_lora': self.fc2_loras[model_name],
            'decoder': self.decoders[model_name]
        }).state_dict()

    # FIXME There is probably a better way to save checkpoints, probably thorugh selecting keys of the within the moduledict to store and doing partial loads.
    def load_state_dict_for_one_model(self, model_name, state_dict):
        self.encoders[model_name].load_state_dict(filter_state_dict(state_dict, 'encoder')) 
        self.fc1_loras[model_name].load_state_dict(filter_state_dict(state_dict, 'fc1_lora')) 
        self.fc2_loras[model_name].load_state_dict(filter_state_dict(state_dict, 'fc2_lora')) 
        self.decoders[model_name].load_state_dict(filter_state_dict(state_dict, 'decoder')) 

    def get_params_for_model_list(self, model_names):
        params = []
        for model in model_names:
            params.extend(self.get_params_for_one_model(model))
        return params

    def fw_one_embed_to_latent(self, embed, model_name):
        # input shape of [B, d_in]
        latent = self.middle_model(
            self.encoders[model_name](embed),
            fc1_lora=self.fc1_loras[model_name],
            fc2_lora=self.fc2_loras[model_name]
        )

        # output shape of [B, d_hidden]
        return latent
    
    def fw_all_embeds_to_latent(self, embeds):
        # input shape of N * [B, d_in]
        # in_model, batch_idx, dim

        latents = []
        for i, model_name in enumerate(self.model_names):
            latent = self.fw_one_embed_to_latent(embeds[i], model_name)
            latents.append(latent)

        # output shape of [N, B, d_hidden]
        # in_model, batch_idx, dim
        return torch.stack(latents, dim=0)

    def fw_latent_to_one_embed(self, latent, model_name):
        # input shape of [B, d_hidden] or [N, B, d_hidden]
        # batch_idx, dim or in_model, batch_idx, dim
        embed = self.decoders[model_name](latent)

        # output shape of [B, d_out] or [N, B, d_out]
        # batch_idx, dim or in_model, batch_idx, dim
        return embed

    def fw_latent_to_all_embeds(self, latent):
        # input shape of [B, d_hidden] or [N, B, d_hidden]
        # batch_idx, dim or in_model, batch_idx, dim
        embeds = []
        for model_name in self.model_names:
            embed = self.fw_latent_to_one_embed(latent, model_name)
            embeds.append(embed)

        # output shape of N * [B, d_out] or N * [N, B, d_out]
        # out_model, batch_idx, dim or out_model, in_model, batch_idx, dim
        return embeds

    def adapt_one_to_one(self, in_embed, input_model_name, output_model_name):
        # input shape of [B, d_in]
        latent = self.fw_one_embed_to_latent(in_embed, input_model_name)
        out_embed = self.fw_latent_to_one_embed(latent, output_model_name)
        # output shape of [B, d_out]
        return out_embed

    def adapt_one_to_all(self, in_embed, input_model_name):
        # input shape of [B, d_in]
        latent = self.fw_one_embed_to_latent(in_embed, input_model_name)
        out_embeds = self.fw_latent_to_all_embeds(latent)
        # output shape of N * [B, d_out]
        # out_model, batch_idx, dim
        return out_embeds

    def adapt_all_to_one(self, embeds, output_model_name):
        # input shape of N * [B, d_in]
        # in_model, batch_idx, dim
        latent = self.fw_one_embed_to_latent(in_embed, input_model_name)
        out_embed = self.fw_latent_to_one_embed(latent, output_model_name)
        # output shape of [N, B, d_out]
        # in_model, batch_idx, dim
        return out_embed

    def adapt_all_to_all(self, embeds):
        # input shape of N * [B, d_in]
        # in_model, batch_idx, dim
        latents = self.fw_all_embeds_to_latent(embeds)
        all_embeds = self.fw_latent_to_all_embeds(latents)
        # output shape of N * [N, B, d_out]
        # out_model, in_model, batch_idx, dim
        return all_embeds

    def fw_cycle_latents(self, adapted_embeds):
        # N * [N, B, d_cycle]
        # cycle_model, in_model, batch_idx, dim

        cycle_latents = []
        # index over cycle models
        for i, model_name in enumerate(self.model_names):
            #print(model_name)
            #print(adapted_embeds[i].shape)
            # [N, B, d_cycle] -> [N, B, d_hidden]
            # in_model, batch_idx, dim
            cycle_latent = self.fw_one_embed_to_latent(adapted_embeds[i], model_name)
            cycle_latents.append(cycle_latent)
        # [N, N, B, d_hidden]
        # in_model, cycle_model, batch_idx, dim
        cycle_latents = torch.stack(cycle_latents, dim=1)
        return cycle_latents

    def fw_cycle_latents_to_self_cycle_embeds(self, cycle_latents):
        # [N, N, B, d_hidden]
        # in_model, cycle_model, batch_idx, dim
        self_cycle_embeds = []
        # index over input models
        for i, model_name in enumerate(self.model_names):
            # [N, B, d_hidden] -> [N, B, d_in]
            # cycle_model, batch_idx, dim
            self_cycle_embed = self.fw_latent_to_one_embed(cycle_latents[i], model_name)
            self_cycle_embeds.append(self_cycle_embed)
        # N * [N, B, d_in]
        # in_model, cycle_model, batch_idx, dim        
        return self_cycle_embeds


    def fw_self_cycle_embeds(self, adapted_embeds):
        # N * [N, B, d_cycle]
        # cycle_model, in_model, batch_idx, dim
        cycle_latents = self.fw_cycle_latents(adapted_embeds)
        self_cycle_embeds = self.fw_cycle_latents_to_self_cycle_embeds(cycle_latents)
        # N * [N, B, d_in]
        # in_model, cycle_model, batch_idx, dim        
        return self_cycle_embeds