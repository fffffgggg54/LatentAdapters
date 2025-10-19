import os
os.environ['HF_HOME'] = '/scratch/fyguan/hf_home'
os.environ['TORCH_HOME'] = '/scratch/fyguan/torch_home'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import timm
import timm.data
import timm.optim

import open_clip
import transformers
from transformers import AutoTokenizer, AutoModel

import tqdm

import datasets

from PIL import Image

import gc

# Check for available GPUs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# wrap a huggingface dataset to only return the image column and nothing else
class ImageOnlyDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None, column = "image"):
        self.hf_dataset = hf_dataset
        self.transform = transform
        self.column = column

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        image = self.hf_dataset[idx][self.column]
        if self.transform:
            image = self.transform(image)
        return image, 0

ds = datasets.load_dataset('laion/conceptual-captions-12m-webdataset')['train']
ds_train = ImageOnlyDataset(ds, column = 'txt')

def last_token_pool(last_hidden_states, attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'

# Each query must come with a one-sentence instruction that describes the task
task = 'Given an image caption, retrieve relevant images that match the caption'


def compute_embeddings_qwen():
    # Base batch size per GPU
    base_bs = 64
    model_name = "Qwen/Qwen3-Embedding-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    label = "cc12m_text"
    model_tag = "qwen3_embedding_0.6b"
    
    num_workers = 30
    loader = torch.utils.data.DataLoader(ds_train, batch_size=total_bs, num_workers=num_workers)
    
    all_batches = []
    
    for textBatch, _ in tqdm.tqdm(loader, desc=f"Computing embeddings for {label}"):
        with torch.autograd.grad_mode.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=autocast_dtype):
            batch_dict = tokenizer(
                textBatch,
                padding=True,
                return_tensors="pt",
            )
            batch_dict.to(model.device)
            outputs = model(**batch_dict)
            embedsBatch = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    embeds = torch.cat(embedsBatch, dim=0)

    save_path = f'/scratch/fyguan/embeds_{label}_{model_tag}.pt'
    print(f"Saving embeddings to {save_path}")
    torch.save(embeds, save_path)

    del all_batches
    del embeds
    gc.collect()

def compute_embeddings_openclip():
    # Base batch size per GPU
    base_bs = 64
    model_id = 'hf-hub:timm/PE-Core-bigG-14-448'
    model, _, preprocess = open_clip.create_model_and_transforms(model_id)
    tokenizer = open_clip.get_tokenizer(model_id)

    label = "cc12m_text"
    model_tag = "pe_core_text"
    
    num_workers = 30
    loader = torch.utils.data.DataLoader(ds_train, batch_size=total_bs, num_workers=num_workers)
    
    all_batches = []
    
    for textBatch, _ in tqdm.tqdm(loader, desc=f"Computing embeddings for {label}"):
        with torch.autograd.grad_mode.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=autocast_dtype):
            batch_dict = tokenizer(textBatch, context_length=model.context_length)
            batch_dict.to(model.device)
            embedsBatch = model.encode_text(text)
    embeds = torch.cat(embedsBatch, dim=0)

    save_path = f'/scratch/fyguan/embeds_{label}_{model_tag}.pt'
    print(f"Saving embeddings to {save_path}")
    torch.save(embeds, save_path)

    del all_batches
    del embeds
    gc.collect()



compute_embeddings_openclip()  
compute_embeddings_qwen()
    