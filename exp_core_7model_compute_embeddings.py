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

import tqdm

import datasets

from PIL import Image

import gc

# Check for available GPUs
num_gpus = torch.cuda.device_count()
devices = [torch.device(f'cuda:{i}') for i in range(num_gpus)] if num_gpus > 0 else [torch.device('cpu')]
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

interpolation_modes = {'bicubic': transforms.InterpolationMode.BICUBIC, 'bilinear': transforms.InterpolationMode.BILINEAR}
in1k_train = timm.data.create_dataset("", root="/scratch/fyguan/temp", split='ILSVRC2012_img_train.tar')
in1k_val = timm.data.create_dataset("torch/imagenet", root="/scratch/fyguan/temp", split='val')
ds = datasets.load_dataset('laion/conceptual-captions-12m-webdataset')['train']
ds_train = ImageOnlyDataset(ds, column = 'jpg')

def fw_enc(model, x):
    x = model.forward_features(x)
    return model.forward_head(x, pre_logits=True)

def compute_embeddings(models, dataset, label):
    
    # Use the first model to get config, assuming all models are the same
    model_cfg = models[0].default_cfg
    if "dinov3" in model_cfg['architecture']:
        input_size = (512,512)
    else:
        input_size = model_cfg['test_input_size'][1:] if 'test_input_size' in model_cfg.keys() else model_cfg['input_size'][1:]
    transform = transforms.Compose([
        transforms.Resize(
            input_size,
            interpolation=interpolation_modes[model_cfg['interpolation']]
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            model_cfg['mean'],
            model_cfg['std'])
    ])
    
    dataset.transform = transform
    
    # Base batch size per GPU
    base_bs = 64
    
    # Multiply batch size by number of GPUs
    total_bs = base_bs * len(devices)
    
    num_workers = 30
    loader = torch.utils.data.DataLoader(dataset, batch_size=total_bs, num_workers=num_workers)
    
    all_batches = []
    
    for imageBatch, _ in tqdm.tqdm(loader, desc=f"Computing embeddings for {label}"):
        # Split the batch across the number of GPUs
        chunk_size = (imageBatch.size(0) + len(devices) - 1) // len(devices)
        image_chunks = list(imageBatch.split(chunk_size))
        
        # Ensure number of chunks matches number of devices for zipping
        # This handles the case where the last batch is smaller
        if len(image_chunks) != len(devices):
            # Create empty tensors for unused GPUs if the last batch is small
            active_devices = devices[:len(image_chunks)]
        else:
            active_devices = devices

        with torch.autograd.grad_mode.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=autocast_dtype):
            batch_embeds_list = [
                fw_enc(model, chunk.to(device, non_blocking=True)).to('cpu', non_blocking=True)
                for model, chunk, device in zip(models, image_chunks, active_devices)
            ]

        torch.cuda.synchronize()
        
        all_batches.append(torch.cat(batch_embeds_list, dim=0))

    embeds = torch.cat(all_batches, dim=0)

    save_path = f'/scratch/fyguan/embeds_{label}_{model_cfg["architecture"]}.{model_cfg["tag"]}.pt'
    print(f"Saving embeddings to {save_path}")
    torch.save(embeds, save_path)

    del all_batches
    del embeds
    gc.collect()
 
model_ids = [
    
    'vit_7b_patch16_dinov3.lvd1689m',
    'vit_pe_core_gigantic_patch14_448.fb',
    'vit_pe_lang_gigantic_patch14_448.fb',
    'vit_pe_spatial_gigantic_patch14_448.fb',
    'eva02_large_patch14_224.mim_m38m',
]

for model_id in model_ids:
    print(f"Processing model: {model_id} on {len(devices)} GPUs")
    
    # Create a list of models, one for each GPU
    models = [timm.create_model(model_id, pretrained=True, num_classes=1000).to(device).eval() for device in devices]
    
    compute_embeddings(models, in1k_val, "in1k_val")
    compute_embeddings(models, in1k_train, "in1k_train")
    compute_embeddings(models, ds_train, "cc12m")
    
    # Clean up
    del models
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    
    
    