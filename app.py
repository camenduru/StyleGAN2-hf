#!/usr/bin/env python

from __future__ import annotations

import argparse
import functools
import os
import pickle
import sys

sys.path.insert(0, 'stylegan3')

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

ORIGINAL_REPO_URL = 'https://github.com/NVlabs/stylegan3'
TITLE = 'StyleGAN2'
DESCRIPTION = f'This is a demo for {ORIGINAL_REPO_URL}.'
SAMPLE_IMAGE_DIR = 'https://huggingface.co/spaces/hysts/StyleGAN2/resolve/main/samples'
ARTICLE = f'''## Generated images
- truncation: 0.7
### CIFAR-10
- size: 32x32
- class index: 0-9
- seed: 0-9
![CIFAR-10 samples]({SAMPLE_IMAGE_DIR}/cifar10.jpg)
### AFHQ-Cat
- size: 512x512
- seed: 0-99
![AFHQ-Cat samples]({SAMPLE_IMAGE_DIR}/afhq-cat.jpg)
### AFHQ-Dog
- size: 512x512
- seed: 0-99
![AFHQ-Dog samples]({SAMPLE_IMAGE_DIR}/afhq-dog.jpg)
### AFHQ-Wild
- size: 512x512
- seed: 0-99
![AFHQ-Wild samples]({SAMPLE_IMAGE_DIR}/afhq-wild.jpg)
### AFHQv2
- size: 512x512
- seed: 0-99
![AFHQv2 samples]({SAMPLE_IMAGE_DIR}/afhqv2.jpg)
### LSUN-Dog
- size: 256x256
- seed: 0-99
![LSUN-Dog samples]({SAMPLE_IMAGE_DIR}/lsun-dog.jpg)
### BreCaHAD
- size: 512x512
- seed: 0-99
![BreCaHAD samples]({SAMPLE_IMAGE_DIR}/brecahad.jpg)
### CelebA-HQ
- size: 256x256
- seed: 0-99
![CelebA-HQ samples]({SAMPLE_IMAGE_DIR}/celebahq.jpg)
### FFHQ
- size: 1024x1024
- seed: 0-99
![FFHQ samples]({SAMPLE_IMAGE_DIR}/ffhq.jpg)
### FFHQ-U
- size: 1024x1024
- seed: 0-99
![FFHQ-U samples]({SAMPLE_IMAGE_DIR}/ffhq-u.jpg)
### MetFaces
- size: 1024x1024
- seed: 0-99
![MetFaces samples]({SAMPLE_IMAGE_DIR}/metfaces.jpg)
### MetFaces-U
- size: 1024x1024
- seed: 0-99
![MetFaces-U samples]({SAMPLE_IMAGE_DIR}/metfaces-u.jpg)
'''

TOKEN = os.environ['TOKEN']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--theme', type=str)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--port', type=int)
    parser.add_argument('--disable-queue',
                        dest='enable_queue',
                        action='store_false')
    parser.add_argument('--allow-flagging', type=str, default='never')
    parser.add_argument('--allow-screenshot', action='store_true')
    return parser.parse_args()


def generate_z(z_dim: int, seed: int, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.random.RandomState(seed).randn(
        1, z_dim)).to(device).float()


@torch.inference_mode()
def generate_image(model_name: str, class_index: int, seed: int,
                   truncation_psi: float, model_dict: dict[str, nn.Module],
                   device: torch.device) -> np.ndarray:
    model = model_dict[model_name]
    seed = int(np.clip(seed, 0, np.iinfo(np.uint32).max))

    z = generate_z(model.z_dim, seed, device)
    label = torch.zeros([1, model.c_dim], device=device)
    class_index = round(class_index)
    class_index = min(max(0, class_index), model.c_dim - 1)
    class_index = torch.tensor(class_index, dtype=torch.long)
    if class_index >= 0:
        label[:, class_index] = 1

    out = model(z, label, truncation_psi=truncation_psi)
    out = (out.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return out[0].cpu().numpy()


def load_model(file_name: str, device: torch.device) -> nn.Module:
    path = hf_hub_download('hysts/StyleGAN2',
                           f'models/{file_name}',
                           use_auth_token=TOKEN)
    with open(path, 'rb') as f:
        model = pickle.load(f)['G_ema']
    model.eval()
    model.to(device)
    with torch.inference_mode():
        z = torch.zeros((1, model.z_dim)).to(device)
        label = torch.zeros([1, model.c_dim], device=device)
        model(z, label)
    return model


def main():
    gr.close_all()

    args = parse_args()
    device = torch.device(args.device)

    model_names = {
        'AFHQ-Cat-512': 'stylegan2-afhqcat-512x512.pkl',
        'AFHQ-Dog-512': 'stylegan2-afhqdog-512x512.pkl',
        'AFHQv2-512': 'stylegan2-afhqv2-512x512.pkl',
        'AFHQ-Wild-512': 'stylegan2-afhqwild-512x512.pkl',
        'BreCaHAD-512': 'stylegan2-brecahad-512x512.pkl',
        'CelebA-HQ-256': 'stylegan2-celebahq-256x256.pkl',
        'CIFAR-10': 'stylegan2-cifar10-32x32.pkl',
        'FFHQ-256': 'stylegan2-ffhq-256x256.pkl',
        'FFHQ-512': 'stylegan2-ffhq-512x512.pkl',
        'FFHQ-1024': 'stylegan2-ffhq-1024x1024.pkl',
        'FFHQ-U-256': 'stylegan2-ffhqu-256x256.pkl',
        'FFHQ-U-1024': 'stylegan2-ffhqu-1024x1024.pkl',
        'LSUN-Dog-256': 'stylegan2-lsundog-256x256.pkl',
        'MetFaces-1024': 'stylegan2-metfaces-1024x1024.pkl',
        'MetFaces-U-1024': 'stylegan2-metfacesu-1024x1024.pkl',
    }

    model_dict = {
        name: load_model(file_name, device)
        for name, file_name in model_names.items()
    }

    func = functools.partial(generate_image,
                             model_dict=model_dict,
                             device=device)
    func = functools.update_wrapper(func, generate_image)

    gr.Interface(
        func,
        [
            gr.inputs.Radio(list(model_names.keys()),
                            type='value',
                            default='FFHQ-1024',
                            label='Model'),
            gr.inputs.Number(default=0, label='Class index'),
            gr.inputs.Number(default=0, label='Seed'),
            gr.inputs.Slider(
                0, 2, step=0.05, default=0.7, label='Truncation psi'),
        ],
        gr.outputs.Image(type='numpy', label='Output'),
        title=TITLE,
        description=DESCRIPTION,
        article=ARTICLE,
        theme=args.theme,
        allow_screenshot=args.allow_screenshot,
        allow_flagging=args.allow_flagging,
        live=args.live,
    ).launch(
        enable_queue=args.enable_queue,
        server_port=args.port,
        share=args.share,
    )


if __name__ == '__main__':
    main()
