from __future__ import annotations

import os
import pathlib
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

current_dir = pathlib.Path(__file__).parent
submodule_dir = current_dir / 'stylegan3'
sys.path.insert(0, submodule_dir.as_posix())

HF_TOKEN = os.environ['HF_TOKEN']


class Model:
    MODEL_NAME_DICT = {
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

    def __init__(self, device: str | torch.device):
        self.device = torch.device(device)
        self._download_all_models()
        self.model_name = 'FFHQ-1024'
        self.model = self._load_model(self.model_name)

    def _load_model(self, model_name: str) -> nn.Module:
        file_name = self.MODEL_NAME_DICT[model_name]
        path = hf_hub_download('hysts/StyleGAN2',
                               f'models/{file_name}',
                               use_auth_token=HF_TOKEN)
        with open(path, 'rb') as f:
            model = pickle.load(f)['G_ema']
        model.eval()
        model.to(self.device)
        return model

    def set_model(self, model_name: str) -> None:
        if model_name == self.model_name:
            return
        self.model_name = model_name
        self.model = self._load_model(model_name)

    def _download_all_models(self):
        for name in self.MODEL_NAME_DICT.keys():
            self._load_model(name)

    def generate_z(self, seed: int) -> torch.Tensor:
        seed = int(np.clip(seed, 0, np.iinfo(np.uint32).max))
        z = np.random.RandomState(seed).randn(1, self.model.z_dim)
        return torch.from_numpy(z).float().to(self.device)

    def postprocess(self, tensor: torch.Tensor) -> np.ndarray:
        tensor = (tensor.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(
            torch.uint8)
        return tensor.cpu().numpy()

    def make_label_tensor(self, class_index: int) -> torch.Tensor:
        class_index = round(class_index)
        class_index = min(max(0, class_index), self.model.c_dim - 1)
        class_index = torch.tensor(class_index, dtype=torch.long)

        label = torch.zeros([1, self.model.c_dim], device=self.device)
        if class_index >= 0:
            label[:, class_index] = 1
        return label

    @torch.inference_mode()
    def generate(self, z: torch.Tensor, label: torch.Tensor,
                 truncation_psi: float) -> torch.Tensor:
        return self.model(z, label, truncation_psi=truncation_psi)

    def generate_image(self, seed: int, truncation_psi: float,
                       class_index: int) -> np.ndarray:
        z = self.generate_z(seed)
        label = self.make_label_tensor(class_index)

        out = self.generate(z, label, truncation_psi)
        out = self.postprocess(out)
        return out[0]

    def set_model_and_generate_image(self, model_name: str, seed: int,
                                     truncation_psi: float,
                                     class_index: int) -> np.ndarray:
        self.set_model(model_name)
        return self.generate_image(seed, truncation_psi, class_index)
