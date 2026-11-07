"""Transformari aplicate pe features (nu pe imagini brute).

Pre-procesarea "grea" (cadre -> VGG19, audio -> log-mel -> VGGish) se face o
singura data, offline, in src/preprocessing/extract_features.py. Aici sunt
transformarile aplicate la incarcare, pe fiecare esantion:

  * SpatialAvgPool  - (10, 7, 7, 512) -> (10, 512), pentru clasificatoare compacte
  * Standardize     - (x - mean) / std, cu statistici calculate DOAR pe train
  * GaussianNoise   - augmentare: zgomot gaussian (doar la antrenare)
  * FeatureDropout  - augmentare: anuleaza aleator dimensiuni (doar la antrenare)

Fiecare transformare primeste si intoarce un dict {"audio", "video", "label", ...}.
"""
import torch


class Compose:
    def __init__(self, transforms):
        self.transforms = [t for t in transforms if t is not None]

    def __call__(self, sample: dict) -> dict:
        for t in self.transforms:
            sample = t(sample)
        return sample

    def __repr__(self):
        inner = "\n".join(f"    {t!r}" for t in self.transforms)
        return f"Compose(\n{inner}\n)"


class SpatialAvgPool:
    """Medie peste harta spatiala 7x7: (T, 7, 7, C) -> (T, C)."""

    def __call__(self, sample):
        if "video" in sample and sample["video"].dim() == 4:
            sample["video"] = sample["video"].mean(dim=(1, 2))
        return sample

    def __repr__(self):
        return "SpatialAvgPool()"


class Standardize:
    """Normalizare per canal (ultima dimensiune) cu media/abaterea din train."""

    def __init__(self, stats: dict, eps: float = 1e-6):
        self.stats = {k: (torch.as_tensor(m, dtype=torch.float32),
                          torch.as_tensor(s, dtype=torch.float32).clamp_min(eps))
                      for k, (m, s) in stats.items()}

    def __call__(self, sample):
        for key, (mean, std) in self.stats.items():
            if key in sample:
                sample[key] = (sample[key] - mean) / std
        return sample

    def __repr__(self):
        return f"Standardize(keys={list(self.stats)})"


class GaussianNoise:
    def __init__(self, std: float = 0.1, keys=("audio", "video")):
        self.std, self.keys = std, keys

    def __call__(self, sample):
        for key in self.keys:
            if key in sample:
                sample[key] = sample[key] + self.std * torch.randn_like(sample[key])
        return sample

    def __repr__(self):
        return f"GaussianNoise(std={self.std}, keys={self.keys})"


class FeatureDropout:
    """Anuleaza aleator canale intregi (aceleasi pentru toate segmentele)."""

    def __init__(self, p: float = 0.1, keys=("audio", "video")):
        self.p, self.keys = p, keys

    def __call__(self, sample):
        for key in self.keys:
            if key in sample:
                x = sample[key]
                keep = (torch.rand(x.shape[-1]) >= self.p).to(x.dtype)
                sample[key] = x * keep / (1.0 - self.p)
        return sample

    def __repr__(self):
        return f"FeatureDropout(p={self.p}, keys={self.keys})"


def build_transforms(stats: dict | None, video_pool: bool, train: bool,
                     noise_std: float = 0.1, dropout_p: float = 0.1) -> Compose:
    return Compose([
        SpatialAvgPool() if video_pool else None,
        Standardize(stats) if stats else None,
        GaussianNoise(noise_std) if train and noise_std > 0 else None,
        FeatureDropout(dropout_p) if train and dropout_p > 0 else None,
    ])
