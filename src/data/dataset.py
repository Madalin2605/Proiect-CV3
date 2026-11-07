"""Dataset si DataLoader pentru AVE (Audio-Visual Event localization).

Fiecare esantion = un clip de 10 s, cu:
    audio : (10, 128)               float32   - features VGGish per secunda
    video : (10, 7, 7, 512)         float32   - features VGG19 pool5 per secunda
            sau (10, 512) daca video_pool=True
    label : (10,)                   int64     - clasa per secunda (0..27, 28 = fundal)
    index : int                               - randul din Annotations.txt

Parametrul `modality` ("audio" | "video" | "av") alege ce features se incarca,
corespunzator celor trei configuratii comparate in proiect.
"""
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src import config
from src.data.annotations import load_annotations, load_split, segment_labels
from src.data.transforms import SpatialAvgPool, build_transforms

MODALITIES = ("audio", "video", "av")


class AVEDataset(Dataset):
    def __init__(self, split: str, modality: str = "av", transform=None,
                 audio_h5: Path = config.AUDIO_H5, video_h5: Path = config.VIDEO_H5,
                 subset_fraction: float = 1.0, seed: int = config.SEED):
        if modality not in MODALITIES:
            raise ValueError(f"modality trebuie sa fie una din {MODALITIES}")
        self.split, self.modality, self.transform = split, modality, transform
        self.use_audio = modality in ("audio", "av")
        self.use_video = modality in ("video", "av")
        self.audio_h5, self.video_h5 = Path(audio_h5), Path(video_h5)

        indices = load_split(split)
        if subset_fraction < 1.0:  # subset reproductibil, pentru resurse limitate
            rng = np.random.default_rng(seed)
            n = max(1, int(round(len(indices) * subset_fraction)))
            indices = np.sort(rng.choice(indices, size=n, replace=False))
        self.indices = indices
        self.labels = segment_labels(load_annotations())  # (N_total, 10)

        # Audio e mic (~40 MB) -> se incarca integral in memorie.
        self.audio = None
        if self.use_audio:
            with h5py.File(self.audio_h5, "r") as hf:
                self.audio = hf[config.H5_KEY][:].astype(np.float32)

        # Video e mare (~8 GB) -> se citeste lazy din .h5, per esantion.
        # Handle-ul se deschide in fiecare worker (h5py nu se poate serializa).
        self._video = None
        if self.use_video:
            with h5py.File(self.video_h5, "r") as hf:
                n_video = hf[config.H5_KEY].shape[0]
            if n_video != len(self.labels):
                raise ValueError(f"{self.video_h5}: {n_video} clipuri, adnotari: {len(self.labels)}")

    def _video_ds(self):
        if self._video is None:
            self._video = h5py.File(self.video_h5, "r")[config.H5_KEY]
        return self._video

    def __getstate__(self):  # pentru DataLoader cu num_workers > 0
        state = self.__dict__.copy()
        state["_video"] = None
        return state

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int) -> dict:
        idx = int(self.indices[i])
        sample = {"label": torch.from_numpy(self.labels[idx]), "index": idx}
        if self.use_audio:
            sample["audio"] = torch.from_numpy(self.audio[idx])
        if self.use_video:
            sample["video"] = torch.from_numpy(self._video_ds()[idx].astype(np.float32))
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def compute_stats(dataset: AVEDataset, video_pool: bool, max_clips: int = 400,
                  seed: int = config.SEED) -> dict:
    """Media/abaterea per canal pe split-ul de TRAIN (fara data leakage).

    Pentru video se foloseste un esantion de `max_clips` clipuri (citirea
    integrala a celor ~8 GB ar dura mult); pentru audio se folosesc toate.
    """
    stats = {}
    if dataset.use_audio:
        a = dataset.audio[dataset.indices].reshape(-1, config.AUDIO_DIM).astype(np.float64)
        stats["audio"] = (a.mean(0), a.std(0))
    if dataset.use_video:
        rng = np.random.default_rng(seed)
        idx = dataset.indices
        if len(idx) > max_clips:
            idx = rng.choice(idx, size=max_clips, replace=False)
        pool = SpatialAvgPool()
        s = sq = 0.0
        n = 0
        vds = dataset._video_ds()
        for j in np.sort(idx):
            v = torch.from_numpy(vds[int(j)].astype(np.float32))
            if video_pool:
                v = pool({"video": v})["video"]
            v = v.reshape(-1, v.shape[-1]).double()
            s, sq, n = s + v.sum(0), sq + (v ** 2).sum(0), n + v.shape[0]
        mean = s / n
        stats["video"] = (mean.numpy(), (sq / n - mean ** 2).clamp_min(0).sqrt().numpy())
    return stats


def load_or_compute_stats(modality: str, video_pool: bool, **ds_kwargs) -> dict:
    """Statisticile se salveaza in data/stats/ ca sa nu fie recalculate."""
    config.STATS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.STATS_DIR / f"stats_{'pool' if video_pool else 'full'}.npz"
    stats = {}
    if path.exists():
        f = np.load(path)
        stats = {k: (f[f"{k}_mean"], f[f"{k}_std"]) for k in ("audio", "video") if f"{k}_mean" in f}
    needed = {"audio": modality in ("audio", "av"), "video": modality in ("video", "av")}
    missing = [k for k, need in needed.items() if need and k not in stats]
    if missing:
        train = AVEDataset("train", modality="av" if len(missing) == 2 else missing[0], **ds_kwargs)
        stats.update(compute_stats(train, video_pool))
        np.savez(path, **{f"{k}_{p}": v for k, (m, s) in stats.items() for p, v in (("mean", m), ("std", s))})
    return {k: v for k, v in stats.items() if needed[k]}


def build_dataloaders(modality: str = "av", batch_size: int = 64, video_pool: bool = False,
                      augment: bool = True, num_workers: int = 0, subset_fraction: float = 1.0,
                      **ds_kwargs) -> dict[str, DataLoader]:
    """Construieste DataLoader-ele pentru train/val/test.

    - normalizarea foloseste statistici calculate doar pe train;
    - augmentarea (zgomot + feature dropout) se aplica doar pe train;
    - train e amestecat (shuffle), val/test nu.
    """
    stats = load_or_compute_stats(modality, video_pool, **ds_kwargs)
    loaders = {}
    for split in ("train", "val", "test"):
        is_train = split == "train"
        tf = build_transforms(stats, video_pool, train=is_train and augment)
        ds = AVEDataset(split, modality, transform=tf,
                        subset_fraction=subset_fraction if is_train else 1.0, **ds_kwargs)
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=is_train, num_workers=num_workers,
            pin_memory=torch.cuda.is_available(), drop_last=False,
            persistent_workers=num_workers > 0,
        )
    return loaders
