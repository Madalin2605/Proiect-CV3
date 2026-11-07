"""Verifica Dataset-ul si DataLoader-ele AVE.

    python -m scripts.check_data
    python -m scripts.check_data --audio-h5 cale/audio_feature.h5 --video-h5 cale/visual_feature.h5

Implicit, features sunt cautate in data/features/ (vezi src/config.py).

Afiseaza: dimensiunea si proportia split-urilor, lipsa suprapunerilor,
distributia claselor la nivel de segment, forma batch-urilor pentru fiecare
modalitate (audio / video / av) si efectul normalizarii.
"""
import argparse
import time
from pathlib import Path

import numpy as np

from src import config
from src.data.annotations import (BACKGROUND_ID, CLASS_NAMES, NUM_CLASSES, SPLITS,
                                  load_annotations, load_split, segment_labels)
from src.data.dataset import build_dataloaders


def check_splits():
    print("=" * 70 + "\nSPLIT-URI\n" + "=" * 70)
    idx = {s: load_split(s) for s in SPLITS}
    total = sum(len(v) for v in idx.values())
    for s, v in idx.items():
        print(f"  {s:5s}: {len(v):5d} clipuri ({100 * len(v) / total:5.1f}%)")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        inter = np.intersect1d(idx[a], idx[b])
        assert len(inter) == 0, f"suprapunere {a}/{b}: {len(inter)}"
    print(f"  total: {total} clipuri, fara suprapuneri intre split-uri")

    labels = segment_labels(load_annotations())
    print("\nDISTRIBUTIA CLASELOR (segmente de 1 s)")
    counts = {s: np.bincount(labels[v].ravel(), minlength=NUM_CLASSES) for s, v in idx.items()}
    print(f"  {'clasa':32s}" + "".join(f"{s:>8s}" for s in SPLITS))
    for c in range(NUM_CLASSES):
        print(f"  {c:2d} {CLASS_NAMES[c][:29]:29s}" + "".join(f"{counts[s][c]:8d}" for s in SPLITS))
    for s in SPLITS:
        bg = counts[s][BACKGROUND_ID] / counts[s].sum()
        print(f"  {s}: {counts[s].sum()} segmente, {100 * bg:.1f}% fundal")


def check_loaders(modality, video_pool, n_batches, ds_kwargs):
    tag = f"modality={modality}, video_pool={video_pool}"
    print("\n" + "=" * 70 + f"\nDATALOADER ({tag})\n" + "=" * 70)
    t0 = time.time()
    loaders = build_dataloaders(modality, batch_size=32, video_pool=video_pool, **ds_kwargs)
    print(f"  construit in {time.time() - t0:.1f}s")
    for split, dl in loaders.items():
        print(f"  {split:5s}: {len(dl.dataset)} clipuri, {len(dl)} batch-uri")
    print(f"  transforms train:\n{loaders['train'].dataset.transform}")

    for split in ("train", "val"):
        t0 = time.time()
        for b, batch in enumerate(loaders[split]):
            if b == 0:
                shapes = {k: tuple(v.shape) for k, v in batch.items() if hasattr(v, "shape")}
                print(f"  [{split}] batch: {shapes}")
                for k in ("audio", "video"):
                    if k in batch:
                        x = batch[k]
                        print(f"  [{split}] {k}: dtype={x.dtype}, mean={x.mean():.3f}, "
                              f"std={x.std():.3f}, finite={bool(x.isfinite().all())}")
                y = batch["label"]
                assert y.min() >= 0 and y.max() < NUM_CLASSES
            if b + 1 >= n_batches:
                break
        print(f"  [{split}] {b + 1} batch-uri in {time.time() - t0:.2f}s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio-h5", type=Path, default=config.AUDIO_H5)
    p.add_argument("--video-h5", type=Path, default=config.VIDEO_H5)
    p.add_argument("--batches", type=int, default=3)
    args = p.parse_args()
    ds_kwargs = dict(audio_h5=args.audio_h5, video_h5=args.video_h5)

    check_splits()
    if not args.audio_h5.exists() or not args.video_h5.exists():
        print(f"\nFisierele de features lipsesc ({args.audio_h5}, {args.video_h5}).")
        print("Vezi README -> 'Date'. Verificarea DataLoader-elor a fost sarita.")
        return
    check_loaders("audio", False, args.batches, ds_kwargs)
    check_loaders("video", True, args.batches, ds_kwargs)
    check_loaders("av", False, args.batches, ds_kwargs)
    print("\nOK - Dataset si DataLoader functionale.")


if __name__ == "__main__":
    main()
