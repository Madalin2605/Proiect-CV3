"""Genereaza data/splits/{train,val,test}.txt (indici in Annotations.txt).

Se foloseste split-ul oficial AVE (Tian et al., ECCV 2018):
3339 / 402 / 402 clipuri (~80% / 10% / 10%).

Sursa poate fi:
  * fisierele *_order.h5 din implementarea originala (--order-dir), sau
  * fisierele trainSet/valSet/testSet.txt (implicit), potrivite cu randurile
    din Annotations.txt. Randurile identice duplicate sunt alocate in ordine.

Utilizare:
    python -m scripts.make_splits
    python -m scripts.make_splits --order-dir path/catre/AVE-ECCV18/data
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from src import config
from src.data.annotations import SPLITS, load_annotations


def from_txt() -> dict[str, np.ndarray]:
    samples = load_annotations()
    free = defaultdict(list)  # (cat, id, start, end) -> randuri inca nealocate
    for s in samples:
        free[(s.category, s.video_id, s.start, s.end)].append(s.index)
    out = {}
    for split in SPLITS:
        rows = []
        for line in open(config.ANNOTATIONS_DIR / f"{split}Set.txt", encoding="utf-8"):
            if not line.strip() or line.startswith("Category&"):
                continue
            cat, vid, _q, st, en = line.strip().split("&")
            rows.append(free[(cat, vid, int(st), int(en))].pop(0))
        out[split] = np.array(rows, dtype=np.int64)
    return out


def from_order_h5(order_dir: Path) -> dict[str, np.ndarray]:
    import h5py
    out = {}
    for split in SPLITS:
        with h5py.File(order_dir / f"{split}_order.h5", "r") as hf:
            out[split] = hf["order"][:].astype(np.int64)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--order-dir", type=Path, default=None)
    args = p.parse_args()

    splits = from_order_h5(args.order_dir) if args.order_dir else from_txt()
    config.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for split, idx in splits.items():
        np.savetxt(config.SPLITS_DIR / f"{split}.txt", idx, fmt="%d")
        print(f"{split:5s}: {len(idx)} clipuri -> {config.SPLITS_DIR / f'{split}.txt'}")


if __name__ == "__main__":
    main()
