"""Citirea adnotarilor AVE si construirea etichetelor la nivel de segment.

Annotations.txt are cate o linie per esantion:
    Category&VideoID&Quality&StartTime&EndTime
Fiecare esantion e un clip de 10 s; segmentele [StartTime, EndTime) apartin
evenimentului, restul sunt fundal. Randul i din Annotations.txt corespunde
randului i din fisierele de features .h5.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src import config

# Ordinea claselor = ordinea primei aparitii in Annotations.txt
# (aceeasi ca in labels.h5 din implementarea originala AVE-ECCV18).
CATEGORIES = [
    "Church bell", "Male speech, man speaking", "Bark",
    "Fixed-wing aircraft, airplane", "Race car, auto racing",
    "Female speech, woman speaking", "Helicopter", "Violin, fiddle", "Flute",
    "Ukulele", "Frying (food)", "Truck", "Shofar", "Motorcycle",
    "Acoustic guitar", "Train horn", "Clock", "Banjo", "Goat",
    "Baby cry, infant cry", "Bus", "Chainsaw", "Cat", "Horse", "Toilet flush",
    "Rodents, rats, mice", "Accordion", "Mandolin",
]
BACKGROUND = "Background"
CLASS_NAMES = CATEGORIES + [BACKGROUND]
BACKGROUND_ID = len(CATEGORIES)          # 28
NUM_CLASSES = len(CLASS_NAMES)           # 29
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Sample:
    index: int        # randul din Annotations.txt / din fisierele .h5
    category: str
    video_id: str
    start: int
    end: int


def load_annotations(path: Path = config.ANNOTATIONS_FILE) -> list[Sample]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    for line in lines:
        if line.startswith("Category&"):
            continue
        category, video_id, _quality, start, end = line.split("&")
        samples.append(Sample(len(samples), category, video_id, int(start), int(end)))
    return samples


def segment_labels(samples: list[Sample]) -> np.ndarray:
    """Etichete (N, 10) int64: id-ul clasei pentru fiecare secunda, 28 = fundal."""
    labels = np.full((len(samples), config.NUM_SEGMENTS), BACKGROUND_ID, dtype=np.int64)
    for s in samples:
        labels[s.index, s.start:s.end] = CATEGORIES.index(s.category)
    return labels


def load_split(split: str, splits_dir: Path = config.SPLITS_DIR) -> np.ndarray:
    """Indicii (randuri din Annotations.txt) pentru split-ul cerut."""
    if split not in SPLITS:
        raise ValueError(f"split trebuie sa fie unul din {SPLITS}, nu {split!r}")
    return np.loadtxt(splits_dir / f"{split}.txt", dtype=np.int64)
