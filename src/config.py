"""Cai si constante globale ale proiectului.

Caile catre fisierele mari (features .h5) se aleg, in ordinea prioritatii, din:
  1. variabilele de mediu AVE_AUDIO_H5 / AVE_VIDEO_H5
  2. fisierul local `local_paths.json` din radacina repository-ului
     (nu e pus pe git), de forma:
         {"audio_h5": "D:/.../audio_feature.h5", "video_h5": "D:/.../visual_feature.h5"}
  3. implicit: data/features/audio_feature.h5 si data/features/visual_feature.h5
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_LOCAL = ROOT / "local_paths.json"
_local_paths = json.loads(_LOCAL.read_text(encoding="utf-8")) if _LOCAL.exists() else {}
DATA_DIR = ROOT / "data"

ANNOTATIONS_DIR = DATA_DIR / "AVE_Dataset"
ANNOTATIONS_FILE = ANNOTATIONS_DIR / "Annotations.txt"
VIDEO_DIR = ANNOTATIONS_DIR / "AVE"  # videoclipurile .mp4 (doar pentru extragere)

SPLITS_DIR = DATA_DIR / "splits"
STATS_DIR = DATA_DIR / "stats"
FEATURES_DIR = DATA_DIR / "features"

AUDIO_H5 = Path(os.environ.get("AVE_AUDIO_H5")
                or _local_paths.get("audio_h5")
                or FEATURES_DIR / "audio_feature.h5")
VIDEO_H5 = Path(os.environ.get("AVE_VIDEO_H5")
                or _local_paths.get("video_h5")
                or FEATURES_DIR / "visual_feature.h5")
H5_KEY = "avadataset"  # cheia folosita in fisierele .h5

NUM_SEGMENTS = 10      # fiecare clip AVE are 10 segmente de 1 secunda
AUDIO_DIM = 128        # VGGish
VIDEO_SHAPE = (7, 7, 512)  # VGG19 pool5
SEED = 3344
