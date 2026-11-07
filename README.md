# Localizarea si clasificarea evenimentelor audio-vizuale prin compararea modalitatilor audio, video si audio-video

Fiecare segment de 1 secunda dintr-un videoclip AVE este clasificat intr-una din
28 de clase de eveniment sau in clasa *fundal* (29 de clase in total). Se compara
trei configuratii: **doar audio**, **doar video** si **audio + video**.

## Structura

```
data/
  AVE_Dataset/          Annotations.txt, train/val/testSet.txt (adnotarile AVE)
  splits/               train.txt, val.txt, test.txt - indicii clipurilor pe split
  features/             audio_feature.h5, visual_feature.h5   (nu sunt pe git)
src/
  config.py             cai si constante
  data/
    annotations.py      citirea adnotarilor, clase, etichete per segment, split-uri
    transforms.py       transformari pe features (pooling, normalizare, augmentare)
    dataset.py          AVEDataset (torch Dataset) + build_dataloaders
  preprocessing/
    extract_features.py videoclip -> features VGG19 (video) si VGGish (audio)
scripts/
  make_splits.py        genereaza data/splits/*.txt
  check_data.py         verifica Dataset-ul si DataLoader-ele
```

## Date

Setul de date **AVE** (Tian et al., ECCV 2018): 4143 de clipuri de 10 s, 28 de
categorii de evenimente, adnotate temporal (inceputul si sfarsitul evenimentului).

### Split train / val / test

Se foloseste split-ul oficial AVE, ca rezultatele sa fie comparabile cu literatura:

| split | clipuri | proportie | segmente de 1 s |
|-------|--------:|----------:|----------------:|
| train | 3339    | ~80%      | 33390           |
| val   | 402     | ~10%      | 4020            |
| test  | 402     | ~10%      | 4020            |

Split-urile nu se suprapun. Fisierele `data/splits/*.txt` contin indicii
randurilor din `Annotations.txt` si se pot regenera cu `python -m scripts.make_splits`.

### Pre-procesare

1. **Offline**: `src/preprocessing/extract_features.py` extrage features cu encodere
   preantrenate si inghetate:
   - **video**: 16 cadre/s -> resize 224x224 + normalizare ImageNet -> VGG19 (pool5)
     -> medie pe secunda -> `(10, 7, 7, 512)`
   - **audio**: mono 16 kHz -> spectrograma log-mel 96x64 pe secunda -> VGGish -> `(10, 128)`
2. **La incarcare** (`src/data/transforms.py`):
   - `SpatialAvgPool`: `(10, 7, 7, 512) -> (10, 512)` (optional, pentru clasificatoare compacte)
   - `Standardize`: normalizare per canal, cu medie/abatere calculate **doar pe train**
   - `GaussianNoise`, `FeatureDropout`: augmentari aplicate **doar pe train**
3. **Etichete**: construite din `Annotations.txt`; secundele `[start, end)` primesc clasa
   evenimentului, restul primesc clasa 28 (fundal).

### Obtinerea features

Fisierele `.h5` (~8 GB pentru video) nu sunt incluse in repository. Fie le extragi
cu scriptul de mai sus (videoclipurile in `data/AVE_Dataset/AVE/`), fie folosesti
features-urile publicate de autorii AVE
([github.com/YapengTian/AVE-ECCV18](https://github.com/YapengTian/AVE-ECCV18)).
Ca sa nu fie copiate, caile catre ele se pun intr-un fisier local
`local_paths.json` (nu e pus pe git), pornind de la `local_paths.example.json`:

```json
{
  "audio_h5": "cale/catre/audio_feature.h5",
  "video_h5": "cale/catre/visual_feature.h5"
}
```

Alternativ: variabilele de mediu `AVE_AUDIO_H5` / `AVE_VIDEO_H5`, sau fisierele
puse direct in `data/features/`.

## Utilizare

### Mediu virtual (`.venv`)

Testat cu Python 3.12 pe Windows, GPU NVIDIA (CUDA 12.1).

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1            # Git Bash: source .venv/Scripts/activate
python -m pip install --upgrade pip
# PyTorch cu suport CUDA 12.1 (pentru CPU: scoate --index-url)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Rulare

```bash
python -m scripts.check_data          # verifica split-urile si DataLoader-ele
```

```python
from src.data import build_dataloaders

loaders = build_dataloaders(modality="av", batch_size=64)   # "audio" | "video" | "av"
for batch in loaders["train"]:
    audio, video, label = batch["audio"], batch["video"], batch["label"]
    # audio (B, 10, 128), video (B, 10, 7, 7, 512), label (B, 10) cu valori 0..28
```

Pentru resurse limitate: `video_pool=True` (video devine `(B, 10, 512)`) si/sau
`subset_fraction=0.3` (antrenare pe 30% din train).
