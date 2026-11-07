"""Pre-procesare offline: videoclipuri AVE -> features audio si video (.h5).

  * video: 16 cadre / secunda -> Resize 224 + normalizare ImageNet -> VGG19
           (pana la pool5) -> medie pe cele 16 cadre -> (10, 7, 7, 512)
  * audio: ffmpeg -> mono 16 kHz -> pe fiecare secunda log-mel 96x64 ->
           VGGish -> (10, 128)

Randul i din fisierele .h5 corespunde randului i din Annotations.txt.
Rulare (din radacina repository-ului):
    python -m src.preprocessing.extract_features
"""
import os
import cv2
import torch
import numpy as np
import h5py
from torchvision import models, transforms
from tqdm import tqdm
import torchvggish

from src import config

VIDEO_DIR       = str(config.VIDEO_DIR)
ANNOTATIONS_DIR = str(config.ANNOTATIONS_DIR)
OUTPUT_AUDIO    = str(config.AUDIO_H5)
OUTPUT_VIDEO    = str(config.VIDEO_H5)
H5_KEY          = config.H5_KEY

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


def load_video_ids(annotations_dir):
    """
    Citeste video_id-ul fiecarui rand din Annotations.txt, in ordine.
    Un videoclip poate aparea pe mai multe randuri (4143 randuri, ~4097 clipuri
    unice); se pastreaza toate, ca randul i din .h5 sa corespunda randului i
    din adnotari (si deci etichetelor si split-urilor).
    """
    video_ids = []
    with open(os.path.join(annotations_dir, "Annotations.txt"), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Category&"):
                continue
            parts = line.split("&")
            if len(parts) < 2:
                continue
            video_ids.append(parts[1])
    return video_ids


class VideoFeatureExtractor:
    def __init__(self, device):
        self.device = device
        # VGG19 pre-antrenat pe ImageNet, extragem de la block5_pool
        vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        # Luam pana la features (fara classifier)
        self.model = vgg19.features.to(device).eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def extract(self, video_path, num_seconds=10, frames_per_second=16):
        """
        Extrage features VGG19 pool5 pentru un videoclip.
        - Samplez 16 frame-uri per secunda
        - Mediez features pentru cele 16 frame-uri → 1 vector per secunda
        - Output: numpy array (10, 512, 7, 7) → dupa average → (10, 7, 7, 512)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or total_frames <= 0:
            cap.release()
            return None

        # Calculeaza frame-urile de extras pentru fiecare secunda
        frame_interval = total_frames / num_seconds
        frame_indices = []
        for sec in range(num_seconds):
            for i in range(frames_per_second):
                idx = int(sec * frame_interval + (i / frames_per_second) * frame_interval)
                frame_indices.append(min(idx, total_frames - 1))

        # Citeste frame-urile necesare
        frames = {}
        for idx in set(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames[idx] = frame
        cap.release()

        # Extrage features per secunda
        second_features = []
        for sec in range(num_seconds):
            sec_frames = []
            for i in range(frames_per_second):
                idx = frame_indices[sec * frames_per_second + i]
                if idx in frames:
                    tensor = self.transform(frames[idx]).unsqueeze(0).to(self.device)
                    sec_frames.append(tensor)

            if not sec_frames:
                second_features.append(np.zeros((7, 7, 512)))
                continue

            batch = torch.cat(sec_frames, dim=0)
            with torch.no_grad():
                feat = self.model(batch)  # (16, 512, 7, 7)

            # Average peste cele 16 frame-uri → (512, 7, 7) → transpose → (7, 7, 512)
            feat = feat.mean(dim=0).cpu().numpy()
            feat = feat.transpose(1, 2, 0)  # (7, 7, 512)
            second_features.append(feat)

        return np.stack(second_features, axis=0)  # (10, 7, 7, 512)


class AudioFeatureExtractor:
    def __init__(self, device):
        self.device = device
        self.model = torchvggish.vggish().to(device).eval()

    def _wav_to_mel(self, segment, sr=16000):
        """
        Converteste un segment audio la mel spectrogram
        cu parametrii exacti asteptati de VGGish:
        - 96 frames x 64 mel bins → (1, 1, 96, 64)
        """
        import librosa

        # Parametrii VGGish
        n_fft       = 512
        hop_length  = int(sr * 0.010)   # 10ms hop
        win_length  = int(sr * 0.025)   # 25ms window
        n_mels      = 64
        fmin        = 125
        fmax        = 7500

        mel = librosa.feature.melspectrogram(
            y=segment.astype(np.float32),
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)  # (64, T)
        log_mel = log_mel.T  # (T, 64)

        # Crop sau pad la exact 96 frames
        if log_mel.shape[0] >= 96:
            log_mel = log_mel[:96, :]
        else:
            pad = 96 - log_mel.shape[0]
            log_mel = np.pad(log_mel, ((0, pad), (0, 0)))

        return log_mel  # (96, 64)

    def extract(self, video_path):
        import subprocess
        import soundfile as sf
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            result = subprocess.run([
                "ffmpeg", "-y",
                "-i", video_path,
                "-ac", "1",
                "-ar", "16000",
                tmp_path,
                "-loglevel", "error"
            ], capture_output=True)

            if result.returncode != 0:
                return None

            audio, sr = sf.read(tmp_path)
            if len(audio) == 0:
                return None

            features = []
            for sec in range(10):
                start = sec * sr
                end   = start + sr
                segment = audio[start:end]

                # Padding daca segmentul e prea scurt
                if len(segment) < sr:
                    segment = np.pad(segment, (0, sr - len(segment)))

                # Mel spectrogram cu parametrii VGGish: (96, 64)
                log_mel = self._wav_to_mel(segment, sr=sr)

                # Tensor (1, 1, 96, 64)
                log_mel_tensor = torch.tensor(
                    log_mel[np.newaxis, np.newaxis, :, :],
                    dtype=torch.float32
                ).to(self.device)

                with torch.no_grad():
                    feat = self.model.forward(log_mel_tensor)  # (1, 128)

                features.append(feat.cpu().numpy().squeeze())  # (128,)

            return np.stack(features, axis=0)  # (10, 128)

        except Exception as e:
            print(f"Eroare audio: {e}")
            return None
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    video_ids = load_video_ids(ANNOTATIONS_DIR)
    n = len(video_ids)
    print(f"Randuri in adnotari: {n} ({len(set(video_ids))} videoclipuri unice)")

    video_extractor = VideoFeatureExtractor(DEVICE)
    audio_extractor = AudioFeatureExtractor(DEVICE)

    # Checkpoint: randurile (indicii) deja scrise in .h5
    os.makedirs(config.FEATURES_DIR, exist_ok=True)
    checkpoint_file = str(config.FEATURES_DIR / "checkpoint.npy")
    if os.path.exists(checkpoint_file):
        processed = set(int(i) for i in np.load(checkpoint_file).tolist())
        print(f"Checkpoint gasit: {len(processed)} randuri deja procesate")
    else:
        processed = set()

    with h5py.File(OUTPUT_VIDEO, "a") as hf_video, \
         h5py.File(OUTPUT_AUDIO, "a") as hf_audio:

        if H5_KEY not in hf_video:
            hf_video.create_dataset(H5_KEY, shape=(n, 10, 7, 7, 512), dtype=np.float32)
        if H5_KEY not in hf_audio:
            hf_audio.create_dataset(H5_KEY, shape=(n, 10, 128), dtype=np.float32)

        first_row = {}  # video_id -> primul rand deja extras (pentru duplicate)
        for idx in sorted(processed):
            first_row.setdefault(video_ids[idx], idx)

        errors = []
        for idx, video_id in enumerate(tqdm(video_ids, desc="Extragere features")):
            if idx in processed:
                continue

            if video_id in first_row:
                # Acelasi videoclip -> aceleasi features, doar copiem randul
                src = first_row[video_id]
                hf_video[H5_KEY][idx] = hf_video[H5_KEY][src]
                hf_audio[H5_KEY][idx] = hf_audio[H5_KEY][src]
            else:
                video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
                if not os.path.exists(video_path):
                    errors.append((idx, video_id, "fisier lipsa"))
                    continue
                vf = video_extractor.extract(video_path)
                if vf is None:
                    errors.append((idx, video_id, "eroare video"))
                    continue
                af = audio_extractor.extract(video_path)
                if af is None:
                    errors.append((idx, video_id, "eroare audio"))
                    continue
                hf_video[H5_KEY][idx] = vf
                hf_audio[H5_KEY][idx] = af
                first_row[video_id] = idx

            processed.add(idx)
            if idx % 50 == 0:
                np.save(checkpoint_file, np.array(sorted(processed)))

        np.save(checkpoint_file, np.array(sorted(processed)))

    print("\n=== Rezultate ===")
    print(f"Randuri procesate: {len(processed)} / {n}")
    print(f"Erori: {len(errors)}")
    if errors:
        print("Primele 5 erori:")
        for e in errors[:5]:
            print(f"  idx={e[0]}, id={e[1]}, motiv={e[2]}")

    print("\nFeatures salvate in:")
    print(f"  {OUTPUT_VIDEO}")
    print(f"  {OUTPUT_AUDIO}")
