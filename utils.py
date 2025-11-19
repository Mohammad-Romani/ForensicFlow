import os
import cv2
import torch
import numpy as np
import argparse
import glob
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import librosa
import torch.nn.functional as F
from torch.utils.data import Dataset
import random


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


base_real = "celeb-df-v2/Celeb-real"
base_fake = "celeb-df-v2/Celeb-synthesis"
base_ytreal = "celeb-df-v2/YouTube-real"
test_list_path = "celeb-df-v2/List_of_testing_videos.txt"

out_base = "ur path/ for save npz files"
os.makedirs(out_base, exist_ok=True)

# ---------------------------
# train/val
# ---------------------------
test_videos = set()
with open(test_list_path, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            rel_path = parts[1].strip()
            test_videos.add(rel_path)
print(f"✅ Total train/val videos: {len(test_videos)}")


def sample_frames_from_video(video_path, K=8):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count == 0:
        cap.release()
        return []
    indices = np.linspace(0, frame_count - 1, K).astype(int)
    frames = []
    idx_set = set(indices.tolist())
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        if i in idx_set:
            frames.append(cv2.resize(frame, (224, 224)))
    cap.release()
    return frames

def fft_map_fast(x):
    x_gray = np.dot(x[..., :3], [0.299, 0.587, 0.114])
    fft = np.fft.fft2(x_gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.log1p(np.abs(fft_shift))
    return np.stack([magnitude]*3, axis=-1).astype(np.float16)

def extract_audio_features(video_path, sr=16000, n_mels=64):
    # Detect and extract audio if present; otherwise return a zero mel-spectrogram.
    # This enables robust tri-modal operation while supporting audio-free videos.
    try:
        probe = subprocess.run(
            ["ffprobe", "-i", video_path, "-show_streams", "-select_streams", "a",
             "-loglevel", "error"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if len(probe.stdout) == 0:
            return np.zeros((n_mels, n_mels), dtype=np.float16)

        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        command = [
            "ffmpeg", "-i", video_path,
            "-ar", str(sr), "-ac", "1", "-f", "wav",
            "-loglevel", "quiet", tmp_wav.name
        ]
        subprocess.run(command, check=True)

        y, sr = librosa.load(tmp_wav.name, sr=sr)
        os.remove(tmp_wav.name)
        if len(y) == 0:
            return np.zeros((n_mels, n_mels), dtype=np.float16)

        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return mel_db.astype(np.float16)

    except Exception as e:
        return np.zeros((n_mels, n_mels), dtype=np.float16)


def extract_audio_mel_from_video(video_path, sr=16000, n_mels=64, mel_len=64):
    """
    Extract audio via ffmpeg (mono sr) and compute mel-spectrogram (n_mels x mel_len).
    Returns float32 array shape (n_mels, mel_len).
    """
    try:
        # check audio stream existence
        probe = subprocess.run(["ffprobe","-i",video_path,"-show_streams","-select_streams","a","-loglevel","error"],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if len(probe.stdout) == 0:
            return np.zeros((n_mels, mel_len), dtype=np.float32)

        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        cmd = ["ffmpeg","-y","-i", video_path, "-ar", str(sr), "-ac", "1", "-vn", "-loglevel","error", tmp_wav.name]
        subprocess.run(cmd, check=True)

        y, _ = librosa.load(tmp_wav.name, sr=sr)
        os.remove(tmp_wav.name)
        if y.size == 0:
            return np.zeros((n_mels, mel_len), dtype=np.float32)

        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        # fix length (pad or trim time axis)
        if mel_db.shape[1] < mel_len:
            pad_width = mel_len - mel_db.shape[1]
            mel_db = np.pad(mel_db, ((0,0),(0,pad_width)), mode='constant', constant_values=mel_db.min())
        else:
            mel_db = mel_db[:, :mel_len]
        return mel_db.astype(np.float32)
    except Exception:
        return np.zeros((n_mels, mel_len), dtype=np.float32)


def process_video(args):
    video_path, out_path, K, split, cls = args
    if os.path.exists(out_path):
        return
    frames = sample_frames_from_video(video_path, K=K)
    if len(frames) == 0:
        return
    if len(frames) < K:
        frames = frames + [frames[-1]] * (K - len(frames))

    freq_maps = [fft_map_fast(f) for f in frames]  # K entries

    # extract mel (n_mels x mel_len)
    audio_mel = extract_audio_mel_from_video(video_path, sr=16000, n_mels=64, mel_len=64)

    np.savez_compressed(out_path,
                        rgb=np.array(frames, dtype=np.uint8),
                        freq=np.array(freq_maps, dtype=np.float32),
                        audio=audio_mel)

# ---------------------------
# train/val
# ---------------------------
def prepare_jobs(input_folder, cls, prefix, output_folder, test_videos, K=8):
    jobs = []
    videos = [f for f in os.listdir(input_folder) if f.endswith(".mp4")]
    for vid in videos:
        rel_path = f"{prefix}/{vid}"
        split = "val" if rel_path in test_videos else "train"
        out_dir = os.path.join(output_folder, split, cls)
        os.makedirs(out_dir, exist_ok=True)
        in_path = os.path.join(input_folder, vid)
        out_path = os.path.join(out_dir, vid.replace(".mp4", ".npz"))
        jobs.append((in_path, out_path, K, split, cls))
    return jobs


all_jobs = []
all_jobs += prepare_jobs(base_real, "real", "Celeb-real", out_base, test_videos)
all_jobs += prepare_jobs(base_ytreal, "real", "YouTube-real", out_base, test_videos)
all_jobs += prepare_jobs(base_fake, "fake", "Celeb-synthesis", out_base, test_videos)

print(f" Total videos to process: {len(all_jobs)}")

with Pool(processes=cpu_count()) as p:
    list(tqdm(p.imap_unordered(process_video, all_jobs), total=len(all_jobs)))

gc.collect()
print(" Done! Saved to:", out_base)


class DeepFakeNPZDataset(Dataset):
    def __init__(self, npz_folder=None, files=None, K=8, train=True, n_mels=64, mel_len=64, augment_prob=0.3):
        if files is not None:
            self.files = sorted(files)
        else:
            assert npz_folder is not None
            self.files = sorted([os.path.join(npz_folder, f) for f in os.listdir(npz_folder) if f.endswith('.npz')])

        self.K = K
        self.train = train
        self.n_mels = n_mels
        self.mel_len = mel_len
        self.augment_prob = augment_prob

        # transforms (applied deterministically per video using seed)
        self.simple = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
        self.strong = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.85,1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.12,0.12,0.12,0.02)
        ])
        self.imagenet_norm = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        data = np.load(path)
        rgb = data['rgb']            # (K,H,W,C)
        freq = data['freq']          # (K,H,W,C)
        audio = data['audio']        # (n_mels, mel_len) float32

        # ensure shapes
        def pad_to_k(a):
            if a.shape[0] >= self.K:
                return a[:self.K]
            pad_len = self.K - a.shape[0]
            return np.pad(a, ((0,pad_len),(0,0),(0,0),(0,0)), mode='edge')
        rgb = pad_to_k(rgb)
        freq = pad_to_k(freq)

        # augmentation deterministic: sample seed per video
        apply_strong = self.train and (np.random.rand() < self.augment_prob)
        transform = self.strong if apply_strong else self.simple
        seed = np.random.randint(2**31 - 1)

        out_rgb, out_freq = [], []
        for t in range(self.K):
            img_rgb = Image.fromarray(rgb[t].astype('uint8'))
            img_freq = Image.fromarray(freq[t].clip(0,255).astype('uint8'))

            random.seed(seed); torch.manual_seed(seed)
            img_rgb = transform(img_rgb)
            random.seed(seed); torch.manual_seed(seed)
            img_freq = transform(img_freq)

            out_rgb.append(np.array(img_rgb))
            out_freq.append(np.array(img_freq))

        # to tensors (T,C,H,W)
        rgb_t = torch.from_numpy(np.stack(out_rgb)).permute(0,3,1,2).float() / 255.0
        freq_t = torch.from_numpy(np.stack(out_freq)).permute(0,3,1,2).float() / 255.0

        # normalize per-frame
        rgb_t = self.imagenet_norm(rgb_t)
        freq_t = self.imagenet_norm(freq_t)

        # audio -> convert to tensor and add channel dim: (1, n_mels, mel_len)
        audio_t = torch.from_numpy(audio).unsqueeze(0).float()

        # Zero-mean unit-variance normalization for the audio branch
        # This stabilizes training and ensures the audio features are on the same scale
        # as the RGB and frequency branches (which use ImageNet normalization).
        audio_t = (audio_t - audio_t.mean()) / (audio_t.std() + 1e-8)  # prevent division by zero

        # label
        label = 1.0 if os.path.sep + "fake" + os.path.sep in path.lower() else 0.0
        label_t = torch.tensor(label, dtype=torch.float32)

        return rgb_t, freq_t, audio_t, label_t