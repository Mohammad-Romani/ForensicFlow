import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np
import os
from utils import DeepFakeNPZDataset, set_seed
from model import MultiBranchVideoModel

def get_loader(fake_dir, real_dir, batch_size=4, num_workers=4, K=8, train=True, n_mels=64, mel_len=64):
    fake_files = sorted(glob.glob(os.path.join(fake_dir, "*.npz")))
    real_files = sorted(glob.glob(os.path.join(real_dir, "*.npz")))

    fake_ds = DeepFakeNPZDataset(files=fake_files, K=K, train=train, n_mels=n_mels, mel_len=mel_len)
    real_ds = DeepFakeNPZDataset(files=real_files, K=K, train=train, n_mels=n_mels, mel_len=mel_len)
    full_ds = ConcatDataset([fake_ds, real_ds])

    labels = np.array([1]*len(fake_files) + [0]*len(real_files))
    class_count = np.array([(labels==0).sum(), (labels==1).sum()])
    weight_per_class = 1.0 / (class_count + 1e-12)
    samples_weight = weight_per_class[labels]
    sampler = WeightedRandomSampler(torch.from_numpy(samples_weight).double(), num_samples=len(samples_weight), replacement=True) if train else None

    loader = DataLoader(full_ds, batch_size=batch_size, sampler=sampler, shuffle=(not train and False), num_workers=num_workers, pin_memory=True)
    return loader, int((labels==1).sum()), int((labels==0).sum())

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: logits (batch_size,)
        # targets: (batch_size,)
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
    
    # === Simple plotting helper ===
def plot_history(history, out_path=None):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.legend(); plt.title('Loss')
    plt.subplot(1,2,2)
    plt.plot(history['train_acc'], label='train_acc')
    plt.plot(history['val_acc'], label='val_acc')
    plt.plot(history['val_auc'], label='val_auc')
    plt.legend(); plt.title('Acc / AUC')
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
    plt.show()

def bootstrap_ci(y_true, y_score, metric_fn, n_boot=1000, seed=42):
    rng = np.random.RandomState(seed)
    stats = []
    n = len(y_true)
    if n == 0:
        return (None, None, None)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        try:
            stats.append(metric_fn(np.array(y_true)[idx], np.array(y_score)[idx]))
        except Exception:
            continue
    stats = np.array(stats)
    if len(stats) == 0:
        return (None, None, None)
    return float(stats.mean()), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))

# === Metrics safe wrappers (handle degenerate cases) ===

def safe_auc(y_true, y_score):
    try:
        if len(np.unique(y_true)) < 2:
            return float('nan')  # cannot compute AUC with single class
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float('nan')

def safe_f1(y_true, y_pred):
    try:
        return float(f1_score(y_true, y_pred))
    except Exception:
        return float('nan')
    
def train_model(model,
                train_fake_dir, train_real_dir,
                val_fake_dir, val_real_dir,
                epochs=16, lr=1.5e-5, batch_size=4, K=8, num_workers=4,
                device=None, seed=42, patience=4, use_reduce_on_plateau=True):
    # reproducibility
    set_seed(seed)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader, n_fake, n_real = get_loader(train_fake_dir, train_real_dir,
                                             batch_size=batch_size, num_workers=num_workers, K=K, train=True)
    val_loader, v_fake, v_real = get_loader(val_fake_dir, val_real_dir,
                                           batch_size=batch_size, num_workers=num_workers, K=K, train=False)
    print(f"Train → Fake: {n_fake}, Real: {n_real}\nVal → Fake: {v_fake}, Real: {v_real}")

    criterion = FocalLoss(alpha=1.0, gamma=2.0)

    # freeze backbones initially (as before)
    for p in model.rgb_back.parameters():
        p.requires_grad = False
    for p in model.tex_back.parameters():
        p.requires_grad = False
    if hasattr(model, 'freq_branch'):
        for p in model.freq_branch.parameters():
            p.requires_grad = False
    if hasattr(model, 'audio_branch'):
        for p in model.audio_branch.parameters():
            p.requires_grad = False

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    # scheduler: ReduceLROnPlateau is stable for training when using dynamic unfreeze
    if use_reduce_on_plateau:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    else:
        scheduler = None

    scaler = torch.amp.GradScaler('cuda')
    best_val_auc = -1.0
    no_improve = 0
    val_history = deque(maxlen=patience+2)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_auc": [], "val_f1": []}

    # unfreeze schedule (params you used)
    unfreeze_epoch_stage2 = 5
    unfreeze_epoch_stage1 = 8
    full_unfreeze_epoch = 11

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        # dynamic unfreeze (same logic as تو قبلا)
        if epoch == unfreeze_epoch_stage2:
            for name, param in model.rgb_back.named_parameters():
                if 'stages.3' in name or 'downsample_layers.3' in name:
                    param.requires_grad = True
            for name, param in model.tex_back.named_parameters():
                if 'layers.3' in name or 'blocks.3' in name:
                    param.requires_grad = True
            if hasattr(model, 'audio_branch'):
                for param in model.audio_branch.parameters():
                    param.requires_grad = True
            optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr/2, weight_decay=1e-4)
            print(f"✅ Unfroze last backbone blocks and audio_branch at epoch {epoch}")

        if epoch == unfreeze_epoch_stage1:
            for name, param in model.rgb_back.named_parameters():
                if 'stages.2' in name:
                    param.requires_grad = True
            for name, param in model.tex_back.named_parameters():
                if 'layers.2' in name:
                    param.requires_grad = True
            optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr/5, weight_decay=1e-4)
            print(f"✅ Unfroze mid backbone blocks at epoch {epoch}")

        if epoch == full_unfreeze_epoch:
            for p in model.rgb_back.parameters():
                p.requires_grad = True
            for p in model.tex_back.parameters():
                p.requires_grad = True
            optimizer = AdamW(model.parameters(), lr=lr/10, weight_decay=1e-4)
            print(f"✅ Fully unfroze backbone at epoch {epoch}")

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}] - Train", leave=False)
        for batch in loop:
            # flexible unpack: support (rgb,freq,label) or (rgb,freq,audio,label)
            if len(batch) == 3:
                rgb, freq, label = batch
                audio = None
            elif len(batch) == 4:
                rgb, freq, audio, label = batch
            else:
                # fallback: try first three
                rgb, freq, label = batch[0], batch[1], batch[2]
                audio = None

            rgb = rgb.to(device, non_blocking=True)
            freq = freq.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            if audio is not None:
                audio = audio.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast():
                try:
                    outputs = model(rgb, freq, audio) if audio is not None else model(rgb, freq)
                except TypeError:
                    # some model variants expect (rgb,freq) only
                    outputs = model(rgb, freq)

                if outputs.dim() > 1:
                    outputs = outputs.squeeze(1)
                if label.dim() == 2:
                    label = label.squeeze(1)
                loss = criterion(outputs, label)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * rgb.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            total_correct += (preds == label).sum().item()
            total_samples += label.size(0)
            loop.set_postfix(loss=total_loss/total_samples if total_samples>0 else 0.0,
                             acc=total_correct/total_samples if total_samples>0 else 0.0)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_probs = []
        val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    rgb, freq, label = batch
                    audio = None
                elif len(batch) == 4:
                    rgb, freq, audio, label = batch
                else:
                    rgb, freq, label = batch[0], batch[1], batch[2]
                    audio = None

                rgb = rgb.to(device, non_blocking=True)
                freq = freq.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)
                if audio is not None:
                    audio = audio.to(device, non_blocking=True)

                with autocast():
                    try:
                        outputs = model(rgb, freq, audio) if audio is not None else model(rgb, freq)
                    except TypeError:
                        outputs = model(rgb, freq)

                    if outputs.dim() > 1:
                        outputs = outputs.squeeze(1)
                    if label.dim() == 2:
                        label = label.squeeze(1)

                    loss = criterion(outputs, label)

                val_loss += loss.item() * rgb.size(0)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()

                val_probs.extend(probs.cpu().numpy())
                val_labels.extend(label.cpu().numpy())
                val_correct += (preds == label).sum().item()
                val_total += label.size(0)

        # compute epoch metrics
        val_loss = val_loss / val_total if val_total>0 else float('nan')
        val_acc = val_correct / val_total if val_total>0 else float('nan')
        val_auc = safe_auc(np.array(val_labels), np.array(val_probs))
        val_f1 = safe_f1(np.array(val_labels), (np.array(val_probs) > 0.5).astype(int))

        print(f"Epoch [{epoch+1}/{epochs}] → Train Loss: {total_loss/total_samples:.4f}, "
              f"Train Acc: {total_correct/total_samples:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
              f"Val AUC: {val_auc:.4f}, Val F1: {val_f1:.4f}")

        # record history
        history["train_loss"].append(total_loss/total_samples)
        history["train_acc"].append(total_correct/total_samples)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc if not math.isnan(val_auc) else 0.0)
        history["val_f1"].append(val_f1 if not math.isnan(val_f1) else 0.0)

        # Bootstrap CI (optional, inexpensive with n_boot=200)
        try:
            mean_auc, lo_auc, hi_auc = bootstrap_ci(np.array(val_labels), np.array(val_probs), roc_auc_score, n_boot=1000, seed=seed+epoch)
            mean_f1, lo_f1, hi_f1 = bootstrap_ci(np.array(val_labels), (np.array(val_probs) > 0.5).astype(int), f1_score, n_boot=1000, seed=seed+epoch)
            print(f" → AUC CI (95%): {mean_auc:.4f} [{lo_auc:.4f}, {hi_auc:.4f}]  |  F1 CI (95%): {mean_f1:.4f} [{lo_f1:.4f}, {hi_f1:.4f}]")
        except Exception:
            pass

        # Save best by val_auc
        if (not math.isnan(val_auc)) and (val_auc > best_val_auc):
            #os.makedirs("weights", exist_ok=True)
            best_val_auc = val_auc
            torch.save(model.state_dict(), "weights/best_model.pth")
            print("✅ Saved best model (AUC improved)")

        # Early stopping on AUC
        val_history.append(val_auc if not math.isnan(val_auc) else -1.0)
        if val_auc <= max(val_history):
            no_improve += 1
        else:
            no_improve = 0
       # if no_improve >= patience:
          #  print("⏱ Early stopping triggered (no AUC improvement)")
           # break

        # scheduler step
        if use_reduce_on_plateau and scheduler is not None:
            # use val_loss (you may choose val_auc instead)
            try:
                scheduler.step(val_loss)
            except Exception:
                pass

        # free memory
        gc.collect()
        torch.cuda.empty_cache()

    # final plotting / return
    try:
        plot_history(history, out_path="training_history.png")
    except Exception:
        pass

    return history

# --------------------------- train the model ---------------------------
fake_train = "npz/fake_train"
real_train = "npz/real_train"
fake_val = "npz/fake_val"
real_val = "npz/real_val"

model = MultiBranchVideoModel(embed_dim=512, use_freq=True, use_audio=False, temporal='att', pretrained=True)
history = train_model(model, fake_train, real_train, fake_val, real_val, epochs=16,
                      lr=2e-5, batch_size=4, K=8, 
                      num_workers=4, seed=42)
