import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from einops import rearrange



"""Multi-Branch Fusion (modern) for video-level DeepFake detection.
- Backbones via timm (convnext_tiny, swin_tiny_patch4_window7_224 by default)
- Frequency branch: deeper CNN + SE-like channel attention
- Temporal pooling: mean / max / attention
- Fusion: attention MLP that learns weights for branches
- Returns logits (use BCEWithLogitsLoss)
"""

# -------------------- Utility: create backbone + proj --------------------
def make_backbone_with_proj(model_name='convnext_tiny', embed_dim=512, pretrained=True):
    """
    Create timm backbone that returns pooled features (num_classes=0, global_pool='avg').
    Returns: backbone (nn.Module), proj (nn.Module for mapping feat_dim -> embed_dim)
    """
    backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='avg')
    feat_dim = backbone.num_features
    proj = nn.Sequential(
        nn.Linear(feat_dim, embed_dim),
        nn.BatchNorm1d(embed_dim),
        nn.ReLU(inplace=True)
    )
    return backbone, proj


# -------------------- Frequency branch (improved) --------------------
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)


class FrequencyBranch(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.se = SEBlock(128, reduction=8)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x: (B,3,H,W)
        h = self.conv(x)             # (B,128,1,1)
        h = self.se(h)               # channel attention
        h = h.view(h.size(0), -1)    # (B,128)
        h = self.fc(h)               # (B,out_dim)
        return h


# -------------------- Temporal pooling --------------------
class TemporalPool(nn.Module):
    def __init__(self, method='att', embed_dim=512):
        super().__init__()
        assert method in ('mean', 'max', 'att'), "temporal must be 'mean'|'max'|'att'"
        self.method = method
        if method == 'att':
            self.att = nn.Sequential(
                nn.Linear(embed_dim, max(32, embed_dim // 8)),
                nn.ReLU(inplace=True),
                nn.Linear(max(32, embed_dim // 8), 1)
            )

    def forward(self, feats):
        # feats: (B, K, D)
        if self.method == 'mean':
            return feats.mean(dim=1)
        elif self.method == 'max':
            return feats.max(dim=1)[0]
        else:
            B, K, D = feats.shape
            scores = self.att(feats.view(B * K, D)).view(B, K)  # (B, K)
            weights = torch.softmax(scores, dim=1).unsqueeze(2)  # (B,K,1)
            fused = (feats * weights).sum(dim=1)
            return fused


# -------------------- Fusion attention (branch weighting) --------------------
class AttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_branches=3, hidden=128):
        super().__init__()
        self.num_branches = num_branches
        self.att_mlp = nn.Sequential(
            nn.Linear(embed_dim * num_branches, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_branches)
        )

    def forward(self, feats):
        # feats: list of (B, embed_dim)
        x = torch.cat(feats, dim=1)                # (B, embed_dim * num_branches)
        scores = self.att_mlp(x)                   # (B, num_branches)
        weights = torch.softmax(scores, dim=1)     # (B, num_branches)
        fused = 0
        for i, f in enumerate(feats):
            w = weights[:, i].unsqueeze(1)        # (B,1)
            fused = fused + f * w
        return fused

# ----------------------------
# Audio branch (embed audio -> same embed_dim)
# ----------------------------

class AudioBranch(nn.Module):
    def __init__(self, in_ch=1, embed_dim=512, n_mfcc=40, audio_len=128):
        super().__init__()
        # simple conv stack for spectrogram-like input (1, n_mfcc, audio_len)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # -> (B,128,1,1)
            nn.Flatten(),             # -> (B,128)
            nn.Linear(128, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x: (B, 1, n_mfcc, audio_len)
        return self.net(x)

# ----------------------------
# Modify MultiBranchVideoModel to accept audio
# ----------------------------
# (Insert after FrequencyBranch, TemporalPool, AttentionFusion definitions)
class MultiBranchVideoModel(nn.Module):
    def __init__(self,
                 backbone_rgb='convnext_tiny',
                 backbone_texture='swin_tiny_patch4_window7_224',
                 embed_dim=512,
                 use_freq=True,
                 use_audio=False,
                 temporal='att',
                 pretrained=True,
                 freeze_backbones=False):
        super().__init__()
        self.embed_dim = embed_dim  
        # backbones + proj
        self.rgb_back, self.rgb_proj = make_backbone_with_proj(backbone_rgb, embed_dim, pretrained=pretrained)
        self.tex_back, self.tex_proj = make_backbone_with_proj(backbone_texture, embed_dim, pretrained=pretrained)

        # enable gradient checkpointing where available
        try:
            if hasattr(self.rgb_back, 'gradient_checkpointing_enable'):
                self.rgb_back.gradient_checkpointing_enable()
            elif hasattr(self.rgb_back, 'grad_checkpointing_enable'):
                self.rgb_back.grad_checkpointing_enable()
            if hasattr(self.tex_back, 'gradient_checkpointing_enable'):
                self.tex_back.gradient_checkpointing_enable()
            elif hasattr(self.tex_back, 'grad_checkpointing_enable'):
                self.tex_back.grad_checkpointing_enable()
            print("✅ Gradient checkpointing enabled for backbones")
        except Exception as e:
            print("⚠️ Could not enable gradient checkpointing:", e)

        self.use_freq = use_freq
        self.use_audio = use_audio
        if use_freq:
            self.freq_branch = FrequencyBranch(out_dim=embed_dim)
        if use_audio:
            self.audio_branch = AudioBranch(in_ch=1, embed_dim=embed_dim)

        if freeze_backbones:
            for p in self.rgb_back.parameters():
                p.requires_grad = False
            for p in self.tex_back.parameters():
                p.requires_grad = False

        self.temporal_pool = TemporalPool(method=temporal, embed_dim=embed_dim)
        num_br = 2 + (1 if use_freq else 0) + (1 if use_audio else 0)
        self.fusion = AttentionFusion(embed_dim=embed_dim, num_branches=num_br)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, x_rgb, x_freq=None, x_audio=None):
        B, K, C, H, W = x_rgb.shape
        x_flat = x_rgb.view(B * K, C, H, W)

        r_feat = self.rgb_back(x_flat)
        r_emb = self.rgb_proj(r_feat)

        t_feat = self.tex_back(x_flat)
        t_emb = self.tex_proj(t_feat)

        r_emb = r_emb.view(B, K, -1)
        t_emb = t_emb.view(B, K, -1)

        r_p = self.temporal_pool(r_emb)
        t_p = self.temporal_pool(t_emb)
        feats = [r_p, t_p]

        if self.use_freq:
            if x_freq is None:
                raise ValueError("x_freq required when use_freq=True")
            xf_flat = x_freq.view(B * K, C, H, W)
            f_emb = self.freq_branch(xf_flat)
            f_emb = f_emb.view(B, K, -1)
            f_p = self.temporal_pool(f_emb)
            feats.append(f_p)

        if self.use_audio:
            if x_audio is None:
                
                a_emb = torch.zeros((B, self.embed_dim), device=x_rgb.device)
            else:
                a_emb = self.audio_branch(x_audio)  # (B, embed_dim)
            feats.append(a_emb)

        fused = self.fusion(feats)
        logits = self.classifier(fused).squeeze(1)
        return logits

# -------------------- Quick sanity test --------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiBranchVideoModel(use_audio=True, pretrained=False).to(device)
    rgb = torch.randn(2, 8, 3, 224, 224).to(device)
    freq = torch.randn(2, 8, 3, 224, 224).to(device)
    # بدون audio (x_audio=None)
    out = model(rgb, freq)
    print("out.shape:", out.shape)  # shoukd be (,2)
# ======================================================
# 🔹 DeepFake Detection - MultiBranch Video Model (Full Training)
# ======================================================
