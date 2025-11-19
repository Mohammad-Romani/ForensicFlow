import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from model import MultiBranchVideoModel

# ---- helper functions (as before, but with FFT freq builder) ----
def find_last_conv_module(module):
    last = None
    for _, m in module.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    return last

def normalize_cam(cam):
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam

def preprocess_rgb_tensor(frame_bgr, device):
    # returns tensor shape (1,1,3,224,224)
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224,224)).astype(np.float32) / 255.0
    t = torch.from_numpy(img.transpose(2,0,1)).unsqueeze(0).unsqueeze(0)  # (1,1,3,H,W)
    mean = torch.tensor([0.485,0.456,0.406]).view(1,1,3,1,1)
    std  = torch.tensor([0.229,0.224,0.225]).view(1,1,3,1,1)
    t = (t - mean) / std
    return t.to(next(iter(model.parameters())).device)

def build_freq_from_rgb(frame_bgr):
    # replicate your fft_map_fast logic -> make 3-channel magnitude map
    # input: BGR HxWx3 uint8
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    gray = 0.299*rgb[...,0] + 0.587*rgb[...,1] + 0.114*rgb[...,2]
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.log1p(np.abs(fft_shift)).astype(np.float32)
    # normalize magnitude to 0..255 roughly (for visual/consistency)
    mag = magnitude - magnitude.min()
    if mag.max() > 0:
        mag = mag / mag.max()
    mag = (mag * 255.0).astype(np.uint8)
    mag3 = np.stack([mag,mag,mag], axis=-1)  # H,W,3
    return mag3

def preprocess_freq_tensor(frame_bgr, device):
    mag3 = build_freq_from_rgb(frame_bgr)
    mag3 = cv2.resize(mag3, (224,224)).astype(np.float32) / 255.0
    t = torch.from_numpy(mag3.transpose(2,0,1)).unsqueeze(0).unsqueeze(0)  # (1,1,3,H,W)
    mean = torch.tensor([0.485,0.456,0.406]).view(1,1,3,1,1)
    std  = torch.tensor([0.229,0.224,0.225]).view(1,1,3,1,1)
    t = (t - mean) / std
    return t.to(next(iter(model.parameters())).device)


def show_and_save_overlay(orig_bgr, cam_mask, out_path="gradcam_overlay.png", alpha=0.5):
    """
        Overlays the Grad-CAM heatmap on the original image and saves the result.
    
    Args:
        orig_bgr (np.array): Original image in BGR format (e.g., from cv2.imread).
        cam_mask (np.array): Grayscale Grad-CAM mask with shape (H, W) and values in [0, 1].
        out_path (str): Path to save the output image.
        alpha (float): Transparency factor for the heatmap overlay (0 = image only, 1 = heatmap only).
    """
    
    H_orig, W_orig, C_orig = orig_bgr.shape
    
    # 1. Convert original image to RGB and normalize to [0, 1]
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # 2. Convert grayscale CAM mask to colored heatmap using JET colormap
    heatmap_jet = cv2.applyColorMap((cam_mask * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # 3. Critical fix: Resize heatmap to match the original image dimensions
    # Using INTER_LINEAR for smooth interpolation
    heatmap_resized = cv2.resize(heatmap_jet, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
    
    # 4. Normalize heatmap and convert from BGR to RGB
    heatmap_rgb = heatmap_resized[..., ::-1] / 255.0 # BGR to RGB

    # 5. Blend the original image and heatmap using alpha blending
    # Both arrays now have shape (H_orig, W_orig, 3)
    overlay = (1 - alpha) * orig_rgb + alpha * heatmap_rgb
    
    # 6. Clip values to valid range and convert to 8-bit for saving/display
    overlay = np.clip(overlay, 0, 1)
    overlay_bgr_8bit = (overlay[..., ::-1] * 255).astype(np.uint8)
    
    # Save the result
    cv2.imwrite(out_path, overlay_bgr_8bit)
    
    plt.figure(figsize=(6,6))
    plt.imshow(overlay)
    plt.title(f"Grad-CAM Overlay (Alpha={alpha})")
    plt.axis('off')
    plt.show()

# ----------------------------------------------------------------------------------

def generate_gradcam_frame_with_freq(model, frame_bgr, target_module=None, device=None, class_idx=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # prepare inputs
    rgb_t = preprocess_rgb_tensor(frame_bgr, device=device)   # (1,1,3,224,224)
    freq_t = preprocess_freq_tensor(frame_bgr, device=device) # (1,1,3,224,224)

    # if model expects shape (B,K,3,H,W) where B=1,K=1 -> our tensors are ready
    # choose target module automatically if not provided
    if target_module is None:
        target_module = find_last_conv_module(model.rgb_back)
        if target_module is None:
            raise RuntimeError("Could not find target conv in rgb_back; pass target_module explicitly.")

    activations = {}
    gradients = {}

    def fhook(module, inp, out):
        activations['value'] = out

    # use full backward hook if available to avoid FutureWarning
    if hasattr(target_module, "register_full_backward_hook"):
        bhook = lambda module, grad_in, grad_out: gradients.__setitem__('value', grad_out[0])
        fh = target_module.register_forward_hook(fhook)
        bh = target_module.register_full_backward_hook(bhook)
    else:
        def bhook_normal(module, grad_in, grad_out):
            gradients['value'] = grad_out[0]
        fh = target_module.register_forward_hook(fhook)
        bh = target_module.register_backward_hook(bhook_normal)

    # forward (model may require rgb,freq,audio). we pass freq_t; audio=None
    # ensure requires_grad on inputs (not strictly necessary)
    rgb_t.requires_grad_(True)
    freq_t.requires_grad_(True)

    # model expects (B,K,3,H,W) -> we have that shape
    # call with both rgb and freq
    outputs = None
    try:
        outputs = model(rgb_t, freq_t, None)   # prefer explicit signature
    except Exception:
        try:
            outputs = model(rgb_t, freq_t)
        except Exception as e:
            # last fallback: try single-arg
            outputs = model(rgb_t)

    # ensure scalar logit
    if outputs.dim() > 1:
        outputs = outputs.squeeze(1)
    if class_idx is None:
        score = outputs[0]
    else:
        score = outputs[0] if outputs.dim()==0 else outputs[0, class_idx]

    model.zero_grad()
    score.backward(retain_graph=True)

    act = activations.get('value')   # tensor
    grad = gradients.get('value')    # tensor

    # remove hooks
    fh.remove(); bh.remove()

    if act is None or grad is None:
        raise RuntimeError("Couldn't capture activations or gradients for target layer.")

    # weights: global avg pool of grad over spatial dims
    weights = grad.mean(dim=(2,3), keepdim=True)  # (B*K, C,1,1)
    cam = (weights * act).sum(dim=1)  # (B*K, Hf, Wf)
    cam = F.relu(cam)
    cam = cam.squeeze(0).detach().cpu().numpy()  # take first sample
    # resize to 224
    cam_resized = cv2.resize(cam, (224,224))
    cam_resized = normalize_cam(cam_resized)
    return cam_resized

# --------- usage example ----------
if __name__ == "__main__":
    model_path = "/weights/best_model.pth"   # change to your model path
    frame_path = ""       # path to frame (or extract one from a video)
    out_path = "/kaggle/working/gradcam_overlay.jpg"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiBranchVideoModel(embed_dim=512, use_freq=True, use_audio=False, temporal='att', pretrained=False)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    img_bgr = cv2.imread(frame_path)
    if img_bgr is None:
        raise FileNotFoundError("Frame not found: " + frame_path)

    target = find_last_conv_module(model.rgb_back)
    print("Using target module:", target)

    cam_mask = generate_gradcam_frame_with_freq(model, img_bgr, target_module=target, device=device)
    show_and_save_overlay(img_bgr, cam_mask, out_path=out_path, alpha=0.5)
    print("Saved overlay:", out_path)
