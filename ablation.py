import os
import torch
import pandas as pd
import argparse
from model import MultiBranchVideoModel
from train import train_model

fake_train = "npz/fake_train"
real_train = "npz/real_train"
fake_val = "npz/fake_val"
real_val = "npz/real_val"

# ------------------------------------------------------------------
# Backup the original best model (optional but recommended)
# ------------------------------------------------------------------
try:
    shutil.copy(
        "/weights/best_model.pth",
        "/ur path/deepfake_best_model_backup.pth",
    )
    print("Backup created: deepfake_best_model_backup.pth")
except Exception:
    pass

# ------------------------------------------------------------------
# Configurations for the three ablation variants
# ------------------------------------------------------------------
configs = [
    {"name": "RGB-only",      "use_freq": False, "temporal": "mean"},
    {"name": "RGB+Freq",      "use_freq": True,  "temporal": "mean"},
    {"name": "RGB+Freq+Att",  "use_freq": True,  "temporal": "att"},
]

results = []

# ------------------------------------------------------------------
# Run each variant
# ------------------------------------------------------------------
for i, cfg in enumerate(configs):
    print(f"\nRunning step {i+1}/{len(configs)}: {cfg['name']}")

    # --------------------------------------------------------------
    # Step 1: start from pretrained weights (only for the first run)
    # --------------------------------------------------------------
    checkpoint_path = f"/weights/{cfg['name']}_checkpoint.pth"

    if i == 0 or not os.path.exists(checkpoint_path):
        print("Starting from pretrained weights...")
        model = MultiBranchVideoModel(
            embed_dim=512,
            use_freq=cfg["use_freq"],
            use_audio=False,
            temporal=cfg["temporal"],
            pretrained=True,          # only for the first variant
        ).to("cuda")
    else:
        # ----------------------------------------------------------
        # Subsequent steps: resume from the saved checkpoint
        # ----------------------------------------------------------
        print("Resuming from checkpoint...")
        model = MultiBranchVideoModel(
            embed_dim=512,
            use_freq=cfg["use_freq"],
            use_audio=False,
            temporal=cfg["temporal"],
            pretrained=False,         # important!
        ).to("cuda")
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Weights loaded from: {checkpoint_path}")

    # --------------------------------------------------------------
    # Train for 5 epochs
    # --------------------------------------------------------------
    hist = train_model(
        model=model,
        train_fake_dir=fake_train,
        train_real_dir=real_train,
        val_fake_dir=fake_val,
        val_real_dir=real_val,
        epochs=5,
        lr=2e-5,
        batch_size=4,
        K=8,
        seed=42,
    )

    # --------------------------------------------------------------
    # Save checkpoint for the next ablation step
    # --------------------------------------------------------------
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Checkpoint saved: {cfg['name']}_checkpoint.pth")

    # --------------------------------------------------------------
    # Record the best metric (AUC) over the 5 epochs
    # --------------------------------------------------------------
    best_auc = max(hist["val_auc"])
    best_idx = hist["val_auc"].index(best_auc)

    results.append(
        {
            "Model": cfg["name"],
            "Best AUC": f"{best_auc:.4f}",
            "Best F1": f"{hist['val_f1'][best_idx]:.4f}",
            "Epoch": best_idx + 1,
        }
    )

# ------------------------------------------------------------------
# Final table
# ------------------------------------------------------------------
df = pd.DataFrame(results)
display(df.style.background_gradient(cmap="Greens", subset=["Best AUC"]))

# ------------------------------------------------------------------
# Export LaTeX and CSV (ready for the paper)
# ------------------------------------------------------------------
df.to_csv("/ablation_final.csv", index=False)
print("\nLaTeX table:\n")
print(df.to_latex(index=False))
print("\nCSV saved to: /ablation_final.csv")