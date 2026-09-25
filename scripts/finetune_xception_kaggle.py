# %% [markdown]
# # 🚀 Xception71 Fine-Tuning for DeepFake Detection (Kaggle Notebook)
# 
# **Purpose:** Fine-tune the pre-trained Xception71 checkpoint (`best_xception71.pth`) on your dataset to eliminate false positives on real images and reliably detect deepfakes.
# 
# ---
# 
# ## 📋 Setup Instructions (Kaggle):
# 1. **Notebook Accelerator:** Settings (Right Sidebar) → Accelerator → **GPU T4 x2** (or GPU P100)
# 2. **Upload Datasets (Right Sidebar → Add Input):**
#    - **Input 1 (Images):** The same `finetune-data` dataset you already created with `train` and `val` folders:
#      ```
#      finetune-data/
#      ├── train/
#      │   ├── real/
#      │   └── fake/
#      └── val/
#          ├── real/
#          └── fake/
#      ```
#    - **Input 2 (Xception Model):** Upload your local `best_xception71.pth` (from `models/results(Xception)/best_xception71.pth`) as a new dataset named `xception-checkpoint`.
# 
# 3. **Run All Cells:** Training takes about 4 to 8 minutes on GPU.
# 4. **Download:** Download `best_xception71_finetuned.pth` from the Output folder `/kaggle/working`.

# %% [markdown]
# ## Cell 1: Install Dependencies

# %%
!pip install timm --quiet

# %% [markdown]
# ## Cell 2: Configuration & Paths

# %%
import os
import copy
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION — Paths on Kaggle
# ============================================================

# Search for the dataset directory automatically
POSSIBLE_DATA_DIRS = [
    Path("/kaggle/input/finetune-data"),
    Path("/kaggle/input/real-and-fake-dataset"),
    Path("/kaggle/input/deepfake-dataset"),
]
DATA_DIR = None
for p in POSSIBLE_DATA_DIRS:
    if p.exists():
        DATA_DIR = p
        break

if DATA_DIR is None:
    # Look for any folder in /kaggle/input that has a train directory
    for root, dirs, _ in os.walk("/kaggle/input"):
        if "train" in dirs:
            DATA_DIR = Path(root)
            break

print(f"Dataset path: {DATA_DIR}")

# Checkpoint path — search automatically in /kaggle/input
POSSIBLE_CKPTS = [
    Path("/kaggle/input/xception-checkpoint/best_xception71.pth"),
    Path("/kaggle/input/best-xception71/best_xception71.pth"),
]
CHECKPOINT_PATH = None
for cp in POSSIBLE_CKPTS:
    if cp.exists():
        CHECKPOINT_PATH = cp
        break

if CHECKPOINT_PATH is None:
    # Search for any .pth file with xception in name
    for root, _, files in os.walk("/kaggle/input"):
        for f in files:
            if "xception" in f.lower() and f.endswith(".pth"):
                CHECKPOINT_PATH = Path(root) / f
                break

print(f"Checkpoint path: {CHECKPOINT_PATH}")

# Output path
OUTPUT_DIR = Path("/kaggle/working")
OUTPUT_PATH = OUTPUT_DIR / "best_xception71_finetuned.pth"

# Training Hyperparameters
BATCH_SIZE = 16          # Xception 299x299 fits comfortably in VRAM
NUM_EPOCHS = 25          # Maximum epochs (early stopping will trigger earlier)
LEARNING_RATE = 5e-5     # Safe learning rate for fine-tuning
WEIGHT_DECAY = 1e-4      # L2 Regularization
PATIENCE = 6             # Early stopping patience
INPUT_SIZE = 299         # Xception71 native resolution
NUM_WORKERS = 2

# Xception71 Normalization: (x - 0.5) / 0.5 -> maps [0, 1] to [-1, 1]
XCEPTION_MEAN = [0.5, 0.5, 0.5]
XCEPTION_STD = [0.5, 0.5, 0.5]

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# %% [markdown]
# ## Cell 3: Verify Dataset & Class Balance

# %%
print("=" * 55)
print("Data Verification")
print("=" * 55)

assert DATA_DIR is not None and DATA_DIR.exists(), "ERROR: Dataset not found in /kaggle/input! Please upload your dataset."
assert CHECKPOINT_PATH is not None and CHECKPOINT_PATH.exists(), f"ERROR: Checkpoint {CHECKPOINT_PATH} not found in /kaggle/input!"

for split in ["train", "val"]:
    for cls in ["real", "fake"]:
        folder = DATA_DIR / split / cls
        if not folder.exists():
            alt = DATA_DIR / split / ("0_real" if cls == "real" else "1_fake")
            if alt.exists():
                folder = alt
        if folder.exists():
            count = len(list(folder.glob("*.*")))
            print(f"  [{split:5s}] {cls:4s}: {count:4d} images")
        else:
            print(f"  WARNING: Missing folder {folder}")

# %% [markdown]
# ## Cell 4: Load Model & Strict Layer Freezing

# %%
import timm

def create_model(checkpoint_path, device):
    """Load pre-trained Xception71 and freeze early feature extraction layers."""
    print("Loading pre-trained Xception71 architecture...")
    model = timm.create_model("xception71", pretrained=False, num_classes=2)
    
    # Load weights
    print(f"Loading weights from {checkpoint_path}...")
    raw = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    sd = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
    model.load_state_dict(sd, strict=True)
    print("  ✓ Checkpoint loaded successfully!")
    
    # ============================================================
    # LAYER FREEZE STRATEGY:
    # 1. Freeze stem and blocks 0 to 21 (preserves low-level edge/texture filters)
    # 2. Unfreeze block 22 (last residual block) + classification head
    # ============================================================
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze head (Linear classifier: 2048 -> 2)
    for param in model.head.parameters():
        param.requires_grad = True
        
    # Unfreeze the last XceptionModule block
    for param in model.blocks[-1].parameters():
        param.requires_grad = True
        
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ Parameters: {total:,} total, {trainable:,} trainable ({trainable/total*100:.1f}%)")
    print(f"  ✓ Frozen: {(total - trainable)/total*100:.1f}%")
    
    return model.to(device)

model = create_model(CHECKPOINT_PATH, device)

# %% [markdown]
# ## Cell 5: Data Augmentation & DataLoader Setup

# %%
# Train: Augmentation to generalize against compression and lighting
train_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE + 24, INPUT_SIZE + 24)),
    transforms.RandomCrop(INPUT_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05),
    transforms.RandomGrayscale(p=0.05),
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
    ], p=0.25),
    transforms.ToTensor(),
    transforms.Normalize(mean=XCEPTION_MEAN, std=XCEPTION_STD),
])

# Validation: Clean deterministic transform
val_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=XCEPTION_MEAN, std=XCEPTION_STD),
])

class DeepfakeDataset(datasets.ImageFolder):
    """Enforce exact class mapping: Real=0, Fake=1."""
    def find_classes(self, directory):
        classes, _ = super().find_classes(directory)
        class_to_idx = {}
        for c in classes:
            c_low = c.lower()
            if "fake" in c_low:
                class_to_idx[c] = 1
            elif "real" in c_low:
                class_to_idx[c] = 0
            else:
                raise ValueError(f"Unknown class folder '{c}'. Must contain 'real' or 'fake'.")
        return classes, class_to_idx

train_dataset = DeepfakeDataset(DATA_DIR / "train", transform=train_transform)
val_dataset = DeepfakeDataset(DATA_DIR / "val", transform=val_transform)

print(f"Classes mapping: {train_dataset.class_to_idx}")
print(f"Train samples:   {len(train_dataset)}")
print(f"Val samples:     {len(val_dataset)}")

# Verify: Real=0, Fake=1
for cls_name, idx in train_dataset.class_to_idx.items():
    if "real" in cls_name.lower():
        assert idx == 0, f"ERROR: '{cls_name}' must be class 0"
    elif "fake" in cls_name.lower():
        assert idx == 1, f"ERROR: '{cls_name}' must be class 1"
print("  ✓ Class mapping verified: Real=0, Fake=1")

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True, drop_last=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True
)

# %% [markdown]
# ## Cell 6: Training with Automatic Early Stopping

# %%
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)

# Mixed precision (FP16)
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
torch.cuda.empty_cache()

history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
best_val_acc = 0.0
best_val_loss = float("inf")
best_model_state = None
epochs_no_improve = 0

print("=" * 70)
print(f"  Starting Xception71 Training | Epochs: {NUM_EPOCHS} | Batch: {BATCH_SIZE} | LR: {LEARNING_RATE}")
print("=" * 70)

start_time = time.time()

for epoch in range(1, NUM_EPOCHS + 1):
    # ---------- TRAIN ----------
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        train_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
        
    train_loss /= train_total
    train_acc = 100.0 * train_correct / train_total
    
    # ---------- VALIDATE ----------
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    val_preds, val_targets = [], []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
            val_loss += loss.item() * inputs.size(0)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
            
            val_preds.extend(probs.cpu().tolist())
            val_targets.extend(labels.cpu().tolist())
            
    val_loss /= val_total
    val_acc = 100.0 * val_correct / val_total
    
    # Update scheduler
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]["lr"]
    
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
    
    # Check best model
    improved = False
    if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
        best_val_acc = val_acc
        best_val_loss = val_loss
        best_model_state = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0
        improved = True
    else:
        epochs_no_improve += 1
        
    mark = " ⭐ BEST" if improved else ""
    print(f"Epoch [{epoch:2d}/{NUM_EPOCHS}] | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}% | "
          f"LR: {current_lr:.1e}{mark}")
          
    if epochs_no_improve >= PATIENCE:
        print(f"\n⏹ Early stopping triggered after {epoch} epochs (no improvement for {PATIENCE} epochs)")
        break

elapsed = time.time() - start_time
print("=" * 70)
print(f"Training completed in {elapsed/60:.1f} minutes")
print(f"Best Validation Accuracy: {best_val_acc:.2f}% (Loss: {best_val_loss:.4f})")
print("=" * 70)

# %% [markdown]
# ## Cell 7: Save Fine-Tuned Checkpoint

# %%
checkpoint_to_save = {
    "state_dict": best_model_state,
    "val_acc": best_val_acc,
    "val_loss": best_val_loss,
    "model_name": "xception71",
    "history": history,
}

torch.save(checkpoint_to_save, OUTPUT_PATH)
file_size_mb = os.path.getsize(OUTPUT_PATH) / 1024**2

print(f"  ✓ Checkpoint saved successfully!")
print(f"  ✓ Path: {OUTPUT_PATH}")
print(f"  ✓ Size: {file_size_mb:.1f} MB")
print("\n👉 Download this file and place it in your project at:")
print("   models/results(Xception)/best_xception71.pth")

# %% [markdown]
# ## Cell 8: Plot Training Curves

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss
ax1.plot(history["train_loss"], label="Train Loss", color="#e74c3c", linewidth=2)
ax1.plot(history["val_loss"], label="Val Loss", color="#3498db", linewidth=2)
ax1.set_title("Xception71 Fine-Tuning Loss", fontsize=13, fontweight="bold")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Accuracy
ax2.plot(history["train_acc"], label="Train Acc", color="#e74c3c", linewidth=2)
ax2.plot(history["val_acc"], label="Val Acc", color="#2ecc71", linewidth=2)
ax2.axhline(y=best_val_acc, color="#f39c12", linestyle="--", label=f"Best Val: {best_val_acc:.1f}%")
ax2.set_title("Xception71 Fine-Tuning Accuracy", fontsize=13, fontweight="bold")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "xception_training_curves.png", dpi=200)
plt.show()
