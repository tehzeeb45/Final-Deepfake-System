# %% [markdown]
# # 🔧 EfficientNet-B5 Fine-Tuning for DeepFake Detection
# 
# **Purpose:** Fine-tune the existing EfficientNet-B5 model on WhatsApp/social media images to fix false positives.
# 
# ## Setup Instructions:
# 1. **GPU Enable:** Settings → Accelerator → GPU T4 x2 (or P100)
# 2. **Upload 2 Datasets:**
#    - Dataset 1: `finetune-data` → your real/fake images (folder structure below)
#    - Dataset 2: `efficientnet-checkpoint` → your `best_efficientnet_b5.pth` file
#
# ### Data folder structure (upload as Kaggle Dataset):
# ```
# finetune-data/
# ├── train/
# │   ├── real/    (200+ real photos)
# │   └── fake/    (200+ fake photos)
# └── val/
#     ├── real/    (50+ real photos)
#     └── fake/    (50+ fake photos)
# ```

# %% [markdown]
# ## Cell 1: Install Dependencies

# %%
!pip install timm --quiet

# %% [markdown]
# ## Cell 2: Configuration

# %%
import os
import copy
import time
import glob
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION — Update these paths to match your Kaggle datasets
# ============================================================

# Option 1: If you uploaded data as a Kaggle Dataset named "finetune-data"
DATA_DIR = Path("/kaggle/input/finetune-data")

# Option 2: If you uploaded a zip file, uncomment and update:
# !unzip /kaggle/input/your-dataset-name/finetune_data.zip -d /kaggle/working/
# DATA_DIR = Path("/kaggle/working/finetune_data")

# Checkpoint path — update to match your uploaded dataset name
CHECKPOINT_PATH = Path("/kaggle/input/efficientnet-checkpoint/best_efficientnet_b5.pth")

# Output (saved to /kaggle/working/ which is downloadable)
OUTPUT_DIR = Path("/kaggle/working")
OUTPUT_PATH = OUTPUT_DIR / "best_efficientnet_b5_finetuned.pth"

# Training hyperparameters
BATCH_SIZE = 8           # Reduced to 8 to fit 456x456 images in GPU VRAM comfortably
NUM_EPOCHS = 30          # Max epochs (early stopping will stop at 10-15)
LEARNING_RATE = 5e-5     # Optimal LR for classifier head fine-tuning
WEIGHT_DECAY = 1e-4      # L2 regularization
PATIENCE = 7             # Early stopping patience
INPUT_SIZE = 456         # EfficientNet-B5 native resolution
NUM_WORKERS = 2          # Kaggle supports multiprocessing

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# %% [markdown]
# ## Cell 3: Verify Data

# %%
# Check if data folders exist
print("=" * 50)
print("Data Verification")
print("=" * 50)

for split in ["train", "val"]:
    for cls in ["real", "fake"]:
        folder = DATA_DIR / split / cls
        # Check both 'real'/'fake' and '0_real'/'1_fake'
        if not folder.exists():
            alt = DATA_DIR / split / ("0_real" if cls == "real" else "1_fake")
            if alt.exists():
                folder = alt
        if folder.exists():
            count = len(list(folder.glob("*")))
            print(f"  ✓ {split}/{cls}: {count} images (folder: {folder.name})")
        else:
            print(f"  ✗ {split}/{cls}: NOT FOUND at {folder}")
            print(f"    → Check your dataset upload!")

# Check checkpoint
if CHECKPOINT_PATH.exists():
    size_mb = CHECKPOINT_PATH.stat().st_size / 1024 / 1024
    print(f"\n  ✓ Checkpoint: {CHECKPOINT_PATH.name} ({size_mb:.0f} MB)")
else:
    print(f"\n  ✗ Checkpoint NOT FOUND at {CHECKPOINT_PATH}")
    print(f"    → Upload best_efficientnet_b5.pth as a Kaggle Dataset")
    # List available input datasets to help debug
    print(f"\n  Available inputs:")
    for p in Path("/kaggle/input").iterdir():
        print(f"    /kaggle/input/{p.name}/")
        for f in list(p.rglob("*"))[:5]:
            print(f"      {f}")

# %% [markdown]
# ## Cell 4: Create Model (Load & Freeze)

# %%
import timm

def create_model(checkpoint_path, device):
    """Load pre-trained EfficientNet-B5 and freeze early layers."""
    
    print("Loading pre-trained EfficientNet-B5...")
    model = timm.create_model(
        "tf_efficientnet_b5.ns_jft_in1k", pretrained=False, num_classes=2
    )
    
    # Load existing checkpoint
    raw = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    sd = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
    model.load_state_dict(sd, strict=True)
    print("  ✓ Checkpoint loaded")
    
    # ============================================================
    # STRICT FREEZE STRATEGY
    # Freeze entire backbone; train ONLY conv_head + classifier (1M params)
    # This prevents overfitting on small datasets (~200 images)
    # ============================================================
    
    # First freeze ALL
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze only classifier head + top conv layer
    for param in model.classifier.parameters():
        param.requires_grad = True
    for param in model.conv_head.parameters():
        param.requires_grad = True
    for param in model.bn2.parameters():
        param.requires_grad = True
    
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ Parameters: {total:,} total, {trainable:,} trainable ({trainable/total*100:.1f}%)")
    print(f"  ✓ Frozen: {(total-trainable)/total*100:.1f}%")
    
    return model.to(device)

model = create_model(CHECKPOINT_PATH, device)

# %% [markdown]
# ## Cell 5: Create DataLoaders (with Heavy Augmentation)

# %%
# Training: HEAVY augmentation to prevent overfitting
train_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE + 32, INPUT_SIZE + 32)),
    transforms.RandomCrop(INPUT_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    transforms.RandomGrayscale(p=0.05),
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
    ], p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
])

# Validation: NO augmentation
val_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

class DeepfakeDataset(datasets.ImageFolder):
    """
    Ensures Real=0 and Fake=1 regardless of folder alphabetical order.
    Works whether folders are named 'real'/'fake' or '0_real'/'1_fake'.
    """
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
                raise ValueError(f"Unknown class folder '{c}'. Folder name must contain 'real' or 'fake'.")
        return classes, class_to_idx

train_dataset = DeepfakeDataset(DATA_DIR / "train", transform=train_transform)
val_dataset = DeepfakeDataset(DATA_DIR / "val", transform=val_transform)

print(f"Classes: {train_dataset.class_to_idx}")
print(f"Train: {len(train_dataset)} images")
print(f"Val:   {len(val_dataset)} images")

# Verify: Real=0, Fake=1
for cls_name, idx in train_dataset.class_to_idx.items():
    if "real" in cls_name.lower():
        assert idx == 0, f"ERROR: '{cls_name}' must map to class 0"
    elif "fake" in cls_name.lower():
        assert idx == 1, f"ERROR: '{cls_name}' must map to class 1"
print("  ✓ Class mapping verified: Real=0, Fake=1 (matches pre-trained model)")

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True, drop_last=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True
)

# %% [markdown]
# ## Cell 6: Training Loop

# %%
# Loss, optimizer, scheduler
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)

# Mixed precision (FP16) - cuts GPU memory in half & speeds up T4 GPU
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
torch.cuda.empty_cache()

# Training history for plotting
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

# Early stopping state
best_val_acc = 0.0
best_val_loss = float("inf")
best_model_state = None
epochs_no_improve = 0

print("=" * 70)
print(f"  Starting Training | Epochs: {NUM_EPOCHS} | Batch: {BATCH_SIZE} | LR: {LEARNING_RATE}")
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
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            val_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    
    val_loss /= val_total
    val_acc = 100.0 * val_correct / val_total
    
    # Record history
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
    
    # LR scheduler
    scheduler.step(val_loss)
    lr = optimizer.param_groups[0]["lr"]
    
    # Print progress & save BEST model by validation accuracy
    marker = ""
    improved = False
    if val_acc > best_val_acc:
        improved = True
    elif abs(val_acc - best_val_acc) < 1e-4 and val_loss < best_val_loss:
        improved = True
        
    if improved:
        best_val_loss = val_loss
        best_val_acc = val_acc
        best_model_state = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0
        marker = " ★ BEST"
    else:
        epochs_no_improve += 1
    
    print(
        f"  Epoch {epoch:2d}/{NUM_EPOCHS} | "
        f"Train: {train_loss:.4f} ({train_acc:.1f}%) | "
        f"Val: {val_loss:.4f} ({val_acc:.1f}%) | "
        f"LR: {lr:.1e}{marker}"
    )
    
    # Early stopping
    if epochs_no_improve >= PATIENCE:
        print(f"\n  ⏹ Early stopping! No improvement for {PATIENCE} epochs.")
        break

total_time = time.time() - start_time
print(f"\n  Training complete in {total_time/60:.1f} minutes")
print(f"  Best validation accuracy: {best_val_acc:.1f}%")

# %% [markdown]
# ## Cell 7: Training Curves (Loss & Accuracy)

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

epochs_range = range(1, len(history["train_loss"]) + 1)

# Loss
ax1.plot(epochs_range, history["train_loss"], "b-o", label="Train Loss", markersize=4)
ax1.plot(epochs_range, history["val_loss"], "r-o", label="Val Loss", markersize=4)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Training vs Validation Loss")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Accuracy
ax2.plot(epochs_range, history["train_acc"], "b-o", label="Train Acc", markersize=4)
ax2.plot(epochs_range, history["val_acc"], "r-o", label="Val Acc", markersize=4)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Training vs Validation Accuracy")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/kaggle/working/training_curves.png", dpi=150, bbox_inches="tight")
plt.show()

print("  ✓ Saved: /kaggle/working/training_curves.png")

# %% [markdown]
# ## Cell 8: Save Fine-Tuned Model

# %%
if best_model_state is not None:
    save_dict = {
        "state_dict": best_model_state,
        "val_acc": best_val_acc,
        "val_loss": best_val_loss,
        "class_to_idx": train_dataset.class_to_idx,
        "fine_tuned": True,
    }
    
    torch.save(save_dict, str(OUTPUT_PATH))
    size_mb = OUTPUT_PATH.stat().st_size / 1024 / 1024
    print(f"  ✓ Model saved: {OUTPUT_PATH}")
    print(f"  ✓ Size: {size_mb:.1f} MB")
    print(f"  ✓ Val Accuracy: {best_val_acc:.1f}%")
    print()
    print("=" * 50)
    print("  DOWNLOAD INSTRUCTIONS")
    print("=" * 50)
    print(f"""
  1. Click 'Output' tab (right side panel)
  2. Download: best_efficientnet_b5_finetuned.pth
  3. Put it in: models/results(Eff)/ folder
  4. Update config.yaml:
     checkpoint: ./models/results(Eff)/best_efficientnet_b5_finetuned.pth
  5. Restart app: python app.py
  6. Test with your images!
""")
else:
    print("  ✗ No best model found. Training may have failed.")

# %% [markdown]
# ## Cell 9: Quick Validation Test (Optional)
# 
# Run this to see how the fine-tuned model performs on individual images.

# %%
from PIL import Image

@torch.no_grad()
def predict_single(model, image_path, device):
    """Predict P(Fake) for a single image."""
    transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    
    model.eval()
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)
    p_fake = probs[0, 1].item()
    
    return p_fake

# Load best model for testing
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    model.eval()

# Test on validation images
print("Validation Results (first 10):")
print("-" * 50)

val_images = list((DATA_DIR / "val").rglob("*.jpg")) + list((DATA_DIR / "val").rglob("*.png"))
for img_path in val_images[:10]:
    p_fake = predict_single(model, img_path, device)
    truth = "FAKE" if "fake" in str(img_path).lower() else "REAL"
    pred = "FAKE" if p_fake >= 0.50 else "REAL"
    status = "✓" if truth == pred else "✗"
    print(f"  {status} {img_path.name[:25]:25s} | Truth: {truth} | Pred: {pred} | P(Fake): {p_fake:.1%}")
