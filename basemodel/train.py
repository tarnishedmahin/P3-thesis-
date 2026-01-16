import os, math, time, json
import pandas as pd
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer
from rarvit import RAG_RVT

# -----------------------------
# User paths (edit these two lines)
# -----------------------------
MODEL_TAG = "RAG_RVT_OLD" 
csv_path = r"C:\Users\User\Downloads\d_19 - Dataset 19.csv"
img_dir  = r"F:\Dataset19"

@dataclass
class CFG:
    img_size: int = 224
    num_classes: int = 4
    epochs: int = 200
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 5e-2
    num_workers: int = 4
    label_smoothing: float = 0.05
    dropout: float = 0.15
    model_save: str = f"best_{MODEL_TAG}.pt"
    out_dir: str = f"outputs_{MODEL_TAG}"
    patience: int = 20  # <-- Early stopping patience (epochs) on val macro-F1

cfg = CFG()
os.makedirs(cfg.out_dir, exist_ok=True)

# -----------------------------
# Dataset from CSV
# -----------------------------
class FundusCSVDataset(Dataset):
    """
    Expects a CSV with a filename column (one of: image, filename, file, img)
    and labels in one of these forms:
      (A) single column named 'label' or 'class' (string or int)
      (B) one-hot columns for class names (0/1); we take argmax as class
    If a 'split' column exists (train/val/test), it's honored by the loader builder.
    """
    def __init__(self, df, img_dir, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        # detect filename column
        fname_candidates = [c for c in df.columns if c.lower() in ["image","filename","file","img"]]
        if not fname_candidates:
            raise ValueError("CSV must contain an image/filename column (image|filename|file|img).")
        self.fname_col = fname_candidates[0]
        # detect label schema
        label_col = None
        for c in ["label","class"]:
            if c in df.columns:
                label_col = c; break
        self.one_hot_cols = None
        if label_col is not None:
            # build class list
            if df[label_col].dtype == object:
                self.classes = sorted(df[label_col].astype(str).unique().tolist())
                self.targets = df[label_col].astype(str).map({c:i for i,c in enumerate(self.classes)}).astype(int).values
            else:
                # numeric labels 0..K-1
                self.targets = df[label_col].astype(int).values
                self.classes = sorted([str(x) for x in np.unique(self.targets)])
        else:
            # try one-hot
            candidate_cols = [c for c in df.columns if c.lower() not in [self.fname_col.lower(), "split"]]
            oh = []
            for c in candidate_cols:
                vals = set(pd.to_numeric(df[c], errors='coerce').fillna(-1).astype(int).unique().tolist())
                if vals.issubset({0,1}):
                    oh.append(c)
            if len(oh) < 2:
                raise ValueError("Could not infer labels. Provide a 'label' column or one-hot columns.")
            self.one_hot_cols = oh
            self.classes = oh
            onehot = df[oh].values.astype(int)
            self.targets = onehot.argmax(axis=1)
        # sanity
        assert len(self.classes) >= 2

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, str(row[self.fname_col]))
        with Image.open(img_path) as im:
            im = im.convert('RGB')
        if self.transform is not None:
            im = self.transform(im)
        target = int(self.targets[idx])
        return im, target

# -----------------------------
# Transforms (augment TRAIN from the start; keep VAL clean)
# -----------------------------
def make_transforms(img_size):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
        transforms.RandAugment(num_ops=2, magnitude=7),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(img_size*1.05)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return train_tf, val_tf

# -----------------------------
# Loader builder using optional 'split' column
# -----------------------------
def build_loaders_from_csv(csv_path, img_dir, img_size, batch_size, num_workers):
    df = pd.read_csv(csv_path)
    train_tf, val_tf = make_transforms(img_size)
    if 'split' in df.columns:
        tr_df = df[df['split'].str.lower().isin(['train','training'])]
        va_df = df[df['split'].str.lower().isin(['val','valid','validation'])]
    else:
        # If no split provided, do 85/15 stratified split by detected label
        tmp = df.copy()
        label_col = None
        for c in ["label","class"]:
            if c in tmp.columns: label_col = c; break
        if label_col is None:
            # try one-hot
            fname_candidates = [c for c in tmp.columns if c.lower() in ["image","filename","file","img"]]
            fname_col = fname_candidates[0]
            candidate_cols = [c for c in tmp.columns if c.lower() not in [fname_col.lower(), "split"]]
            oh = []
            for c in candidate_cols:
                vals = set(pd.to_numeric(tmp[c], errors='coerce').fillna(-1).astype(int).unique().tolist())
                if vals.issubset({0,1}): oh.append(c)
            if len(oh) < 2: raise ValueError("CSV needs a split column or labels to stratify.")
            y = tmp[oh].values.argmax(axis=1)
        else:
            y = pd.to_numeric(tmp[label_col], errors='coerce')
            if y.isna().any():
                # treat as strings → factorize
                y = pd.factorize(tmp[label_col].astype(str))[0]
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        idx_tr, idx_va = next(sss.split(tmp, y))
        tr_df = tmp.iloc[idx_tr]
        va_df = tmp.iloc[idx_va]
    tr_ds = FundusCSVDataset(tr_df, img_dir, train_tf)
    va_ds = FundusCSVDataset(va_df, img_dir, val_tf)
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return tr_loader, va_loader, tr_ds, va_ds

# -----------------------------
# Scheduler
# -----------------------------
def cosine_scheduler(optimizer, base_lr, epochs, steps_per_epoch, min_lr=1e-6, warmup_epochs=5):
    def lr_lambda(step):
        total_steps = epochs * steps_per_epoch
        warmup_steps = warmup_epochs * steps_per_epoch
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        q = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * q))
        return (min_lr / base_lr) + (1 - (min_lr / base_lr)) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# -----------------------------
# Evaluation helpers
# -----------------------------
def evaluate(model, loader, device, class_names):
    model.eval()
    all_preds, all_targets = [], []
    total, correct, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss_sum += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
            all_preds.append(preds.cpu())
            all_targets.append(labels.cpu())
    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_targets).numpy()

    acc = correct / total
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    macro_prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    # Per-class metrics
    per_prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_rec  = recall_score(y_true, y_pred, average=None, zero_division=0)  # per-class accuracy/sensitivity

    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True)
    return (loss_sum/total, acc, macro_f1, macro_prec, macro_rec, cm, per_prec, per_rec, report)

def visualize_dataset(tr_ds, va_ds, cm, class_names, out_dir):
    # --- Print dataset sizes ---
    print(f"\nTrain set size: {len(tr_ds)}")
    print(f"Valid set size: {len(va_ds)}")

    # --- Barplot of distribution ---
    all_labels = np.concatenate([tr_ds.targets, va_ds.targets])
    plt.figure(figsize=(8,5))
    sns.countplot(x=all_labels)
    plt.xticks(ticks=range(len(class_names)), labels=class_names)
    plt.title("Class Distribution (Train+Valid)")
    plt.ylabel("Number of Images")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "class_distribution.png"))
    plt.close()

    # --- Label co-occurrence matrix ---
    mlb = MultiLabelBinarizer(classes=range(len(class_names)))
    Y = mlb.fit_transform([[y] for y in all_labels])
    co_matrix = Y.T @ Y
    plt.figure(figsize=(6,5))
    sns.heatmap(co_matrix, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.title("Label Co-occurrence Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "label_cooccurrence.png"))
    plt.close()

    # --- Sample images grid (one per class) ---
    fig, axes = plt.subplots(1, len(class_names), figsize=(16,4))
    for i, cls in enumerate(class_names):
        idxs = np.where(all_labels == i)[0]
        if len(idxs) == 0:
            continue
        idx = idxs[0]
        img, _ = (tr_ds[idx] if idx < len(tr_ds) else va_ds[idx - len(tr_ds)])
        img_show = img.permute(1,2,0).numpy()
        img_show = (img_show * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406])).clip(0,1)
        axes[i].imshow(img_show)
        axes[i].axis("off")
        axes[i].set_title(f"{cls}\nCount: {np.sum(all_labels==i)}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "class_examples.png"))
    plt.close()

    # --- Confusion matrix ---
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"))
    plt.close()


# -----------------------------
# Train (with Early Stopping on macro-F1)
# -----------------------------
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tr_loader, va_loader, tr_ds, va_ds = build_loaders_from_csv(csv_path, img_dir, cfg.img_size, cfg.batch_size, cfg.num_workers)
    class_names = tr_ds.classes
    assert len(class_names) == cfg.num_classes, f"Detected {len(class_names)} classes: {class_names}. Update cfg.num_classes if needed."

    model = RAG_RVT(
        num_classes=cfg.num_classes,
        img_size=cfg.img_size,
        cnn_backbone='efficientnet_b0',      # You can change this to another EfficientNet, e.g., 'efficientnet_b1'
        vit_backbone='vit_tiny_patch16_224', # Or change this to another ViT, e.g., 'vit_small_patch16_224'
        embed_dim=256,                       # This is the main hyperparameter for our fusion module
        pretrained=True
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = len(tr_loader)
    scheduler = cosine_scheduler(optimizer, cfg.lr, cfg.epochs, steps_per_epoch)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    history = {"epoch":[], "train_loss":[], "val_loss":[], "val_acc":[], "val_macro_f1":[], "val_macro_precision":[], "val_macro_recall":[]}

    best_f1 = 0.0
    best_acc = 0.0
    best_epoch = -1
    epochs_no_improve = 0

    global_step = 0
    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        for imgs, labels in tr_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step(); global_step += 1
            epoch_loss += loss.item() * imgs.size(0)
        train_loss = epoch_loss / len(tr_loader.dataset)

        # ---- validation ----
        val_loss, val_acc, val_f1, val_prec, val_rec, cm, per_prec, per_rec, report = evaluate(model, va_loader, device, class_names)

        # log history
        history["epoch"].append(epoch+1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_f1)
        history["val_macro_precision"].append(val_prec)
        history["val_macro_recall"].append(val_rec)

        print(f"Epoch {epoch+1}/{cfg.epochs} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_acc={val_acc:.4f} val_f1={val_f1:.4f} val_prec={val_prec:.4f} val_rec={val_rec:.4f}")

        # ---- early stopping on macro-F1 (tie-break by accuracy) ----
        improved = (val_f1 > best_f1) or (val_f1 == best_f1 and val_acc > best_acc)
        if improved:
            best_f1, best_acc, best_epoch = val_f1, val_acc, epoch
            torch.save({'model': model.state_dict(), 'classes': class_names, 'cfg': cfg.__dict__}, cfg.model_save)
            print(f"  ✔ Saved new best to {cfg.model_save}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= cfg.patience:
            print(f"Early stopping at epoch {epoch+1}. Best @ epoch {best_epoch+1}: F1={best_f1:.4f}, Acc={best_acc:.4f}")
            break

    # Save training history and plot curves
    hist_df = pd.DataFrame(history)
    hist_csv = os.path.join(cfg.out_dir, 'training_history.csv')
    hist_df.to_csv(hist_csv, index=False)
    print(f"History saved to {hist_csv}")

    # Plot loss curves
    plt.figure()
    plt.plot(hist_df['epoch'], hist_df['train_loss'], label='Train Loss')
    plt.plot(hist_df['epoch'], hist_df['val_loss'], label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss Curve'); plt.legend(); plt.tight_layout()
    loss_png = os.path.join(cfg.out_dir, 'loss_curve.png')
    plt.savefig(loss_png); plt.close()

    # Plot validation metric curves
    plt.figure()
    plt.plot(hist_df['epoch'], hist_df['val_macro_f1'], label='Macro F1')
    plt.plot(hist_df['epoch'], hist_df['val_macro_precision'], label='Macro Precision')
    plt.plot(hist_df['epoch'], hist_df['val_macro_recall'], label='Macro Recall')
    plt.plot(hist_df['epoch'], hist_df['val_acc'], label='Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Score'); plt.title('Validation Metrics'); plt.legend(); plt.tight_layout()
    metrics_png = os.path.join(cfg.out_dir, 'val_metrics_curve.png')
    plt.savefig(metrics_png); plt.close()

    # ---- load best weights for the final report ----
    if os.path.exists(cfg.model_save):
        ckpt = torch.load(cfg.model_save, map_location=device)
        model.load_state_dict(ckpt['model'])

    # Final detailed report on validation set
    val_loss, val_acc, val_f1, val_prec, val_rec, cm, per_prec, per_rec, report = evaluate(model, va_loader, device, class_names)
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:")
    print(pd.DataFrame(report).transpose())

    # Per-class accuracy (recall) printing (accuracy by disease)
    print("\nPer-class accuracy (recall):")
    for cls, r in zip(class_names, per_rec):
        print(f"  {cls}: {r:.4f}")

    # Save artifacts
    with open(os.path.join(cfg.out_dir, 'val_classification_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    np.save(os.path.join(cfg.out_dir, 'confusion_matrix.npy'), cm)

        # Extra dataset & confusion visualizations
    visualize_dataset(tr_ds, va_ds, cm, class_names, cfg.out_dir)
    print(f"Extra plots saved in {cfg.out_dir} (class_distribution.png, label_cooccurrence.png, "
          f"class_examples.png, confusion_matrix.png)")
    # Save final summary metrics for comparison
    final_metrics = {
        "model": MODEL_TAG,
        "val_loss": float(val_loss),
        "val_acc": float(val_acc),
        "val_macro_f1": float(val_f1),
        "val_macro_precision": float(val_prec),
        "val_macro_recall": float(val_rec)
    }

    with open(os.path.join(cfg.out_dir, "final_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)



if __name__=="__main__":
    train()
