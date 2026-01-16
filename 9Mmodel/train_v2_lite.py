"""
Training Script for RAG-RVT V2 Lite (<10M Parameters)
======================================================
"""

import os, math, json
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

from RAG_RVT_V2_LITE import RAG_RVT_V2_Lite

# -----------------------------
# User paths
# -----------------------------
MODEL_TAG = "RAG_RVT_V2_LITE"
csv_path = r"C:\Users\User\Downloads\d_19 - Dataset 19.csv"
img_dir  = r"F:\Dataset19"

# -----------------------------
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
    drop_path: float = 0.1
    embed_dim: int = 192  # Reduced for parameter efficiency
    use_auxiliary_loss: bool = True
    auxiliary_weight: float = 0.3
    model_save: str = f"best_{MODEL_TAG}.pt"
    out_dir: str = f"outputs_{MODEL_TAG}"
    patience: int = 20

cfg = CFG()
os.makedirs(cfg.out_dir, exist_ok=True)

# -----------------------------
# Dataset
# -----------------------------
class FundusCSVDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        fname_candidates = [c for c in df.columns if c.lower() in ["image","filename","file","img"]]
        if not fname_candidates:
            raise ValueError("CSV must contain an image/filename column.")
        self.fname_col = fname_candidates[0]
        label_col = None
        for c in ["label","class"]:
            if c in df.columns:
                label_col = c; break
        self.one_hot_cols = None
        if label_col is not None:
            if df[label_col].dtype == object:
                self.classes = sorted(df[label_col].astype(str).unique().tolist())
                self.targets = df[label_col].astype(str).map({c:i for i,c in enumerate(self.classes)}).astype(int).values
            else:
                self.targets = df[label_col].astype(int).values
                self.classes = sorted([str(x) for x in np.unique(self.targets)])
        else:
            candidate_cols = [c for c in df.columns if c.lower() not in [self.fname_col.lower(), "split"]]
            oh = []
            for c in candidate_cols:
                vals = set(pd.to_numeric(df[c], errors='coerce').fillna(-1).astype(int).unique().tolist())
                if vals.issubset({0,1}): oh.append(c)
            if len(oh) < 2:
                raise ValueError("Could not infer labels.")
            self.one_hot_cols = oh
            self.classes = oh
            self.targets = df[oh].values.astype(int).argmax(axis=1)
        assert len(self.classes) >= 2

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, str(row[self.fname_col]))
        with Image.open(img_path) as im:
            im = im.convert('RGB')
        if self.transform: im = self.transform(im)
        return im, int(self.targets[idx])

# -----------------------------
# Transforms
# -----------------------------
def make_transforms(img_size):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.3),
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
# Loader builder
# -----------------------------
def build_loaders_from_csv(csv_path, img_dir, img_size, batch_size, num_workers):
    df = pd.read_csv(csv_path)
    train_tf, val_tf = make_transforms(img_size)
    if 'split' in df.columns:
        tr_df = df[df['split'].str.lower().isin(['train','training'])]
        va_df = df[df['split'].str.lower().isin(['val','valid','validation'])]
    else:
        tmp = df.copy()
        label_col = None
        for c in ["label","class"]:
            if c in tmp.columns: label_col = c; break
        if label_col is None:
            fname_candidates = [c for c in tmp.columns if c.lower() in ["image","filename","file","img"]]
            fname_col = fname_candidates[0]
            candidate_cols = [c for c in tmp.columns if c.lower() not in [fname_col.lower(), "split"]]
            oh = [c for c in candidate_cols if set(pd.to_numeric(tmp[c], errors='coerce').fillna(-1).astype(int).unique().tolist()).issubset({0,1})]
            y = tmp[oh].values.argmax(axis=1)
        else:
            y = pd.to_numeric(tmp[label_col], errors='coerce')
            if y.isna().any(): y = pd.factorize(tmp[label_col].astype(str))[0]
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        idx_tr, idx_va = next(sss.split(tmp, y))
        tr_df, va_df = tmp.iloc[idx_tr], tmp.iloc[idx_va]
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
        return (min_lr / base_lr) + (1 - min_lr / base_lr) * 0.5 * (1 + math.cos(math.pi * q))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# -----------------------------
# Evaluation
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
    per_prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_rec = recall_score(y_true, y_pred, average=None, zero_division=0)
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True)
    return loss_sum/total, acc, macro_f1, macro_prec, macro_rec, cm, per_prec, per_rec, report

def visualize_dataset(tr_ds, va_ds, cm, class_names, out_dir):
    all_labels = np.concatenate([tr_ds.targets, va_ds.targets])
    plt.figure(figsize=(8,5))
    sns.countplot(x=all_labels)
    plt.xticks(ticks=range(len(class_names)), labels=class_names)
    plt.title("Class Distribution"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "class_distribution.png")); plt.close()
    
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png")); plt.close()

# -----------------------------
# Train
# -----------------------------
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tr_loader, va_loader, tr_ds, va_ds = build_loaders_from_csv(csv_path, img_dir, cfg.img_size, cfg.batch_size, cfg.num_workers)
    class_names = tr_ds.classes
    assert len(class_names) == cfg.num_classes

    model = RAG_RVT_V2_Lite(
        num_classes=cfg.num_classes,
        img_size=cfg.img_size,
        cnn_backbone='efficientnet_b0',
        vit_backbone='vit_tiny_patch16_224',
        embed_dim=cfg.embed_dim,
        dropout=cfg.dropout,
        drop_path=cfg.drop_path,
        pretrained=True,
        use_auxiliary_loss=cfg.use_auxiliary_loss,
        auxiliary_weight=cfg.auxiliary_weight
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params/1e6:.2f}M {'✅ <10M' if n_params < 10_000_000 else '❌ >10M'}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = cosine_scheduler(optimizer, cfg.lr, cfg.epochs, len(tr_loader))
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    history = {"epoch":[], "train_loss":[], "val_loss":[], "val_acc":[], "val_macro_f1":[]}
    best_f1, best_acc, best_epoch, epochs_no_improve = 0.0, 0.0, -1, 0

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        for imgs, labels in tr_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            if cfg.use_auxiliary_loss:
                final_logits, main_logits, cnn_logits, vit_logits = model(imgs, return_all_logits=True)
                total_loss, _, _ = model.compute_auxiliary_loss(main_logits, cnn_logits, vit_logits, labels, criterion)
                final_loss = criterion(final_logits, labels)
                loss = 0.5 * total_loss + 0.5 * final_loss
            else:
                logits = model(imgs)
                loss = criterion(logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item() * imgs.size(0)
            
        train_loss = epoch_loss / len(tr_loader.dataset)
        val_loss, val_acc, val_f1, val_prec, val_rec, cm, _, _, _ = evaluate(model, va_loader, device, class_names)

        history["epoch"].append(epoch+1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_f1)

        print(f"Epoch {epoch+1}/{cfg.epochs} | train={train_loss:.4f} val={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f}")

        improved = (val_f1 > best_f1) or (val_f1 == best_f1 and val_acc > best_acc)
        if improved:
            best_f1, best_acc, best_epoch = val_f1, val_acc, epoch
            torch.save({'model': model.state_dict(), 'classes': class_names, 'cfg': cfg.__dict__}, cfg.model_save)
            print(f"  ✔ Saved best")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= cfg.patience:
            print(f"Early stopping. Best @ epoch {best_epoch+1}: F1={best_f1:.4f}, Acc={best_acc:.4f}")
            break

    # Save history
    pd.DataFrame(history).to_csv(os.path.join(cfg.out_dir, 'training_history.csv'), index=False)

    # Final evaluation
    if os.path.exists(cfg.model_save):
        ckpt = torch.load(cfg.model_save, map_location=device)
        model.load_state_dict(ckpt['model'])
    val_loss, val_acc, val_f1, val_prec, val_rec, cm, per_prec, per_rec, report = evaluate(model, va_loader, device, class_names)
    
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:")
    print(pd.DataFrame(report).transpose())

    with open(os.path.join(cfg.out_dir, 'val_classification_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    visualize_dataset(tr_ds, va_ds, cm, class_names, cfg.out_dir)
    
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

    print(f"\n{'='*50}")
    print(f"🎉 Training Complete! Best: Epoch {best_epoch+1}, F1={best_f1:.4f}, Acc={best_acc:.4f}")
    print(f"{'='*50}")

if __name__=="__main__":
    train()
