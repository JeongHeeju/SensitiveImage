import os
import copy
import argparse
import random
import json
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import models, transforms

from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    classification_report,
    confusion_matrix,
)

from dedup_fold_utils import build_group_ids


# =========================
# Utils
# =========================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_image_path(image_dir, image_path):
    image_path = str(image_path)
    candidates = [
        image_path,
        os.path.join(image_dir, image_path),
        os.path.join(image_dir, os.path.basename(image_path)),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def normalize_img_key(path: str) -> str:
    import re
    return re.sub(r"\.(jpg|jpeg|png)$", "", str(path), flags=re.IGNORECASE)


# =========================
# Dataset
# =========================

class PrivacyImageDataset(Dataset):
    def __init__(self, df, image_dir, label_col, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.label_col = label_col
        self.transform = transform

        self.samples = []
        missing = 0

        for _, row in self.df.iterrows():
            img_path = find_image_path(image_dir, row["image_path"])
            if img_path is None:
                missing += 1
                continue

            label = int(row[label_col])
            self.samples.append((img_path, label))

        if missing > 0:
            print(f"[Warning] Missing images skipped: {missing}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


# =========================
# Model
# =========================

def build_densenet(num_classes):
    model = models.densenet121(
        weights=models.DenseNet121_Weights.IMAGENET1K_V1
    )
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model


# =========================
# Train / Eval
# =========================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)

        total_loss += loss.item() * images.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    qwk = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)

    return avg_loss, qwk, macro_f1, bal_acc, np.array(all_labels), np.array(all_preds)


# =========================
# Main experiment
# =========================

def run_for_label(args, label_col, device):
    print("\n" + "=" * 80)
    print(f"Label: {label_col}")
    print("=" * 80)

    df = pd.read_csv(args.gold_csv)

    if label_col not in df.columns:
        raise ValueError(f"Label column not found: {label_col}")

    df = df.dropna(subset=["image_path", label_col]).copy()
    df[label_col] = df[label_col].astype(int)
    df = df[df[label_col].isin([0, 1, 2])].copy()

    df["__img_exists"] = df["image_path"].apply(
        lambda p: find_image_path(args.image_dir, p) is not None
    )
    df = df[df["__img_exists"]].drop(columns=["__img_exists"]).copy()
    df = df.reset_index(drop=True)

    print("Dataset size:", len(df))
    print("Label distribution:")
    print(df[label_col].value_counts().sort_index())

    num_classes = 3

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    group_ids = build_group_ids(
        image_keys=[normalize_img_key(p) for p in df["image_path"]],
        clip_embedding_npz=args.clip_embedding_npz,
    )
    sgkf = StratifiedGroupKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.seed,
    )

    fold_results = []
    y_all = df[label_col].values

    for fold, (trainval_idx, test_idx) in enumerate(
        sgkf.split(df, y_all, groups=group_ids), start=1
    ):
        print(f"\n[{label_col}] Fold {fold}/{args.folds}")

        trainval_df = df.iloc[trainval_idx].copy()
        test_df = df.iloc[test_idx].copy()

        train_df, val_df = train_test_split(
            trainval_df,
            test_size=0.1,
            stratify=trainval_df[label_col],
            random_state=args.seed + fold,
        )

        train_dataset = PrivacyImageDataset(
            train_df, args.image_dir, label_col, transform=train_transform,
        )
        val_dataset = PrivacyImageDataset(
            val_df, args.image_dir, label_col, transform=eval_transform,
        )
        test_dataset = PrivacyImageDataset(
            test_df, args.image_dir, label_col, transform=eval_transform,
        )

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )

        model = build_densenet(num_classes).to(device)

        
        backbone_params = []
        head_params = []
        for name, param in model.named_parameters():
            if name.startswith("classifier"):
                head_params.append(param)
            else:
                backbone_params.append(param)

        
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": 1e-5},
                {"params": head_params,     "lr": 1e-4},
            ],
            weight_decay=1e-4,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs,
        )

        
        class_weights_np = compute_class_weight(
            class_weight="balanced",
            classes=np.array([0, 1, 2]),
            y=train_df[label_col].values,
        )
        class_weights = torch.tensor(
            class_weights_np, dtype=torch.float32, device=device
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        best_qwk = -999
        best_state = None
        best_metrics = None
        patience = 3
        patience_counter = 0

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device,
            )
            val_loss, qwk, macro_f1, bal_acc, y_true, y_pred = evaluate(
                model, val_loader, criterion, device,
            )
            scheduler.step()

            print(
                f"Epoch {epoch:02d}/{args.epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_qwk={qwk:.4f} | val_macro_f1={macro_f1:.4f} | "
                f"val_balanced_acc={bal_acc:.4f}"
            )

            if qwk > best_qwk:
                best_qwk = qwk
                patience_counter = 0
                best_state = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

        test_loss, test_qwk, test_macro_f1, test_bal_acc, y_true, y_pred = evaluate(
            model, test_loader, criterion, device,
        )
        print(
            f"Test metrics | qwk={test_qwk:.4f} | "
            f"macro_f1={test_macro_f1:.4f} | balanced_acc={test_bal_acc:.4f}"
        )

        label_out_dir = os.path.join(args.output_dir, label_col)
        os.makedirs(label_out_dir, exist_ok=True)
        model_path = os.path.join(label_out_dir, f"densenet121_fold{fold}.pt")
        torch.save(best_state, model_path)

        del best_state
        torch.cuda.empty_cache()

        report = classification_report(
            y_true, y_pred, labels=[0, 1, 2], zero_division=0, output_dict=True,
        )
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

        best_metrics = {
            "fold": fold,
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_test": len(test_df),
            "best_val_qwk": float(best_qwk),
            "qwk": float(test_qwk),
            "macro_f1": float(test_macro_f1),
            "balanced_acc": float(test_bal_acc),
            "test_loss": float(test_loss),
            "y_true": y_true.tolist(),
            "y_pred": y_pred.tolist(),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "model_path": model_path,
        }

        fold_results.append(best_metrics)
        print(f"Best fold QWK (test set): {best_metrics['qwk']:.4f}")
        print("Confusion matrix (test set):")
        print(cm)

    qwk_list = [r["qwk"] for r in fold_results]
    f1_list = [r["macro_f1"] for r in fold_results]
    bal_list = [r["balanced_acc"] for r in fold_results]

    summary = {
        "label": label_col,
        "model": "DenseNet121",
        "qwk_mean": float(np.mean(qwk_list)),
        "qwk_std": float(np.std(qwk_list)),
        "macro_f1_mean": float(np.mean(f1_list)),
        "macro_f1_std": float(np.std(f1_list)),
        "balanced_acc_mean": float(np.mean(bal_list)),
        "balanced_acc_std": float(np.std(bal_list)),
        "folds": fold_results,
    }

    label_out_dir = os.path.join(args.output_dir, label_col)
    os.makedirs(label_out_dir, exist_ok=True)
    summary_path = os.path.join(label_out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nSummary for", label_col)
    print(
        f"QWK: {summary['qwk_mean']:.4f} ± {summary['qwk_std']:.4f} | "
        f"Macro-F1: {summary['macro_f1_mean']:.4f} ± {summary['macro_f1_std']:.4f} | "
        f"Balanced Acc: {summary['balanced_acc_mean']:.4f} ± {summary['balanced_acc_std']:.4f}"
    )

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_csv", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--clip_embedding_npz", type=str, required=True,
                        help="Path to the CLIP embedding npz used for duplicate detection")
    parser.add_argument("--output_dir", type=str, default="results_densenet121_v3")
    parser.add_argument(
        "--labels", type=str, nargs="+",
        default=["identifiability_gold", "location_gold", "activity_gold"],
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("Model: DenseNet121")

    all_summaries = []
    for label_col in args.labels:
        summary = run_for_label(args, label_col, device)
        all_summaries.append(summary)

    final_rows = []
    for s in all_summaries:
        final_rows.append({
            "label": s["label"],
            "model": s["model"],
            "qwk_mean": s["qwk_mean"],
            "qwk_std": s["qwk_std"],
            "macro_f1_mean": s["macro_f1_mean"],
            "macro_f1_std": s["macro_f1_std"],
            "balanced_acc_mean": s["balanced_acc_mean"],
            "balanced_acc_std": s["balanced_acc_std"],
        })

    final_df = pd.DataFrame(final_rows)
    final_csv = os.path.join(args.output_dir, "final_summary.csv")
    final_df.to_csv(final_csv, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("Final summary")
    print(final_df)
    print("Saved:", final_csv)
    print("=" * 80)


if __name__ == "__main__":
    main()
