import os
import copy
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import models, transforms

from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.metrics import cohen_kappa_score, f1_score, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight

from dedup_fold_utils import build_group_ids


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def eval_metrics(y_true, y_pred):
    return {
        "qwk": float(qwk(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
    }


def find_label_col(df, base_name):
    for c in [f"{base_name}_gold", f"{base_name}_gold_max", base_name]:
        if c in df.columns:
            return c
    raise ValueError(f"Label not found for {base_name}")


def normalize_img_key(path: str) -> str:
    import re
    return re.sub(r"\.(jpg|jpeg|png)$", "", str(path), flags=re.IGNORECASE)


class PrivacyImageDataset(Dataset):
    def __init__(self, df, image_dir, label_col, transform):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.label_col = label_col
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.image_dir / Path(str(row["image_path"])).name

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        label = torch.tensor(int(row[self.label_col]), dtype=torch.long)
        return image, label


def build_efficientnet_b0(num_classes=3):
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_true, all_pred = [], []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        pred = torch.argmax(logits, dim=1)

        all_true.extend(y.cpu().numpy().tolist())
        all_pred.extend(pred.cpu().numpy().tolist())

    return eval_metrics(all_true, all_pred), all_true, all_pred


def run_one_label(
    df, image_dir, label_col, output_dir, epochs, batch_size,
    lr_backbone, lr_head, seed, device, clip_embedding_npz,
):
    valid_df = df[df[label_col].notna()].copy()
    valid_df[label_col] = valid_df[label_col].astype(int)

    valid_df["__exists"] = valid_df["image_path"].apply(
        lambda p: (Path(image_dir) / Path(str(p)).name).exists()
    )
    valid_df = valid_df[valid_df["__exists"]].drop(columns=["__exists"]).copy()
    valid_df = valid_df.reset_index(drop=True)

    y_all = valid_df[label_col].values

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
        image_keys=[normalize_img_key(p) for p in valid_df["image_path"]],
        clip_embedding_npz=clip_embedding_npz,
    )
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)

    fold_rows = []
    all_predictions = []

    for fold, (trainval_idx, test_idx) in enumerate(
        sgkf.split(valid_df, y_all, groups=group_ids), start=1
    ):
        print(f"\n[{label_col}] Fold {fold}/5")

        trainval_df = valid_df.iloc[trainval_idx].copy()
        test_df     = valid_df.iloc[test_idx].copy()

        train_df, val_df = train_test_split(
            trainval_df,
            test_size=0.1,
            stratify=trainval_df[label_col],
            random_state=seed + fold,
        )

        train_ds = PrivacyImageDataset(train_df, image_dir, label_col, train_transform)
        val_ds   = PrivacyImageDataset(val_df,   image_dir, label_col, eval_transform)
        test_ds  = PrivacyImageDataset(test_df,  image_dir, label_col, eval_transform)

        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True,  num_workers=2)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                                  shuffle=False, num_workers=2)
        test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                                  shuffle=False, num_workers=2)

        model = build_efficientnet_b0(num_classes=3).to(device)

        backbone_params = []
        head_params     = []
        for name, param in model.named_parameters():
            if name.startswith("classifier"):
                head_params.append(param)
            else:
                backbone_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": lr_backbone},
                {"params": head_params,     "lr": lr_head},
            ],
            weight_decay=1e-4,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
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

        best_val_qwk    = -999.0
        best_state_dict = None
        patience        = 3
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss          = train_one_epoch(
                model, train_loader, optimizer, criterion, device)
            val_metrics, _, _   = evaluate(model, val_loader, device)
            val_qwk             = val_metrics["qwk"]

            scheduler.step()

            print(
                f"Epoch {epoch:02d} | "
                f"loss={train_loss:.4f} | "
                f"val_qwk={val_qwk:.3f} | "
                f"val_f1={val_metrics['macro_f1']:.3f}"
            )

            if val_qwk > best_val_qwk:
                best_val_qwk     = val_qwk
                patience_counter = 0
                best_state_dict  = copy.deepcopy(
                    {k: v.cpu() for k, v in model.state_dict().items()}
                )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

        model.load_state_dict(
            {k: v.to(device) for k, v in best_state_dict.items()}
        )
        test_metrics, y_true, y_pred = evaluate(model, test_loader, device)
        print(f"Test metrics: {test_metrics}")

        del best_state_dict
        torch.cuda.empty_cache()

        fold_rows.append({
            "label":        label_col,
            "fold":         fold,
            "n_train":      len(train_df),
            "n_val":        len(val_df),
            "n_test":       len(test_df),
            "best_val_qwk": best_val_qwk,
            **test_metrics,
        })

        pred_df = test_df[["image_path", label_col]].copy()
        pred_df["fold"]   = fold
        pred_df["y_true"] = y_true
        pred_df["y_pred"] = y_pred
        all_predictions.append(pred_df)

    fold_df = pd.DataFrame(fold_rows)
    pred_df = pd.concat(all_predictions, ignore_index=True)

    fold_df.to_csv(
        os.path.join(output_dir, f"{label_col}_efficientnet_b0_folds.csv"),
        index=False, encoding="utf-8-sig",
    )
    pred_df.to_csv(
        os.path.join(output_dir, f"{label_col}_efficientnet_b0_predictions.csv"),
        index=False, encoding="utf-8-sig",
    )

    return {
        "label":             label_col,
        "qwk_mean":          fold_df["qwk"].mean(),
        "qwk_std":           fold_df["qwk"].std(),
        "macro_f1_mean":     fold_df["macro_f1"].mean(),
        "macro_f1_std":      fold_df["macro_f1"].std(),
        "balanced_acc_mean": fold_df["balanced_acc"].mean(),
        "balanced_acc_std":  fold_df["balanced_acc"].std(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_csv",    type=str, required=True)
    parser.add_argument("--image_dir",   type=str, required=True)
    parser.add_argument("--clip_embedding_npz", type=str, required=True,
                        help="Path to the CLIP embedding npz used for duplicate detection")
    parser.add_argument("--output_dir",  type=str,
                        default="results_efficientnet_b0_v3")
    parser.add_argument("--epochs",      type=int,   default=10)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--lr_head",     type=float, default=1e-4)
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = (
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device: {device}")

    df = pd.read_csv(args.gold_csv)

    label_cols = [
        find_label_col(df, "identifiability"),
        find_label_col(df, "location"),
        find_label_col(df, "activity"),
    ]

    summaries = []
    for label_col in label_cols:
        summary = run_one_label(
            df=df,
            image_dir=args.image_dir,
            label_col=label_col,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr_backbone=args.lr_backbone,
            lr_head=args.lr_head,
            seed=args.seed,
            device=device,
            clip_embedding_npz=args.clip_embedding_npz,
        )
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(
        os.path.join(args.output_dir, "efficientnet_b0_summary.csv"),
        index=False, encoding="utf-8-sig",
    )

    print("\nFinal summary")
    print(summary_df)


if __name__ == "__main__":
    main()
