import os
import random
import argparse
from pathlib import Path
import copy

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.metrics import cohen_kappa_score, f1_score, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight

from transformers import CLIPProcessor, CLIPModel

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
    raise ValueError(f"Label not found: {base_name}")


def normalize_img_key(path: str) -> str:
    import re
    return re.sub(r"\.(jpg|jpeg|png)$", "", str(path), flags=re.IGNORECASE)


class PrivacyDataset(Dataset):
    """
    [FIX] train=True일 때만 RandomHorizontalFlip(p=0.5) 적용.
    (DenseNet/EfficientNet과 동일한 augmentation 정책으로 통일)
    """
    def __init__(self, df, image_dir, label_col, processor, train=False):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.label_col = label_col
        self.processor = processor
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.image_dir / Path(str(row["image_path"])).name

        image = Image.open(img_path).convert("RGB")
        label = int(row[self.label_col])

        if self.train and random.random() < 0.5:
            image = ImageOps.mirror(image)

        inputs = self.processor(images=image, return_tensors="pt")
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


class CLIPClassifier(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.clip.config.projection_dim, 3)

    def forward(self, pixel_values):
        outputs = self.clip.vision_model(pixel_values=pixel_values)
        pooled = outputs.pooler_output
        feats = self.clip.visual_projection(pooled)
        logits = self.classifier(feats)
        return logits


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        x = batch["pixel_values"].to(device)
        y = batch["label"].to(device)

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
    y_true, y_pred = [], []

    for batch in loader:
        x = batch["pixel_values"].to(device)
        y = batch["label"].to(device)

        logits = model(x)
        pred = torch.argmax(logits, dim=1)

        y_true.extend(y.cpu().numpy().tolist())
        y_pred.extend(pred.cpu().numpy().tolist())

    return eval_metrics(y_true, y_pred), y_true, y_pred


def run_one_label(df, image_dir, label_col, output_dir,
                  model_name, epochs, batch_size, seed, device,
                  clip_embedding_npz):

    valid_df = df[df[label_col].notna()].copy()
    valid_df[label_col] = valid_df[label_col].astype(int)

    valid_df["__img_exists"] = valid_df["image_path"].apply(
        lambda p: (Path(image_dir) / Path(str(p)).name).exists()
    )
    valid_df = valid_df[valid_df["__img_exists"]].drop(columns=["__img_exists"]).copy()
    valid_df = valid_df.reset_index(drop=True)

    y_all = valid_df[label_col].values

    # [FIX] StratifiedKFold -> StratifiedGroupKFold
    group_ids = build_group_ids(
        image_keys=[normalize_img_key(p) for p in valid_df["image_path"]],
        clip_embedding_npz=clip_embedding_npz,
    )
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)

    processor = CLIPProcessor.from_pretrained(model_name)

    fold_results = []
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

        train_ds = PrivacyDataset(train_df, image_dir, label_col, processor, train=True)
        val_ds   = PrivacyDataset(val_df,   image_dir, label_col, processor, train=False)
        test_ds  = PrivacyDataset(test_df,  image_dir, label_col, processor, train=False)

        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True,  num_workers=2)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                                  shuffle=False, num_workers=2)
        test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                                  shuffle=False, num_workers=2)

        model = CLIPClassifier(model_name).to(device)

    
        optimizer = torch.optim.AdamW(
            [
                {"params": model.clip.vision_model.parameters(),      "lr": 1e-5},
                {"params": model.clip.visual_projection.parameters(), "lr": 1e-5},
                {"params": model.classifier.parameters(),             "lr": 1e-4},
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
            train_loss  = train_one_epoch(
                model, train_loader, optimizer, criterion, device)
            val_metrics, _, _ = evaluate(model, val_loader, device)
            val_qwk = val_metrics["qwk"]

            scheduler.step()

            print(f"Epoch {epoch:02d} | loss={train_loss:.4f} | "
                  f"val_qwk={val_qwk:.3f} | val_f1={val_metrics['macro_f1']:.3f}")

            if val_qwk > best_val_qwk:
                best_val_qwk    = val_qwk
                best_state_dict = copy.deepcopy(
                    {k: v.cpu() for k, v in model.state_dict().items()}
                )
                patience_counter = 0
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

        fold_results.append({
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

    fold_df = pd.DataFrame(fold_results)
    pred_df = pd.concat(all_predictions, ignore_index=True)

    fold_df.to_csv(
        os.path.join(output_dir, f"{label_col}_clip_finetune_folds.csv"),
        index=False, encoding="utf-8-sig",
    )
    pred_df.to_csv(
        os.path.join(output_dir, f"{label_col}_clip_finetune_predictions.csv"),
        index=False, encoding="utf-8-sig",
    )

    return {
        "label":              label_col,
        "qwk_mean":           fold_df["qwk"].mean(),
        "qwk_std":            fold_df["qwk"].std(),
        "macro_f1_mean":      fold_df["macro_f1"].mean(),
        "macro_f1_std":       fold_df["macro_f1"].std(),
        "balanced_acc_mean":  fold_df["balanced_acc"].mean(),
        "balanced_acc_std":   fold_df["balanced_acc"].std(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_csv",   type=str, required=True)
    parser.add_argument("--image_dir",  type=str, required=True)
    parser.add_argument("--clip_embedding_npz", type=str, required=True,
                        help="중복 탐지에 사용할 CLIP 임베딩 npz 경로")
    parser.add_argument("--output_dir", type=str, default="results_clip_finetune_v5")
    parser.add_argument("--model_name", type=str,
                        default="openai/clip-vit-base-patch16")
    parser.add_argument("--epochs",     type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed",       type=int, default=42)
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
            model_name=args.model_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            device=device,
            clip_embedding_npz=args.clip_embedding_npz,
        )
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(
        os.path.join(args.output_dir, "clip_finetune_summary.csv"),
        index=False, encoding="utf-8-sig",
    )

    print("\nFinal summary")
    print(summary_df)


if __name__ == "__main__":
    main()