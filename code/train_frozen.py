import os
import json
import argparse
import random
from typing import Dict, List

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    balanced_accuracy_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier

from dedup_fold_utils import build_group_ids

# 1. Majority / Random
# 2. CLIP embedding + LogReg / LinearSVM / MLP
# 3. DINOv2 embedding + LogReg / LinearSVM / MLP
# 4. Places365 embedding + LogReg / LinearSVM / MLP

# =========================================================
# Utils
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def quadratic_weighted_kappa(y_true, y_pred) -> float:
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))


def balanced_acc(y_true, y_pred) -> float:
    return float(balanced_accuracy_score(y_true, y_pred))


def evaluate_predictions(y_true, y_pred) -> Dict[str, float]:
    return {
        "qwk": quadratic_weighted_kappa(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
        "balanced_acc": balanced_acc(y_true, y_pred),
    }


# =========================================================
# Data loading
# =========================================================
def load_feature_npz(npz_path: str) -> pd.DataFrame:
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Feature file not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    embeddings = data["embeddings"]
    image_ids = data["image_ids"]

    if len(embeddings) != len(image_ids):
        raise ValueError(f"Mismatch in {npz_path}: embeddings and image_ids length differ.")

    feat_cols = [f"f_{i}" for i in range(embeddings.shape[1])]
    df = pd.DataFrame(embeddings, columns=feat_cols)
    df.insert(0, "image_path", image_ids.astype(str))
    return df


def merge_gold_and_features(gold_csv: str, feature_npz: str) -> pd.DataFrame:
    if not os.path.exists(gold_csv):
        raise FileNotFoundError(f"Gold CSV not found: {gold_csv}")

    gold_df = pd.read_csv(gold_csv)
    feat_df = load_feature_npz(feature_npz)

    if "image_path" not in gold_df.columns:
        raise ValueError(f"'image_path' column not found in {gold_csv}")

    gold_df["_merge_key"] = gold_df["image_path"].apply(_normalize_img_key)
    feat_df["_merge_key"] = feat_df["image_path"].apply(_normalize_img_key)

    merged = gold_df.merge(
        feat_df.drop(columns=["image_path"]), on="_merge_key", how="inner"
    )

    if len(merged) == 0:
        raise ValueError(
            f"No matched rows between gold CSV and feature NPZ "
            f"(even after extension normalization):\n"
            f"gold_csv={gold_csv}\nfeature_npz={feature_npz}\n"
            f"gold sample: {gold_df['image_path'].head(3).tolist()}\n"
            f"feature sample: {feat_df['image_path'].head(3).tolist() if 'image_path' in feat_df.columns else 'N/A'}"
        )

    if len(merged) != len(gold_df):
        n_missing = len(gold_df) - len(merged)
        print(
            f"[merge_gold_and_features] Warning: of {len(gold_df)} gold CSV rows, "
            f"{n_missing} were not matched to embeddings and were excluded ({feature_npz})"
        )

    return merged


def _normalize_img_key(path: str) -> str:
    import re
    return re.sub(r"\.(jpg|jpeg|png)$", "", str(path), flags=re.IGNORECASE)


# =========================================================
# Models
# =========================================================
def build_models(seed: int = 42) -> Dict[str, object]:
    models = {
        "majority": DummyClassifier(strategy="most_frequent"),
        "random": DummyClassifier(strategy="stratified", random_state=seed),

        "logreg": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=seed,
            )),
        ]),

        "linear_svm": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(
                class_weight="balanced",
                max_iter=5000,
                random_state=seed,
            )),
        ]),

        "mlp": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(256,),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                batch_size=64,
                learning_rate_init=1e-3,
                max_iter=200,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                random_state=seed,
            )),
        ]),
    }
    return models


# =========================================================
# CV experiment
# =========================================================
def run_single_task_cv(
    df: pd.DataFrame,
    label_col: str,
    embedding_name: str,
    models: Dict[str, object],
    n_splits: int,
    seed: int,
    clip_embedding_npz: str,
):
    valid_df = df[df[label_col].notna()].copy()
    valid_df[label_col] = valid_df[label_col].astype(int)
    valid_df = valid_df.reset_index(drop=True)

    if valid_df.empty:
        raise ValueError(f"No valid rows for label column: {label_col}")

    feature_cols = [c for c in valid_df.columns if c.startswith("f_")]
    X = valid_df[feature_cols].values.astype(np.float32)
    y = valid_df[label_col].values.astype(int)
    image_paths = valid_df["image_path"].tolist()

    class_counts = pd.Series(y).value_counts().sort_index()
    min_class_count = int(class_counts.min())

    if min_class_count < n_splits:
        raise ValueError(
            f"Cannot run {n_splits}-fold stratified CV for {label_col}. "
            f"Minimum class count is {min_class_count}."
        )

    group_ids = build_group_ids(
        image_keys=[_normalize_img_key(p) for p in image_paths],
        clip_embedding_npz=clip_embedding_npz,
    )

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    rows = []
    split_records = []

    for fold_idx, (train_idx, test_idx) in enumerate(
        sgkf.split(X, y, groups=group_ids), start=1
    ):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        split_records.append({
            "fold": fold_idx,
            "train_ids": [image_paths[i] for i in train_idx],
            "test_ids": [image_paths[i] for i in test_idx],
            "class_distribution_train": pd.Series(y_train).value_counts().sort_index().to_dict(),
            "class_distribution_test": pd.Series(y_test).value_counts().sort_index().to_dict(),
        })

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            metrics = evaluate_predictions(y_test, y_pred)

            rows.append({
                "embedding": embedding_name,
                "classifier": model_name,
                "label": label_col,
                "fold": fold_idx,
                "n_total": int(len(valid_df)),
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                "qwk": metrics["qwk"],
                "macro_f1": metrics["macro_f1"],
                "balanced_acc": metrics["balanced_acc"],
            })

    meta = {
        "label": label_col,
        "n_splits": n_splits,
        "n_total": int(len(valid_df)),
        "n_groups": int(len(set(group_ids))),
        "class_distribution_total": class_counts.to_dict(),
        "folds": split_records,
    }

    return rows, meta


def aggregate_cv_results(results_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        results_df.groupby(["embedding", "classifier", "label"], as_index=False)
        .agg(
            qwk_mean=("qwk", "mean"),
            qwk_std=("qwk", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            balanced_acc_mean=("balanced_acc", "mean"),
            balanced_acc_std=("balanced_acc", "std"),
        )
    )
    return grouped


def run_all_experiments(
    gold_csv: str,
    embedding_map: Dict[str, str],
    output_dir: str,
    n_splits: int,
    seed: int,
):
    ensure_dir(output_dir)
    ensure_dir(os.path.join(output_dir, "cv_splits"))

    label_cols = [
        "identifiability_gold",
        "location_gold",
        "activity_gold",
    ]

    all_rows = []
    split_store = {}
    models = build_models(seed=seed)

    clip_embedding_npz = embedding_map["clip"]

    print("=" * 80)
    print("Privacy Sensitivity Baseline Evaluation (5-fold Grouped CV)")
    print("=" * 80)
    print(f"Gold CSV   : {gold_csv}")
    print(f"Output dir : {output_dir}")
    print(f"n_splits   : {n_splits}")
    print(f"Seed       : {seed}")
    print(f"Dedup ref  : {clip_embedding_npz} (CLIP embedding used for duplicate detection)")
    print("Embeddings :")
    for k, v in embedding_map.items():
        print(f"  - {k}: {v}")
    print("=" * 80)

    for embedding_name, feature_npz in embedding_map.items():
        print("=" * 80)
        print(f"[Embedding] {embedding_name}")
        print(f"Feature path: {feature_npz}")

        df = merge_gold_and_features(gold_csv, feature_npz)
        print(f"Merged rows : {len(df)}")

        for label_col in label_cols:
            print(f"  -> Running {n_splits}-fold grouped CV task: {label_col}")
            rows, split_meta = run_single_task_cv(
                df=df,
                label_col=label_col,
                embedding_name=embedding_name,
                models=models,
                n_splits=n_splits,
                seed=seed,
                clip_embedding_npz=clip_embedding_npz,
            )
            all_rows.extend(rows)

            if label_col not in split_store:
                split_store[label_col] = split_meta

    results_df = pd.DataFrame(all_rows)
    results_csv = os.path.join(output_dir, "all_fold_results.csv")
    results_df.to_csv(results_csv, index=False, encoding="utf-8-sig")

    agg_df = aggregate_cv_results(results_df)
    agg_csv = os.path.join(output_dir, "aggregated_results.csv")
    agg_df.to_csv(agg_csv, index=False, encoding="utf-8-sig")

    for label_col, meta in split_store.items():
        save_json(meta, os.path.join(output_dir, "cv_splits", f"{label_col}_cv_split.json"))

    # -------------------------
    # Main table (QWK mean±std)
    # -------------------------
    main_rows = agg_df[
        (agg_df["classifier"].isin(["majority", "random"])) |
        (agg_df["classifier"] == "logreg")
    ].copy()

    baseline_mask = main_rows["classifier"].isin(["majority", "random"])
    main_rows.loc[baseline_mask, "embedding"] = main_rows.loc[baseline_mask, "classifier"].map({
        "majority": "Majority",
        "random": "Random",
    })
    main_rows.loc[baseline_mask, "classifier"] = "—"

    main_rows["embedding"] = main_rows["embedding"].replace({
        "clip": "CLIP",
        "dinov2": "DINOv2",
        "places365": "Places365",
    })
    main_rows["classifier"] = main_rows["classifier"].replace({
        "logreg": "LogReg",
    })

    main_rows["qwk_mean_std"] = main_rows.apply(
        lambda r: f"{r['qwk_mean']:.3f} ± {r['qwk_std']:.3f}" if pd.notna(r["qwk_std"]) else f"{r['qwk_mean']:.3f}",
        axis=1,
    )

    main_pivot = main_rows.pivot_table(
        index=["embedding", "classifier"],
        columns="label",
        values="qwk_mean_std",
        aggfunc="first",
    ).reset_index()

    main_pivot = main_pivot.rename(columns={
        "identifiability_gold": "Identifiability_QWK",
        "location_gold": "Location_QWK",
        "activity_gold": "Activity_QWK",
    })

    main_csv = os.path.join(output_dir, "main_table_qwk_cv.csv")
    main_pivot.to_csv(main_csv, index=False, encoding="utf-8-sig")

    # -------------------------
    # Appendix tables
    # -------------------------
    appendix_dir = os.path.join(output_dir, "appendix_tables")
    ensure_dir(appendix_dir)

    for label_col in label_cols:
        sub = agg_df[
            (agg_df["label"] == label_col)
            & (agg_df["embedding"].isin(["clip", "dinov2", "places365"]))
            & (agg_df["classifier"].isin(["logreg", "linear_svm", "mlp"]))
        ].copy()

        sub["embedding"] = sub["embedding"].replace({
            "clip": "CLIP",
            "dinov2": "DINOv2",
            "places365": "Places365",
        })
        sub["classifier"] = sub["classifier"].replace({
            "logreg": "LogReg",
            "linear_svm": "LinearSVM",
            "mlp": "MLP",
        })

        sub["qwk_mean_std"] = sub.apply(
            lambda r: f"{r['qwk_mean']:.3f} ± {r['qwk_std']:.3f}" if pd.notna(r["qwk_std"]) else f"{r['qwk_mean']:.3f}",
            axis=1,
        )

        pivot = sub.pivot_table(
            index="embedding",
            columns="classifier",
            values="qwk_mean_std",
            aggfunc="first",
        ).reset_index()

        out_path = os.path.join(appendix_dir, f"{label_col}_classifier_robustness_qwk_cv.csv")
        pivot.to_csv(out_path, index=False, encoding="utf-8-sig")

    # -------------------------
    # Full metric tables
    # -------------------------
    for metric_prefix in ["qwk", "macro_f1", "balanced_acc"]:
        temp = agg_df.copy()
        temp[f"{metric_prefix}_mean_std"] = temp.apply(
            lambda r: f"{r[f'{metric_prefix}_mean']:.3f} ± {r[f'{metric_prefix}_std']:.3f}"
            if pd.notna(r[f"{metric_prefix}_std"]) else f"{r[f'{metric_prefix}_mean']:.3f}",
            axis=1,
        )

        pivot = temp.pivot_table(
            index=["embedding", "classifier"],
            columns="label",
            values=f"{metric_prefix}_mean_std",
            aggfunc="first",
        ).reset_index()

        pivot.to_csv(
            os.path.join(output_dir, f"full_table_{metric_prefix}_cv.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    summary = {
        "gold_csv": gold_csv,
        "embedding_map": embedding_map,
        "n_splits": n_splits,
        "seed": seed,
        "cv_protocol": "StratifiedGroupKFold (duplicate-safe, grouped by CLIP embedding identity)",
        "num_fold_rows": int(len(results_df)),
        "num_aggregated_rows": int(len(agg_df)),
        "saved_files": {
            "all_fold_results": results_csv,
            "aggregated_results": agg_csv,
            "main_table_qwk_cv": main_csv,
            "appendix_dir": appendix_dir,
        },
    }
    save_json(summary, os.path.join(output_dir, "run_summary.json"))

    print("\nDone.")
    print(f"Saved fold results : {results_csv}")
    print(f"Saved aggregated   : {agg_csv}")
    print(f"Saved main table   : {main_csv}")
    print(f"Saved appendix dir : {appendix_dir}")


# =========================================================
# Main
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run privacy sensitivity baselines with 5-fold grouped CV."
    )

    parser.add_argument(
        "--gold_csv",
        type=str,
        default="final_merged_annotation_table.csv",
        help="Path to merged gold annotation CSV.",
    )
    parser.add_argument(
        "--clip_npz",
        type=str,
        default="features/clip/image_features.npz",
        help="Path to CLIP feature NPZ.",
    )
    parser.add_argument(
        "--dinov2_npz",
        type=str,
        default="features/dinov2_vitl14/image_features.npz",
        help="Path to DINOv2 feature NPZ.",
    )
    parser.add_argument(
        "--places365_npz",
        type=str,
        default="features/places365/image_features.npz",
        help="Path to Places365 feature NPZ.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results_baselines_cv_v2",
        help="Directory to save results.",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="Number of CV folds.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    embedding_map = {
        "clip": args.clip_npz,
        "dinov2": args.dinov2_npz,
        "places365": args.places365_npz,
    }

    run_all_experiments(
        gold_csv=args.gold_csv,
        embedding_map=embedding_map,
        output_dir=args.output_dir,
        n_splits=args.n_splits,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
