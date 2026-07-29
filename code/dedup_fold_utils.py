"""
dedup_fold_utils.py

Utility for detecting bit-identical (exact-duplicate) images using CLIP
embeddings and generating group ids so that duplicate images are not split
across different folds during cross-validation (preventing train/test leakage).

Usage:
    from dedup_fold_utils import build_group_ids

    group_ids = build_group_ids(
        image_keys=valid_df["img_key"].tolist(),
        clip_embedding_npz="clip_vitb16_embeddings.npz",
    )
    # group_ids[i] : the group id of the i-th row in valid_df
    #                (duplicate images share the same id)

    from sklearn.model_selection import StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for train_idx, test_idx in sgkf.split(X, y, groups=group_ids):
        ...
"""

import re
from collections import defaultdict

import numpy as np


def _normalize_key(name: str) -> str:
    """Remove the file extension to normalize 'img123.jpg' -> 'img123'."""
    name = str(name)
    return re.sub(r"\.(jpg|jpeg|png)$", "", name, flags=re.IGNORECASE)


def find_exact_duplicate_groups(clip_embedding_npz: str):
    """
    Group bit-identical vectors in the CLIP embedding file.

    Returns
    -------
    key_to_group : dict[str, int]
        Mapping from normalized image key -> group id.
        An image with no duplicates forms its own singleton group.
    n_duplicate_groups : int
        Number of actual duplicate groups containing two or more images
        (for logging/verification).
    """
    data = np.load(clip_embedding_npz, allow_pickle=True)
    embeddings = data["embeddings"].astype(np.float32)
    image_ids = data["image_ids"]

    hash_to_indices = defaultdict(list)
    for i in range(len(embeddings)):
        hash_to_indices[embeddings[i].tobytes()].append(i)

    key_to_group = {}
    group_counter = 0
    n_duplicate_groups = 0

    for indices in hash_to_indices.values():
        if len(indices) > 1:
            n_duplicate_groups += 1
        for idx in indices:
            key = _normalize_key(image_ids[idx])
            key_to_group[key] = group_counter
        group_counter += 1

    return key_to_group, n_duplicate_groups


def build_group_ids(image_keys, clip_embedding_npz: str):
    """
    Build a group id array aligned with the given image_keys order.
    Keys not found in the CLIP embeddings (edge cases) are each treated
    as their own unique group.

    Parameters
    ----------
    image_keys : list[str]
        Image identifiers in the same order as the X, y passed to
        StratifiedGroupKFold (extensions are normalized automatically).
    clip_embedding_npz : str
        Path to the CLIP embedding file used for duplicate detection.

    Returns
    -------
    np.ndarray
        Integer group id array with the same length as image_keys.
    """
    key_to_group, n_duplicate_groups = find_exact_duplicate_groups(clip_embedding_npz)

    print(
        f"[dedup_fold_utils] Exact-duplicate groups found in CLIP embeddings: "
        f"{n_duplicate_groups}"
    )

    max_existing_group = max(key_to_group.values(), default=-1)
    next_fallback_group = max_existing_group + 1

    group_ids = []
    n_fallback = 0
    for key in image_keys:
        norm_key = _normalize_key(key)
        if norm_key in key_to_group:
            group_ids.append(key_to_group[norm_key])
        else:
            # Images not present in the CLIP embeddings are each assigned
            # their own unique group.
            group_ids.append(next_fallback_group)
            next_fallback_group += 1
            n_fallback += 1

    if n_fallback > 0:
        print(
            f"[dedup_fold_utils] Warning: {n_fallback} image(s) not matched in the "
            f"CLIP embeddings were each assigned their own unique group."
        )

    return np.array(group_ids)


def summarize_group_sizes(group_ids: np.ndarray):
    """Debug helper: summarize the distribution of group sizes."""
    unique, counts = np.unique(group_ids, return_counts=True)
    size_distribution = defaultdict(int)
    for c in counts:
        size_distribution[int(c)] += 1
    return dict(sorted(size_distribution.items()))
