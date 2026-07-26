"""
dedup_fold_utils.py

CLIP 임베딩을 이용해 완전 동일(bit-identical) 이미지를 탐지하고,
cross-validation fold 배정 시 중복 이미지가 서로 다른 fold에 나뉘지
않도록(train/test leakage 방지) group id를 생성하는 유틸리티.

사용법:
    from dedup_fold_utils import build_group_ids

    group_ids = build_group_ids(
        image_keys=valid_df["img_key"].tolist(),
        clip_embedding_npz="clip_vitb16_embeddings.npz",
    )
    # group_ids[i] : valid_df의 i번째 행이 속한 그룹 id (중복 이미지는 같은 id 공유)

    from sklearn.model_selection import StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for train_idx, test_idx in sgkf.split(X, y, groups=group_ids):
        ...
"""

import re
from collections import defaultdict

import numpy as np


def _normalize_key(name: str) -> str:
    """파일 확장자를 제거해서 'img123.jpg' -> 'img123' 형태로 통일."""
    name = str(name)
    return re.sub(r"\.(jpg|jpeg|png)$", "", name, flags=re.IGNORECASE)


def find_exact_duplicate_groups(clip_embedding_npz: str):
    """
    CLIP 임베딩 파일에서 bit-identical한 벡터들을 그룹핑한다.

    Returns
    -------
    key_to_group : dict[str, int]
        정규화된 이미지 키 -> 그룹 id 매핑.
        중복이 없는 이미지도 자기 자신만 속한 고유 그룹 id를 가진다.
    n_duplicate_groups : int
        2장 이상이 뭉친 실제 중복 그룹 개수 (로그/검증용).
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
    주어진 image_keys 순서에 맞춰 group id 배열을 만든다.
    CLIP 임베딩에서 발견되지 않는 키(예외 상황)는 각자 고유 그룹으로 취급한다.

    Parameters
    ----------
    image_keys : list[str]
        StratifiedGroupKFold에 넣을 X, y와 같은 순서의 이미지 식별자
        (확장자 유무는 자동으로 정규화됨).
    clip_embedding_npz : str
        중복 탐지에 사용할 CLIP 임베딩 파일 경로.

    Returns
    -------
    np.ndarray
        image_keys와 같은 길이의 정수 group id 배열.
    """
    key_to_group, n_duplicate_groups = find_exact_duplicate_groups(clip_embedding_npz)

    print(
        f"[dedup_fold_utils] CLIP 임베딩에서 발견된 완전 동일 중복 그룹: "
        f"{n_duplicate_groups}개"
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
            # CLIP 임베딩에 없는 이미지는 각자 고유 그룹으로 처리
            group_ids.append(next_fallback_group)
            next_fallback_group += 1
            n_fallback += 1

    if n_fallback > 0:
        print(
            f"[dedup_fold_utils] 경고: CLIP 임베딩에서 매칭되지 않은 이미지 "
            f"{n_fallback}개는 각자 고유 그룹으로 처리됨."
        )

    return np.array(group_ids)


def summarize_group_sizes(group_ids: np.ndarray):
    """디버그용: 그룹 크기 분포 요약."""
    unique, counts = np.unique(group_ids, return_counts=True)
    size_distribution = defaultdict(int)
    for c in counts:
        size_distribution[int(c)] += 1
    return dict(sorted(size_distribution.items()))