# Beyond Identifiability: A Multidimensional Dataset for Human-Perceived Privacy Sensitivity in Social Media Images

This repository contains annotation labels and pre-computed image embeddings for the accompanying submitted AAAI-27 paper.

The dataset provides annotations for 4,999 Korean social media images along three distinct ordinal privacy sensitivity dimensions: identifiability, location sensitivity, and activity sensitivity. It also includes privacy-relevant information types, blur presence, sharing allowance judgments, and image category labels.

Due to the sensitive nature of the image content, we do not release the original images. Instead, we release annotation labels and pre-computed image embeddings (CLIP ViT-B/16, DINOv2 ViT-L/14, and Places365 ResNet-50). For reviewer reference only, this repository also includes 10 anonymized sample images with corresponding annotations.

This repository has been prepared for double-anonymous review. Author names, affiliations, contact information, personal repository links, and institution-specific paths are omitted.

## Repository Contents

```text
├── README.md
├── privacy_sensitivity_annotation.csv     # 4,999 annotation records
├── embeddings/
│   ├── clip_vitb16_embeddings.npz         # CLIP ViT-B/16 embeddings, shape (4999, 512)
│   ├── dinov2_vitl14_embeddings.npz       # DINOv2 ViT-L/14 embeddings, shape (4999, 1024)
│   └── places365_resnet50_embeddings.npz  # Places365 ResNet-50 embeddings, shape (4999, 2048)
├── code/
│   ├── train_frozen.py                    # frozen-representation classifiers + baselines (Table 3)
│   ├── dedup_fold_utils.py                # exact-duplicate detection for group-aware CV
│   ├── clip_finetune.py                   # CLIP fine-tuning (Table 4)
│   ├── densenet_finetune.py               # DenseNet-121 fine-tuning (Table 4)
│   └── efficientnet_finetune.py           # EfficientNet-B0 fine-tuning (Table 4)
├── requirements.txt                       # Python dependencies
├── annotator1_raw.csv                     # raw labels from annotator 1
├── annotator2_raw.csv                     # raw labels from annotator 2
├── sample_annotation.csv                  # 10 sample annotations for reviewer reference
└── sample_images/                         # 10 anonymized sample images for reviewer reference
```

## Data

**TL;DR:** The annotation table is `privacy_sensitivity_annotation.csv`, and pre-computed image embeddings are in `embeddings/`. Match records by `image_path`.

### Annotation Table

Each row in `privacy_sensitivity_annotation.csv` corresponds to one image record.

The columns are:

```text
image_path              : Image identifier matching image_ids in the embeddings file.
information_type        : 8-dimensional binary vector of privacy-relevant information types.
identifiability_gold    : Ordinal score (0–2) for individual identifiability.
location_gold           : Ordinal score (0–2) for privacy harm of disclosing the depicted location.
activity_gold           : Ordinal score (0–2) for privacy harm of the depicted or implied activity.
blur_presence           : Binary label for the presence of intentional privacy-protective modification.
sharing_gold            : 6-dimensional binary vector of audience tiers with whom sharing is appropriate.
high_level_category     : High-level retrieval category used during data collection.
subcategory             : More specific retrieval subcategory.
```

The `information_type` and `sharing_gold` fields are stored as string-encoded
integer lists (e.g., `"[1,0,1,0,0,0,0,0]"`) and can be parsed with
`ast.literal_eval`.
 
The `information_type` vector follows this order:
1. personal identity and sensitive information
2. social relations
3. location information
4. body and appearance
5. personal preferences
6. socio-cultural sensitive information
7. risk- and crime-related information
8. other
The `sharing_gold` vector follows this order:
1. no sharing (not shared with anyone)
2. close relations (e.g., family, partner)
3. general relations (e.g., friends, colleagues)
4. acquaintances (people one knows but is not in frequent contact with)
5. public (openly visible to anyone)
6. broadcast media (very wide disclosure through mass media)
`high_level_category` and `subcategory` describe the hashtag-retrieval strata
used during data collection and are **not** used as model inputs or prediction
targets.

### Image Embeddings

The `embeddings/` directory contains three pre-computed embedding files, each with two arrays:
```text
image_ids   : shape (4999,), image identifiers matching the image_path column.
embeddings  : image embeddings.
```
| File | Model | Embedding dim |
|------|-------|---------------|
| clip_vitb16_embeddings.npz | CLIP ViT-B/16 | 512 |
| dinov2_vitl14_embeddings.npz | DINOv2 ViT-L/14 | 1024 |
| places365_resnet50_embeddings.npz | Places365 ResNet-50 | 2048 |

The `i`-th row of `embeddings` corresponds to `image_ids[i]`. Image identifiers are matched after removing file extensions (e.g., `img1` and `img1.jpg` are treated as the same).

### Individual Annotator Labels

`annotator1_raw.csv` and `annotator2_raw.csv` contain the raw labels from each of the two annotators before aggregation. They are provided to support the aggregation-robustness analysis (Appendix B) and are not required for the main experiments. The columns are `image_path`, `information_type`, `identifiability`, `activity_sensitivity`, `location_sensitivity`, `blur_presence`, and `sharing_allowance`.

### Sample Images for Reviewer Reference

To help reviewers understand the annotation schema, we include 10 representative sample images in `sample_images/` along with their annotations in `sample_annotation.csv`.

The samples illustrate diverse combinations of sensitivity scores across the three dimensions.

Visible faces and identifying cues in these samples have been anonymized where applicable. These images are provided solely for reviewer verification and are not part of the publicly released dataset.

The columns of `sample_annotation.csv` follow the same schema as `privacy_sensitivity_annotation.csv`.

## Setup and Reproduction

Install dependencies:
```bash
pip install -r requirements.txt
```

**Frozen-representation experiments (Table 3) and baselines** — fully reproducible with the provided embeddings:
```bash
python code/train_frozen.py \
    --gold_csv privacy_sensitivity_annotation.csv \
    --clip_npz embeddings/clip_vitb16_embeddings.npz \
    --dinov2_npz embeddings/dinov2_vitl14_embeddings.npz \
    --places365_npz embeddings/places365_resnet50_embeddings.npz \
    --seed 42
```
**Fine-tuning experiments (Table 4)** — these scripts require the original images (`--image_dir`), which are **not released** due to the sensitive 
nature of the content. They are provided for transparency but **cannot be run with the released files alone**:

```bash
python code/clip_finetune.py \
    --gold_csv privacy_sensitivity_annotation.csv \
    --image_dir <path_to_images> \
    --clip_embedding_npz embeddings/clip_vitb16_embeddings.npz
```
(`densenet_finetune.py` and `efficientnet_finetune.py` follow the same interface.)

All experiments use 5-fold cross-validation with group-aware splitting and a fixed seed (42). The `clip_embedding_npz` argument in fine-tuning is used only for exact-duplicate detection during fold assignment.

## Environment

Experiments were run with Python 3.11, PyTorch 2.10.0 (CUDA 12.8), torchvision 0.25.0, transformers 5.6.2, and scikit-learn 1.8.0, on an NVIDIA RTX 4090 GPU. See `requirements.txt` for dependencies.

## Data Release Policy

The complete annotation table and pre-computed embeddings for all 4,999 records are included in this repository. The original images are not released because they may contain personally sensitive information. Ten anonymized sample images are provided solely for reviewer verification. Consequently, the frozen-representation results and baselines are fully reproducible from the released files, whereas the fine-tuning results require the original images and cannot be reproduced from the released files alone.

## Ethical Use

Users should not attempt to reconstruct, identify, or infer the original images or individuals from the released embeddings or annotations.
Users should not use this dataset for surveillance, re-identification, profiling, or other applications that may increase privacy risks.

## Anonymous Review

This repository is provided for double-anonymous review. Author names, affiliations, contact information, personal repository links, and institution-specific paths are removed. Citation and license information will be added after publication.
