# Privacy Sensitivity Annotation Dataset for Social Media Images

This repository contains annotation labels and pre-computed image embeddings for the accompanying submitted AAAI AIES'26 paper.

The dataset provides annotations for 4,999 Korean social media images along three independent ordinal privacy sensitivity dimensions: identifiability, location sensitivity, and activity sensitivity. It also includes privacy-relevant information types, blur presence, sharing allowance judgments, and image category labels.

Due to the sensitive nature of the image content, we do not release the original images. Instead, we release annotation labels and pre-computed CLIP ViT-B/16 image embeddings. For reviewer reference only, this repository also includes 10 anonymized sample images with corresponding annotations.

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
│   ├── dedup_fold_utils.py                # near-duplicate detection for group-aware CV
│   ├── clip_finetune.py                   # CLIP fine-tuning (Table 4)
│   ├── densenet_finetune.py               # DenseNet-121 fine-tuning (Table 4)
│   └── efficientnet_finetune.py           # EfficientNet-B0 fine-tuning (Table 4)
├── requirements.txt                       # Python dependencies
├── sample_annotation.csv                  # 10 sample annotations for reviewer reference
└── sample_images/                         # 10 anonymized sample images for reviewer reference
```

## Data

**TL;DR:** The annotation table is `privacy_sensitivity_annotation.csv`, and the image embeddings are `clip_vitb16_embeddings.npz`. Match records by `image_id`.

### Annotation Table

Each row in `privacy_sensitivity_annotation.csv` corresponds to one image record.

The columns are:

```text
image_id              : Image identifier matching image_ids in the embeddings file.
information_type      : 8-dimensional binary vector of privacy-relevant information types.
identifiability       : Ordinal score (0–2) for individual identifiability.
location_sensitivity  : Ordinal score (0–2) for privacy harm of disclosing the depicted location.
activity_sensitivity  : Ordinal score (0–2) for privacy harm of the depicted or implied activity.
blur_presence         : Binary label for the presence of intentional privacy-protective modification.
sharing_allowance     : 6-dimensional binary vector of audience tiers with whom sharing is appropriate.
high_level_category   : High-level retrieval category used during data collection.
subcategory           : More specific retrieval subcategory.
```

### Image Embeddings

`clip_vitb16_embeddings.npz` contains two arrays:

```text
image_ids   : shape (4999,), image identifiers matching the image_id column.
embeddings  : shape (4999, 512), CLIP ViT-B/16 image embeddings.
```

The `i`-th row of `embeddings` corresponds to `image_ids[i]`.

### Sample Images for Reviewer Reference

To help reviewers understand the annotation schema, we include 10 representative sample images in `sample_images/` along with their annotations in `sample_annotation.csv`.

The samples illustrate diverse combinations of sensitivity scores across the three dimensions.

Visible faces and identifying cues in these samples have been anonymized where applicable. These images are provided solely for reviewer verification and are not part of the publicly released dataset.

The columns of `sample_annotation.csv` follow the same schema as `privacy_sensitivity_annotation.csv`.


## Data Release Policy

Since the original images may contain personally sensitive information, the original images are not publicly released. For review purposes, we provide image embeddings and annotation labels for a limited subset of samples. The full dataset will be shared in the form of image embeddings upon request.

## Ethical Use

Users should not attempt to reconstruct, identify, or infer the original images or individuals from the released embeddings or annotations.
Users should not use this dataset for surveillance, re-identification, profiling, or other applications that may increase privacy risks.

## Anonymous Review

This repository is provided for double-anonymous review. Author names, affiliations, contact information, personal repository links, and institution-specific paths are removed. Citation and license information will be added after publication.
