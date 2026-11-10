# Korean Sign Language to text from human keypoints

This repository contains my sign-to-text work for [Cheongseol, a KUBIG project on two-way Korean Sign Language translation](https://github.com/KU-BIG/KUBIG_2025_SPRING/tree/main/KUBIG%20CONFERENCE/Image%20processing). The broader team also explored text-to-sign generation; the code here covers the keypoint-to-Korean-text direction.

## Motivation

Full video contains background, lighting, and other information that a translation model may not need. We used pose and hand keypoints as a more compact representation. The data still posed a major challenge: many target sentences contain specific times, locations, and addresses that appear rarely in training.

## My contribution

- Matched keypoint XML files with Korean sentence annotations, selected pose and hand coordinates, and split data by sentence ID.
- Normalized keypoints, augmented sequences through temporal sampling, and replaced selected time, location, and address expressions with placeholder tokens.
- Implemented a BiGRU encoder, attention-based GRU decoder, training pipeline, and prediction code for sign-to-text translation.

```mermaid
flowchart LR
    A[Sign video keypoints] --> B[Match with Korean text]
    B --> C[Select and normalize pose and hands]
    C --> D[Temporal sampling]
    D --> E[BiGRU encoder]
    E --> F[Attention + GRU decoder]
    F --> G[Korean text with placeholder tokens]
```

## Results

The initial sign-to-text model struggled with the many rare expressions in the target sentences. After the team's normalization, masking, and decoding experiments, the best reported configuration reached **BLEU 0.3254** and **METEOR 0.4900**.

![BLEU and METEOR across four sign-to-text configurations](assets/sign2text-metrics.svg)

*Scores from my KUBIG presentation; the chart compares four sign-to-text configurations.*

These numbers describe the reported sign-to-text experiment. The text-to-sign component and its results belong to the broader team project.

## Discussion

The project showed why text preparation mattered as much as model architecture for this dataset. Its fire-alert domain was narrow, and rare names remained difficult even after masking. A video demonstration also required MediaPipe keypoints, whereas the training data used OpenPose-derived keypoints, creating an additional mismatch.

## Code and usage

| Stage | Files |
| --- | --- |
| Match and split samples | [`merge_keypoints.py`](merge_keypoints.py), [`split_samples.py`](split_samples.py) |
| Normalize, mask, and augment | [`normalize_keypoints.py`](normalize_keypoints.py), [`mask_text.py`](mask_text.py), [`augment_frames.py`](augment_frames.py) |
| Prepare arrays and train | [`prepare_arrays.py`](prepare_arrays.py), [`model.py`](model.py), [`train.py`](train.py) |
| Predict and inspect | [`predict.py`](predict.py), [`notebooks/sign_to_text.ipynb`](notebooks/sign_to_text.ipynb) |

The scripts accept explicit input paths. With the AI-Hub data available and dependencies installed from [`requirements.txt`](requirements.txt), run the stages in the order shown in the table. The dataset and trained checkpoints are not included here.
