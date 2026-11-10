# Keypoint-based Korean Sign Language to text

This repository presents my sign-to-text contribution to [Cheongseol, a KUBIG team project on two-way Korean Sign Language translation](https://github.com/KU-BIG/KUBIG_2025_SPRING/tree/main/KUBIG%20CONFERENCE/Image%20processing). The team's full project also includes text-to-sign work. This repository covers the keypoint-to-Korean-text direction that I developed in my local research notebook.

## My contribution

- Merged pose and hand keypoints with Korean sentence annotations, then split, normalized, and augmented the sequences.
- Explored masking of location and time expressions in the target sentences and prepared SentencePiece token sequences.
- Implemented a bidirectional GRU encoder, attention-based GRU decoder, and the related training and prediction code.

The [team README](https://github.com/KU-BIG/KUBIG_2025_SPRING/blob/main/KUBIG%20CONFERENCE/Image%20processing/readme.md) describes the complete collaborative system and its reported results. The code here is organized from my own working files; it does not claim authorship of the team's text-to-sign component.

## Code map

| File | Purpose |
| --- | --- |
| `merge_keypoints.py` | Merge source XML keypoints and JSON sentence annotations |
| `split_samples.py` | Split by sentence ID before normalization |
| `normalize_keypoints.py` | Fit training-set normalization and apply it to both splits |
| `mask_text.py` | Mask selected Korean location and time expressions |
| `augment_frames.py` | Create fixed-length skip-sampled training sequences |
| `prepare_arrays.py` | Train SentencePiece and save paired keypoint/token arrays |
| `dataset.py` | Load and pad variable-length sequences |
| `model.py` | BiGRU encoder, attention, decoder, and sequence-to-sequence model |
| `train.py` | Train the model and save a checkpoint |
| `predict.py` | Decode Korean text from a prepared keypoint array |
| `notebooks/sign_to_text.ipynb` | Cleaned original research notebook |

## Data and workflow

The original work used the AI-Hub sign-language dataset described by the team. The dataset and trained checkpoints are not redistributed here. The scripts accept explicit paths; the expected keypoint feature width is 108.

With the source XML and JSON directories available, the stages are:

```bash
python merge_keypoints.py --xml-root DATA/xml --json-root DATA/annotations --output-dir work/merged
python split_samples.py --input-dir work/merged --output-dir work/split
python normalize_keypoints.py --train-dir work/split/train --val-dir work/split/val --output-dir work/normalized
python mask_text.py work/normalized/train
python mask_text.py work/normalized/val
python augment_frames.py --input-dir work/normalized/train --output-dir work/augmented
python prepare_arrays.py --train-dir work/augmented --val-dir work/normalized/val --output-dir work/processed
python train.py --data-dir work/processed --output checkpoints/best_model.pt
python predict.py --keypoints sample.npy --tokenizer work/processed/spm.model --checkpoint checkpoints/best_model.pt
```

The reorganized scripts make each stage easier to inspect. Exact dataset layout, source naming, and the original experimental choices are documented in the notebook and the team repository.
