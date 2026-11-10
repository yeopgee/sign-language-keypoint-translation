"""Fit pose/hand normalization on training JSON and apply it to both splits."""

import argparse
import json
from pathlib import Path

import numpy as np


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def fit_stats(train_dir):
    values = {'pose': [], 'hand_left': [], 'hand_right': []}
    for path in sorted(Path(train_dir).glob('*.json')):
        for frame in read_json(path)['frames']:
            for key in values:
                if frame.get(key):
                    values[key].append(frame[key])
    pose = np.asarray(values['pose'], dtype=float)
    left = np.asarray(values['hand_left'], dtype=float)
    right = np.asarray(values['hand_right'], dtype=float)
    if min(len(pose), len(left), len(right)) == 0:
        raise ValueError('Training data must contain pose, left-hand, and right-hand keypoints.')
    return dict(pose_mu=pose.mean(0), pose_sd=pose.std(0),
                left_min=left.min(0), left_max=left.max(0),
                right_min=right.min(0), right_max=right.max(0))


def normalize_split(source, destination, stats):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(Path(source).glob('*.json')):
        data = read_json(path)
        for frame in data['frames']:
            if frame.get('pose'):
                frame['pose'] = ((np.asarray(frame['pose']) - stats['pose_mu']) / (stats['pose_sd'] + 1e-8)).tolist()
            for hand in ('left', 'right'):
                key = f'hand_{hand}'
                if frame.get(key):
                    minimum, maximum = stats[f'{hand}_min'], stats[f'{hand}_max']
                    frame[key] = ((np.asarray(frame[key]) - minimum) / (maximum - minimum + 1e-8) - 0.5).tolist()
        (destination / path.name).write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-dir', required=True)
    parser.add_argument('--val-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stats = fit_stats(args.train_dir)
    np.savez(output / 'normalization.npz', **stats)
    normalize_split(args.train_dir, output / 'train', stats)
    normalize_split(args.val_dir, output / 'val', stats)


if __name__ == '__main__':
    main()
