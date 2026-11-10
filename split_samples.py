"""Split merged annotations by sentence ID to avoid sharing sentences across splits."""

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path


def sentence_id(path):
    parts = path.stem.split('_')
    if len(parts) < 4:
        raise ValueError(f'Cannot parse sentence ID from {path.name}')
    return parts[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--val-sentences', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    groups = defaultdict(list)
    for path in sorted(Path(args.input_dir).glob('*.json')):
        groups[sentence_id(path)].append(path)
    if not groups:
        raise ValueError('No merged JSON files found.')
    rng = random.Random(args.seed)
    val_ids = set(rng.sample(sorted(groups), min(args.val_sentences, len(groups))))
    output = Path(args.output_dir)
    for group, paths in groups.items():
        destination = output / ('val' if group in val_ids else 'train')
        destination.mkdir(parents=True, exist_ok=True)
        for path in paths:
            shutil.copy2(path, destination / path.name)
    print(f'{len(groups)} sentence groups; {len(val_ids)} assigned to validation.')


if __name__ == '__main__':
    main()
