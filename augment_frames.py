"""Create fixed-length keypoint sequences by skip-sampling training frames."""

import argparse
import copy
import json
import random
from pathlib import Path


def sample_indices(length, target, rng):
    if length <= 0:
        raise ValueError('A sequence must contain at least one frame.')
    if length < target:
        return list(range(length)) + [length - 1] * (target - length)
    # The baseline grid and per-frame noise follow the notebook's final
    # 100-frame skip-sampling experiment.
    interval = length // (target - 1)
    offset = (length - interval * (target - 1)) // 2
    baseline = [offset + interval * i for i in range(target)]
    if baseline[-1] >= length:
        baseline[-1] = length - 1
    if interval == 1:
        return [min(length - 1, max(0, index)) for index in baseline]

    indices = []
    for i, base in enumerate(baseline):
        if base >= length - 1:
            noise = 0
        else:
            noise_max = max(1, length - 1 - base) if i == target - 1 else max(1, interval)
            noise = rng.randint(0, noise_max - 1)
        indices.append(min(base + noise, length - 1))
    return indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--frames', type=int, default=100)
    parser.add_argument('--copies', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in sorted(Path(args.input_dir).glob('*.json')):
        data = json.loads(path.read_text(encoding='utf-8'))
        frames = data.get('frames', [])
        if not frames:
            continue
        copies = 1 if len(frames) // (args.frames - 1) <= 1 else args.copies
        for copy_id in range(1, copies + 1):
            result = copy.deepcopy(data)
            result['frames'] = [copy.deepcopy(frames[i]) for i in sample_indices(len(frames), args.frames, rng)]
            for i, frame in enumerate(result['frames']):
                frame['frame_idx'] = i
            (output / f'{path.stem}_aug{copy_id}.json').write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')
            count += 1
    print(f'Wrote {count} augmented sequences.')


if __name__ == '__main__':
    main()
