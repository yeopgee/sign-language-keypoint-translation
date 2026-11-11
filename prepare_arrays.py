"""Train a SentencePiece tokenizer and convert normalized JSON to paired arrays."""

import argparse
import json
from pathlib import Path

import numpy as np
import sentencepiece as spm


SYMBOLS = ['<건물>', '<지역>', '<시간>', '<도로>', '<주소>', '<학교>', '<산>', '<공원>', '<역>', '<숙박시설>']


def samples(directory):
    for path in sorted(Path(directory).glob('*.json')):
        data = json.loads(path.read_text(encoding='utf-8'))
        text = data.get('masked_text') or data.get('korean_text', '')
        frames = []
        for frame in data.get('frames', []):
            vector = frame.get('pose', []) + frame.get('hand_left', []) + frame.get('hand_right', [])
            if len(vector) != 108:
                frames = []
                break
            frames.append(vector)
        if text.strip() and frames:
            yield np.asarray(frames, dtype=np.float32), text.strip(), path.stem


def save_split(source, destination, tokenizer, part_size=10_000):
    destination.mkdir(parents=True, exist_ok=True)
    xs, ys, file_ids = [], [], []
    total = part = 0

    def save_part():
        # Preserve sample IDs so masked predictions can be restored to their
        # original time and location expressions, as in the notebook.
        np.save(destination / f'X_part_{part}.npy', np.asarray(xs, dtype=object))
        np.save(destination / f'y_part_{part}.npy', np.asarray(ys, dtype=object))
        np.save(destination / f'file_id_part_{part}.npy', np.asarray(file_ids))

    for x, text, file_id in samples(source):
        xs.append(x)
        ys.append([tokenizer.bos_id(), *tokenizer.encode(text, out_type=int), tokenizer.eos_id()])
        file_ids.append(file_id)
        total += 1
        if len(xs) == part_size:
            save_part()
            part += 1
            xs, ys, file_ids = [], [], []
    if xs:
        save_part()
    if not total:
        raise ValueError(f'No valid samples in {source}')
    print(f'{destination}: {total} samples')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-dir', required=True)
    parser.add_argument('--val-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--vocab-size', type=int, default=1000)
    parser.add_argument('--part-size', type=int, default=10_000)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    text_file = output / 'train_text.txt'
    text_file.write_text('\n'.join(text for _, text, _ in samples(args.train_dir)) + '\n', encoding='utf-8')
    spm.SentencePieceTrainer.train(input=str(text_file), model_prefix=str(output / 'spm'),
        vocab_size=args.vocab_size, character_coverage=0.9995, model_type='bpe',
        unk_id=0, bos_id=1, eos_id=2, pad_id=3, pad_piece='<pad>',
        user_defined_symbols=SYMBOLS)
    tokenizer = spm.SentencePieceProcessor(model_file=str(output / 'spm.model'))
    save_split(Path(args.train_dir), output / 'train', tokenizer, args.part_size)
    save_split(Path(args.val_dir), output / 'val', tokenizer, args.part_size)


if __name__ == '__main__':
    main()
