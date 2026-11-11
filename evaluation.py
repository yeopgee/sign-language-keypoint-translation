"""Evaluate sign-to-text checkpoints with the notebook's decoding methods."""

import argparse
import json
from pathlib import Path

import evaluate as hf_evaluate
import numpy as np
import pandas as pd
import sentencepiece as spm
import torch
from torch.nn import functional as F

from dataset import PAD_ID, make_loader
from model import Decoder, Encoder, Seq2Seq


def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float('Inf')):
    """Filter logits with the notebook's top-k and nucleus-sampling rules."""
    logits = logits.clone()
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[remove] = filter_value
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = 0
        logits[sorted_indices[remove]] = filter_value
    return logits


def sample_decode(model, input_tensor, tokenizer, device, max_len=100,
                  eos_id=2, repetition_penalty=1.2, top_k=5, top_p=0.9):
    """Sample one sentence using the notebook's repetition, top-k, and top-p rules."""
    model.eval()
    input_tensor = input_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(input_tensor)
        hidden = torch.tanh(hidden[0] + hidden[1]).unsqueeze(0)
        input_token = torch.tensor([tokenizer.bos_id()], device=device)
        decoded, used_tokens = [], set()
        for _ in range(max_len):
            output, hidden, _ = model.decoder(input_token, hidden, encoder_outputs)
            output = output.squeeze(0)
            for token_id in used_tokens:
                if output[token_id] < 0:
                    output[token_id] *= repetition_penalty
                else:
                    output[token_id] /= repetition_penalty
            filtered = top_k_top_p_filtering(output, top_k=top_k, top_p=top_p)
            token_id = torch.multinomial(F.softmax(filtered, dim=-1), 1).item()
            if token_id == eos_id:
                break
            decoded.append(token_id)
            used_tokens.add(token_id)
            input_token = torch.tensor([token_id], device=device)
    return tokenizer.decode(decoded).replace('▁', ' ').strip()


def decode_tokens(tokenizer, sequences, eos_id=2, pad_id=PAD_ID):
    """Decode reference or predicted token IDs up to EOS or PAD."""
    results = []
    for sequence in sequences:
        tokens = []
        for token in sequence:
            if token == eos_id or token == pad_id:
                break
            tokens.append(token)
        results.append(tokenizer.decode(tokens).replace('▁', ' ').strip())
    return results


def compute_text_metrics(predictions, references):
    """Compute the BLEU, METEOR, and ROUGE-L values used in the notebook."""
    bleu = hf_evaluate.load('bleu')
    meteor = hf_evaluate.load('meteor')
    rouge = hf_evaluate.load('rouge')
    return {
        'BLEU': bleu.compute(predictions=predictions, references=[[r] for r in references])['bleu'],
        'METEOR': meteor.compute(predictions=predictions, references=references)['meteor'],
        'ROUGE': rouge.compute(predictions=predictions, references=references)['rougeL'],
    }


def restore_slots(masked_text, slot_values):
    """Restore the saved time and location values as in the notebook."""
    restored = masked_text
    for slot, value in slot_values.items():
        restored = restored.replace(slot, value)
    return restored


def validation_file_ids(directory):
    ids = []
    for x_path in sorted(Path(directory).glob('X_part_*.npy')):
        id_path = x_path.with_name(x_path.name.replace('X_part_', 'file_id_part_'))
        if not id_path.exists():
            raise FileNotFoundError(f'Missing sample IDs: {id_path}')
        ids.extend(np.load(id_path, allow_pickle=True).tolist())
    return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True, help='Output directory from prepare_arrays.py.')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', default='val_predictions.csv')
    parser.add_argument('--json-dir', help='Validation JSON directory for restoring masked slots.')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--repetition-penalty', type=float, default=1.2)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    tokenizer = spm.SentencePieceProcessor(model_file=str(data_dir / 'spm.model'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Seq2Seq(Encoder(), Decoder(output_dim=tokenizer.vocab_size()), device).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    predictions, references = [], []
    for x_batch, y_batch in make_loader(data_dir / 'val', args.batch_size):
        # Sample each sequence independently, as in the notebook's validation pass.
        predictions.extend(sample_decode(model, x, tokenizer, device,
                                         repetition_penalty=args.repetition_penalty,
                                         top_k=args.top_k, top_p=args.top_p)
                           for x in x_batch)
        references.extend(decode_tokens(tokenizer, y_batch.tolist()))

    ids = validation_file_ids(data_dir / 'val')
    if len(ids) != len(predictions):
        raise ValueError('Validation sample IDs and predictions have different lengths.')
    result = pd.DataFrame({'file_id': ids, 'prediction': predictions,
                           'reference': references})
    if args.json_dir:
        slot_values = []
        for file_id in ids:
            path = Path(args.json_dir) / f'{file_id}.json'
            data = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
            slot_values.append(data.get('slot_values', {}))
        result['restored_prediction'] = [restore_slots(text, slots)
                                         for text, slots in zip(predictions, slot_values)]
        result['restored_reference'] = [restore_slots(text, slots)
                                        for text, slots in zip(references, slot_values)]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding='utf-8-sig')
    print(compute_text_metrics(predictions, references))
    print(f'Saved {len(result)} validation predictions to {args.output}')


if __name__ == '__main__':
    main()
