"""Greedy sign-to-text inference from a prepared (frames, 108) NumPy array."""

import argparse

import numpy as np
import sentencepiece as spm
import torch

from model import Decoder, Encoder, Seq2Seq


@torch.no_grad()
def translate(model, keypoints, tokenizer, device, max_tokens=100):
    model.eval()
    x = torch.as_tensor(keypoints, dtype=torch.float32, device=device).unsqueeze(0)
    encoder_outputs, hidden = model.encoder(x)
    hidden = torch.tanh(hidden[0] + hidden[1]).unsqueeze(0)
    token = torch.tensor([tokenizer.bos_id()], device=device)
    output_ids = []
    for _ in range(max_tokens):
        logits, hidden, _ = model.decoder(token, hidden, encoder_outputs)
        token = logits.argmax(dim=-1)
        next_id = token.item()
        if next_id == tokenizer.eos_id():
            break
        output_ids.append(next_id)
    return tokenizer.decode(output_ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--keypoints', required=True, help='Path to an .npy array of shape (frames, 108).')
    parser.add_argument('--tokenizer', required=True, help='Path to spm.model.')
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = spm.SentencePieceProcessor(model_file=args.tokenizer)
    model = Seq2Seq(Encoder(), Decoder(output_dim=tokenizer.vocab_size()), device).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(translate(model, np.load(args.keypoints), tokenizer, device))


if __name__ == '__main__':
    main()
