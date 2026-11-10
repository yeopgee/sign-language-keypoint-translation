"""Train the BiGRU-attention model on prepared keypoint/text arrays."""

import argparse
from pathlib import Path

import sentencepiece as spm
import torch
from torch import nn

from dataset import PAD_ID, make_loader
from model import Decoder, Encoder, Seq2Seq


def epoch_loss(model, loader, criterion, device, optimizer=None):
    model.train(optimizer is not None)
    total = count = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            output = model(x, y, teacher_forcing_ratio=0.5 if optimizer else 0.0)
            # Position 0 is the BOS token; predict positions 1 onward.
            loss = criterion(output[:, 1:].reshape(-1, output.size(-1)), y[:, 1:].reshape(-1))
            if optimizer is not None:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        total += loss.item() * len(x)
        count += len(x)
    return total / count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True, help='Directory containing spm.model, train/, and val/.')
    parser.add_argument('--output', default='checkpoints/best_model.pt')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    tokenizer = spm.SentencePieceProcessor(model_file=str(data_dir / 'spm.model'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Seq2Seq(Encoder(), Decoder(output_dim=tokenizer.vocab_size()), device).to(device)
    train_loader = make_loader(data_dir / 'train', args.batch_size, shuffle=True)
    val_loader = make_loader(data_dir / 'val', args.batch_size)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best = float('inf')
    for epoch in range(1, args.epochs + 1):
        train_loss = epoch_loss(model, train_loader, criterion, device, optimizer)
        val_loss = epoch_loss(model, val_loader, criterion, device)
        print(f'Epoch {epoch}: train loss {train_loss:.4f}, val loss {val_loss:.4f}')
        if val_loss < best:
            best = val_loss
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'model_state_dict': model.state_dict(), 'vocab_size': tokenizer.vocab_size()}, output)


if __name__ == '__main__':
    main()
