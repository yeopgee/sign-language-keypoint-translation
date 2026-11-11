"""Train sign-to-text with the notebook's label smoothing and BLEU selection."""

import argparse
from pathlib import Path

import sentencepiece as spm
import torch
from torch import nn
from torch.nn import functional as F

from dataset import PAD_ID, make_loader
from evaluation import compute_text_metrics, decode_tokens
from model import Decoder, Encoder, Seq2Seq


class LabelSmoothingLoss(nn.Module):
    """The KL-divergence label-smoothing loss used in the notebook."""

    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        super().__init__()
        self.ignore_index = ignore_index
        self.confidence = 1.0 - label_smoothing
        self.smoothing = label_smoothing
        self.vocab_size = tgt_vocab_size

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.vocab_size - 2))
        ignore = target == self.ignore_index
        target = target.masked_fill(ignore, 0)
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist.masked_fill_(ignore.unsqueeze(1), 0.0)
        return F.kl_div(pred, true_dist, reduction='batchmean')


def run_epoch(model, loader, criterion, device, teacher_forcing, optimizer=None, clip=1.0):
    model.train(optimizer is not None)
    total_loss = 0.0
    predictions, references = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            # Match the notebook: decoder input excludes EOS and the target
            # starts after BOS. Validation uses no teacher forcing.
            output = model(x, y[:, :-1], teacher_forcing_ratio=teacher_forcing)
            target = y[:, 1:]
            loss = criterion(output.reshape(-1, output.size(-1)), target.reshape(-1))
            if optimizer is not None:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
        total_loss += loss.item()
        if optimizer is None:
            predictions.extend(output.argmax(-1).tolist())
            references.extend(y.tolist())
    return total_loss / len(loader), predictions, references


def save_checkpoint(path, model, optimizer, epoch, metrics, val_loss):
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch, 'val_bleu': metrics['BLEU'],
                'val_meteor': metrics['METEOR'], 'val_loss': val_loss}, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True, help='Directory containing spm.model, train/, and val/.')
    parser.add_argument('--checkpoint-dir', default='checkpoints')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--teacher-forcing', type=float, default=0.5)
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--patience', type=int, default=5)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = spm.SentencePieceProcessor(model_file=str(data_dir / 'spm.model'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Seq2Seq(Encoder(), Decoder(output_dim=tokenizer.vocab_size()), device).to(device)
    train_loader = make_loader(data_dir / 'train', args.batch_size, shuffle=True)
    val_loader = make_loader(data_dir / 'val', args.batch_size)
    criterion = LabelSmoothingLoss(args.label_smoothing, tokenizer.vocab_size(),
                                   ignore_index=PAD_ID)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_bleu, patience = -1.0, 0

    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = run_epoch(model, train_loader, criterion, device,
                                     args.teacher_forcing, optimizer, args.clip)
        val_loss, predicted_ids, reference_ids = run_epoch(model, val_loader,
                                                            criterion, device, 0.0)
        # The notebook selects checkpoints using BLEU on argmax token IDs.
        predictions = decode_tokens(tokenizer, predicted_ids)
        references = decode_tokens(tokenizer, reference_ids)
        metrics = compute_text_metrics(predictions, references)
        print(f"Epoch {epoch}: train loss {train_loss:.4f}, val loss {val_loss:.4f}, "
              f"BLEU {metrics['BLEU']:.4f}, METEOR {metrics['METEOR']:.4f}, "
              f"ROUGE {metrics['ROUGE']:.4f}")
        save_checkpoint(checkpoint_dir / f'checkpoint_epoch{epoch}.pt', model,
                        optimizer, epoch, metrics, val_loss)
        if metrics['BLEU'] > best_bleu:
            best_bleu, patience = metrics['BLEU'], 0
            save_checkpoint(checkpoint_dir / 'best_checkpoint.pt', model,
                            optimizer, epoch, metrics, val_loss)
        else:
            patience += 1
            if patience >= args.patience:
                print(f'Early stopping at epoch {epoch}; best BLEU {best_bleu:.4f}')
                break


if __name__ == '__main__':
    main()
