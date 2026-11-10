"""BiGRU encoder and attention-based GRU decoder for sign-to-text."""
import torch
from torch import nn
from torch.nn import functional as F

class Attention(nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        super().__init__()
        self.attn = nn.Linear(encoder_hidden_dim + decoder_hidden_dim, decoder_hidden_dim)
        self.v = nn.Linear(decoder_hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: (B, dec_hidden)
        # encoder_outputs: (B, src_len, enc_hidden_dim)
        src_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)  # (B, src_len, dec_hidden)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # (B, src_len, dec_hidden)
        attention = self.v(energy).squeeze(2)  # (B, src_len)
        return F.softmax(attention, dim=1)


class Encoder(nn.Module):
    def __init__(self, input_dim=108, hidden_dim=256):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)

    def forward(self, src):
        # src: (B, T, input_dim)
        outputs, hidden = self.rnn(src)  # outputs: (B, T, 2*H), hidden: (2, B, H)
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim=256, enc_hidden_dim=256, dec_hidden_dim=256):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.attention = Attention(enc_hidden_dim * 2, dec_hidden_dim)
        self.rnn = nn.GRU(emb_dim + enc_hidden_dim * 2, dec_hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(emb_dim + enc_hidden_dim * 2 + dec_hidden_dim, output_dim)

    def forward(self, input, hidden, encoder_outputs):
        # input: (B,)
        # hidden: (1, B, dec_hidden)
        # encoder_outputs: (B, src_len, enc_hidden*2)
        input = input.unsqueeze(1)  # (B, 1)
        embedded = self.embedding(input)  # (B, 1, emb_dim)

        attn_weights = self.attention(hidden.squeeze(0), encoder_outputs)  # (B, src_len)
        attn_weights = attn_weights.unsqueeze(1)  # (B, 1, src_len)
        context = torch.bmm(attn_weights, encoder_outputs)  # (B, 1, enc_hidden*2)

        rnn_input = torch.cat((embedded, context), dim=2)  # (B, 1, emb_dim + ctx)
        output, hidden = self.rnn(rnn_input, hidden)  # output: (B, 1, dec_hidden)

        concat = torch.cat((output, context, embedded), dim=2)  # (B, 1, total_dim)
        prediction = self.fc_out(concat.squeeze(1))  # (B, output_dim)

        return prediction, hidden, attn_weights.squeeze(1)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, pad_id=3):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.pad_id = pad_id

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: (B, T, 108), trg: (B, L)
        batch_size = src.size(0)
        trg_len = trg.size(1)
        output_dim = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len, output_dim).to(self.device)

        # Merge the forward and backward encoder states for the decoder.
        encoder_outputs, hidden = self.encoder(src)  # hidden: (2, B, H)
        hidden = torch.tanh(hidden[0] + hidden[1]).unsqueeze(0)  # (1, B, H)

        # Start decoding from the beginning-of-sequence token.
        input = trg[:, 0]  # (B,)

        for t in range(1, trg_len):
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs)  # output: (B, vocab)
            outputs[:, t] = output

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)  # (B,)

            input = trg[:, t] if teacher_force else top1

        return outputs  # (B, L, vocab)
