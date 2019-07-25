import math

import torch
import torch.nn.functional as F
from torch import nn

from .modules import EmbeddingLayer, EncoderLayer, DecoderLayer


class StyleTransformer(nn.Module):
    def __init__(
        self,
        vocab,
        num_styles: int = 2,
        num_layers: int = 4,
        num_heads: int = 8,
        hidden_dim: int = 256,
        max_seq_len: int = 30,
        dropout: float = 0.1,
        load_pretrained_embed: bool=False,
    ):
        super(StyleTransformer, self).__init__()
        self.vocab_size = len(vocab)
        self.num_styles = num_styles
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.load_pretrained_embed = load_pretrained_embed

        self.eos_idx = vocab.stoi["<eos>"]
        self.pad_idx = vocab.stoi["<pad>"]
        self.style_embed = nn.Embedding(num_styles, hidden_dim)
        self.embed = EmbeddingLayer(
            len(vocab),
            hidden_dim,
            max_seq_len,
            dropout,
            self.pad_idx,
        )
        if load_pretrained_embed:
            self.embed.load_vocab_embedding(vocab.vectors)

        self.sos_token = nn.Parameter(torch.randn(hidden_dim))
        self.encoder = Encoder(num_layers, hidden_dim,
                               len(vocab), num_heads, dropout)
        self.decoder = Decoder(num_layers, hidden_dim,
                               len(vocab), num_heads, dropout)

        # reset parameters
        nn.init.constant_(self.style_embed.weight, 0)

    def forward(
        self,
        inp_tokens,
        gold_tokens,
        inp_lengths,
        style,
        generate=False,
        differentiable_decode=False,
        temperature=1.0,
    ):
        batch_size = inp_tokens.size(0)
        max_enc_len = inp_tokens.size(1)

        msg = f"sequence length is exceeded, length: {max_enc_len} >= {self.max_seq_len}"
        assert max_enc_len <= self.max_seq_len, msg

        pos_idx = torch.arange(self.max_seq_len).unsqueeze(
            0).expand((batch_size, -1))
        pos_idx = pos_idx.to(inp_lengths.device)

        src_mask = pos_idx[:, :max_enc_len] >= inp_lengths.unsqueeze(-1)
        src_mask = torch.cat((torch.zeros_like(src_mask[:, :1]), src_mask), 1)
        src_mask = src_mask.view(batch_size, 1, 1, max_enc_len + 1)
        tgt_mask = torch.ones(
            (self.max_seq_len, self.max_seq_len)).to(src_mask.device)
        tgt_mask = (tgt_mask.tril() == 0).view(
            1, 1, self.max_seq_len, self.max_seq_len)

        style_emb = self.style_embed(style).unsqueeze(1)

        enc_input = torch.cat(
            (style_emb, self.embed(inp_tokens, pos_idx[:, :max_enc_len])), 1
        )
        memory = self.encoder(enc_input, src_mask)

        sos_token = self.sos_token.view(1, 1, -1).expand(batch_size, -1, -1)

        if not generate:
            dec_input = gold_tokens[:, :-1]
            max_dec_len = gold_tokens.size(1)
            dec_input_emb = torch.cat(
                (sos_token, self.embed(dec_input,
                                       pos_idx[:, : max_dec_len - 1])), 1
            )
            log_probs = self.decoder(
                dec_input_emb,
                memory,
                src_mask,
                tgt_mask[:, :, :max_dec_len, :max_dec_len],
                temperature,
            )
        else:

            log_probs = []
            next_token = sos_token
            prev_states = None

            for k in range(self.max_seq_len):
                log_prob, prev_states = self.decoder.incremental_forward(
                    next_token,
                    memory,
                    src_mask,
                    tgt_mask[:, :, k: k + 1, : k + 1],
                    temperature,
                    prev_states,
                )

                log_probs.append(log_prob)

                if differentiable_decode:
                    next_token = self.embed(
                        log_prob.exp(), pos_idx[:, k: k + 1])
                else:
                    next_token = self.embed(
                        log_prob.argmax(-1), pos_idx[:, k: k + 1])

            log_probs = torch.cat(log_probs, 1)

        return log_probs


class Discriminator(nn.Module):
    def __init__(
        self,
        vocab,
        num_styles: int = 2,
        num_layers: int = 4,
        num_heads: int = 8,
        hidden_dim: int = 512,
        max_seq_len: int = 50,
        dropout: float = 0.1,
        discriminator_method: str='Multi',
        load_pretrained_embed: bool=False,
    ):
        super(Discriminator, self).__init__()
        self.vocab_size = len(vocab)
        self.num_styles = num_styles,
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.load_pretrained_embed = load_pretrained_embed
        self.num_classes = num_styles
        self.discriminator_method = discriminator_method
        if discriminator_method == "Multi":
            self.num_classes += 1

        self.pad_idx = vocab.stoi["<pad>"]
        self.style_embed = nn.Embedding(num_styles, hidden_dim)
        self.embed = EmbeddingLayer(
            len(vocab),
            hidden_dim,
            max_seq_len,
            dropout,
            self.pad_idx,
        )
        if load_pretrained_embed:
            self.embed.load_vocab_embedding(vocab.vectors)

        self.cls_token = nn.Parameter(torch.randn(hidden_dim))
        self.encoder = Encoder(num_layers, hidden_dim, len(vocab), num_heads, dropout)
        self.classifier = nn.Linear(hidden_dim, self.num_classes)

        # reset parameters
        nn.init.constant_(self.style_embed.weight, 0)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0.0)

    def forward(self, inp_tokens, inp_lengths, style=None):
        batch_size = inp_tokens.size(0)
        num_extra_token = 1 if style is None else 2
        max_seq_len = inp_tokens.size(1)

        pos_idx = (
            torch.arange(max_seq_len)
            .unsqueeze(0)
            .expand((batch_size, -1))
            .to(inp_lengths.device)
        )
        mask = pos_idx >= inp_lengths.unsqueeze(-1)
        for _ in range(num_extra_token):
            mask = torch.cat((torch.zeros_like(mask[:, :1]), mask), 1)
        mask = mask.view(batch_size, 1, 1, max_seq_len + num_extra_token)

        cls_token = self.cls_token.view(1, 1, -1).expand(batch_size, -1, -1)

        enc_input = cls_token
        if style is not None:
            style_emb = self.style_embed(style).unsqueeze(1)
            enc_input = torch.cat((enc_input, style_emb), 1)

        enc_input = torch.cat((enc_input, self.embed(inp_tokens, pos_idx)), 1)

        encoded_features = self.encoder(enc_input, mask)
        logits = self.classifier(encoded_features[:, 0])

        return F.log_softmax(logits, -1)


class Encoder(nn.Module):
    def __init__(self, num_layers, hidden_dim, vocab_size, h, dropout):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(hidden_dim, h, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-6)

    def forward(self, x, mask):
        y = x

        assert y.size(1) == mask.size(-1)

        for layer in self.layers:
            y = layer(y, mask)

        return self.norm(y)


class Decoder(nn.Module):
    def __init__(self, num_layers, hidden_dim, vocab_size, h, dropout):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(hidden_dim, h, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.generator = Generator(hidden_dim, vocab_size)

    def forward(self, x, memory, src_mask, tgt_mask, temperature):
        y = x

        assert y.size(1) == tgt_mask.size(-1)

        for layer in self.layers:
            y = layer(y, memory, src_mask, tgt_mask)

        return self.generator(self.norm(y), temperature)

    def incremental_forward(
        self, x, memory, src_mask, tgt_mask, temperature, prev_states=None
    ):
        y = x

        new_states = []

        for i, layer in enumerate(self.layers):
            y, new_sub_states = layer.incremental_forward(
                y, memory, src_mask, tgt_mask, prev_states[i] if prev_states else None
            )
            new_states.append(new_sub_states)

        new_states.append(
            torch.cat((prev_states[-1], y), 1) if prev_states else y)
        y = self.norm(new_states[-1])[:, -1:]

        return self.generator(y, temperature), new_states


class Generator(nn.Module):
    def __init__(self, hidden_dim, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, temperature):
        return F.log_softmax(self.proj(x) / temperature, dim=-1)
