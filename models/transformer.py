import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from .modules import DecoderLayer, EmbeddingLayer, EncoderLayer


class Encoder(nn.Module):
    """
    Transformer Encoder
    """

    __slot__ = [
        "num_layers",
        "num_heads",
        "hidden_dim",
        "dropout",
        "layer_stack",
        "output_normalization",
    ]

    def __init__(
        self,
        num_layers: int = 4,
        num_heads: int = 8,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        """
        Create Transformer Encoder with N layers

        :param num_layers: the number of layers
        :param num_heads: the number of heads in multi-head attention
        :param hidden_dim: dimention of hidden layer
        :param dropout: dropout ratio
        """
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.layer_stack = nn.ModuleList(
            [
                EncoderLayer(hidden_dim, num_heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.output_normalization = nn.LayerNorm(hidden_dim, eps=1e-6)

    def forward(
        self, encoder_input: torch.Tensor, pad_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """
        :param encoder_inputs: encoder inputs [batch_size, max_seq_len, hidden_dim]
        :param pad_mask: mask matrix for ignoring PAD_ID [batch_size, 1, max_seq_len]
        :return: encoder_output: [batch_size, max_seq_len, hidden_dim]
            self_attn: [batch_size * num_heads, max_seq_len, max_seq_len]
        NOTE
        ----
        only returns final layer's encoder_output and attention just now
        """
        assert encoder_input.size(1) == pad_mask.size(-1)

        for enc_layer in self.layer_stack:
            encoder_input = enc_layer(encoder_input, pad_mask)

        encoder_output = self.output_normalization(encoder_input)

        return encoder_output


class Decoder(nn.Module):
    """
    Transformer Decoder
    """

    __slot__ = [
        "num_layers",
        "num_heads",
        "hidden_dim",
        "dropout",
        "layer_stack",
        "output_normalization",
    ]

    def __init__(
        self,
        num_layers: int = 4,
        num_heads: int = 4,
        hidden_dim: int = 256,
        vocab_size: int = 16000,
        dropout: float = 0.1,
    ) -> None:
        """
        Create Transformer Decoder with N layers

        :param num_layers: the number of layers
        :param num_heads: the number of heads in multi-head attention
        :param hidden_dim: dimention of hidden layer
        :param vocab_size: size of output vocabulary
        :param dropout: dropout ratio
        """
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        self.layer_stack = nn.ModuleList(
            [
                DecoderLayer(hidden_dim, num_heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.output_normalization = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.generator = Generator(hidden_dim, vocab_size)

    def forward(
        self,
        decoder_input: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """
        :param decoder_input: decoder inputs [batch_size, max_seq_len]
        :param encoder_output: encoder_outputs [batch_size, max_seq_len, hidden_dim]
        :param src_mask: matrix of padding mask, which is denoted as 1 if PAD_ID is there
            [batch_size, 1, max_seq_len]
        :param tgt_mask: self-attention mask to prevent future information from leaking
            It is usually not necessary for Encoder part [batch_size, max_seq_len, max_seq_len]
        :param temperature: softmax temperature
        :return: decoder_output: [batch_size, max_seq_len, hidden_dim]
            self_attn: [batch_size * num_heads, query_max_len, query_max_len]
            src_tgt_attn: [batch_size * num_heads, query_max_len, key_max_len]

        NOTE
        ----
        only returns final layer's encoder_output and attention just now
        """
        assert src_mask is None or encoder_output.size(1) == src_mask.size(-1)
        assert tgt_mask is None or decoder_input.size(1) == tgt_mask.size(-1)

        for dec_layer in self.layer_stack:
            decoder_input = dec_layer(
                decoder_input, encoder_output, src_mask=src_mask, tgt_mask=tgt_mask
            )

        decoder_output = self.output_normalization(decoder_input)
        logits = self.generator(decoder_output, temperature)

        return logits

    def incremental_forward(
        self,
        decoder_input: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        prev_states: Optional[List[List[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        """
        :param decoder_input: decoder inputs [batch_size, query_max_seq_len]
        :param encoder_output: encoder_outputs [batch_size, key_max_seq_len, hidden_dim]
        :param src_mask: matrix of padding mask, which is denoted as 1 if PAD_ID is there
            [batch_size, 1, max_seq_len]
        :param tgt_mask: self-attention mask to prevent future information from leaking
            tgt_mask: [batch_size, query_max_seq_len, query_max_seq_len]
        :param temperature: softmax temperature
        :param prev_states: previous states for incremental forwaring
        NOTE
        ----
        only returns final layer's encoder_output and attention just now

        """
        assert src_mask is None or encoder_output.size(1) == src_mask.size(-1)
        assert tgt_mask is None or decoder_input.size(1) == tgt_mask.size(1)

        new_states = []

        for idx, dec_layer in enumerate(self.layer_stack):
            decoder_input, new_sub_states = dec_layer.incremental_forward(
                decoder_input,
                encoder_output,
                src_mask=src_mask,
                tgt_mask=tgt_mask,
                prev_states=prev_states[idx] if prev_states else None,
            )
            new_states.append(new_sub_states)

        # prev_states[-1] is previous decoder output of incremental forward
        new_states.append(
            torch.cat((prev_states[-1], decoder_input), 1)
            if prev_states
            else decoder_input
        )
        decoder_output = self.output_normalization(new_states[-1])[:, -1:]
        logits = self.generator(decoder_output, temperature)

        return logits, new_states


class Generator(nn.Module):
    """
    Transformer generator
    """

    __slot__ = ["output_projection"]

    def __init__(
        self, hidden_dim: int = 256, vocab_size: int = 16000, bias: bool = True
    ) -> None:
        """
        Generate target tokens from Decoder's outputs

        :param hidden_dim: dimenstion of hidden layer
        :param vocab_size: size of vocabulary
        """
        super(Generator, self).__init__()
        self.output_projection = nn.Linear(hidden_dim, vocab_size, bias=bias)

    def forward(
        self,
        decoder_output: torch.Tensor,
        temperature: Union[float, torch.Tensor] = 1.0,
    ) -> torch.Tensor:
        """
        :param decoder_output: decoder output [batch_size, max_seq_len, hidden_dim]
        :param temperature: softmax temperature
        """
        logits = self.output_projection(decoder_output) / temperature

        return F.log_softmax(logits, dim=-1)


class StyleTransformer(nn.Module):
    """
    Style-Transformer for converting one style to another
    """

    def __init__(
        self,
        vocab,
        num_styles: int = 2,
        num_layers: int = 4,
        num_heads: int = 8,
        hidden_dim: int = 256,
        max_seq_len: int = 30,
        dropout: float = 0.1,
        load_pretrained_embed: bool = False,
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
            len(vocab), hidden_dim, max_seq_len, dropout, self.pad_idx
        )
        if load_pretrained_embed:
            self.embed.load_vocab_embedding(vocab.vectors)

        self.sos_token = nn.Parameter(torch.randn(hidden_dim))
        self.encoder = Encoder(num_layers, num_heads, hidden_dim, dropout)
        self.decoder = Decoder(num_layers, num_heads, hidden_dim, len(vocab), dropout)

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

        msg = (
            f"sequence length is exceeded, length: {max_enc_len} >= {self.max_seq_len}"
        )
        assert max_enc_len <= self.max_seq_len, msg

        pos_idx = torch.arange(self.max_seq_len).unsqueeze(0).expand((batch_size, -1))
        pos_idx = pos_idx.to(inp_lengths.device)

        src_mask = pos_idx[:, :max_enc_len] >= inp_lengths.unsqueeze(-1)
        src_mask = torch.cat((torch.zeros_like(src_mask[:, :1]), src_mask), 1)
        src_mask = src_mask.view(batch_size, 1, 1, max_enc_len + 1)
        tgt_mask = torch.ones((self.max_seq_len, self.max_seq_len)).to(src_mask.device)
        tgt_mask = (tgt_mask.tril() == 0).view(1, 1, self.max_seq_len, self.max_seq_len)

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
                (sos_token, self.embed(dec_input, pos_idx[:, : max_dec_len - 1])), 1
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
                    next_token = self.embed(log_prob.exp(), pos_idx[:, k: k + 1])
                else:
                    next_token = self.embed(log_prob.argmax(-1), pos_idx[:, k: k + 1])

            log_probs = torch.cat(log_probs, 1)

        return log_probs


class Discriminator(nn.Module):
    """
    Discriminator for Style-Transformer
    """

    def __init__(
        self,
        vocab,
        num_styles: int = 2,
        num_layers: int = 4,
        num_heads: int = 8,
        hidden_dim: int = 512,
        max_seq_len: int = 50,
        dropout: float = 0.1,
        discriminator_method: str = "Multi",
        load_pretrained_embed: bool = False,
    ):
        super(Discriminator, self).__init__()
        self.vocab_size = len(vocab)
        self.num_styles = (num_styles,)
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
            len(vocab), hidden_dim, max_seq_len, dropout, self.pad_idx
        )
        if load_pretrained_embed:
            self.embed.load_vocab_embedding(vocab.vectors)

        self.cls_token = nn.Parameter(torch.randn(hidden_dim))
        self.encoder = Encoder(num_layers, num_heads, hidden_dim, dropout)
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
