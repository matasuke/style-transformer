from typing import Optional, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingLayer(nn.Module):
    """
    BERT like Embedding layers, which combines vocabulary embedding and positional embedding
    """
    def __init__(
        self,
        vocab_size: int = 16000,
        hidden_dim: int = 256,
        max_seq_len: int = 50,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ) -> None:
        super(EmbeddingLayer, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.vocab_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_idx)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.embedding_dropout = nn.Dropout(dropout)

        # reset parameters
        nn.init.constant_(self.vocab_embedding.weight, 0)
        nn.init.constant_(self.pos_embedding.weight, 0)

    def forward(self, sequence: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
        forward pass for embedding.

        :param sequences: source sequences: [batch_size, max_seq_len] or
            [batch_size, max_seq_len, vocab_size]
        :param pos: positional sequences: [batch_size, max_seq_len]
        """

        # when input sequences are size of [batch_size, max_seq_len]
        if sequence.dim() == 2:
            embedded = self.vocab_embedding(sequence)
            embedded += self.pos_embedding(pos)
        # when input sequences are size of [batch_size, max_seq_len, vocab_size]
        elif sequence.dim() == 3:
            embedded = torch.matmul(sequence, self.vocab_embedding.weight)
            embedded += self.pos_embedding(pos)
        else:
            raise Exception("Size of src_sequences has to be size(2) or size(3)")

        embedded = self.embedding_dropout(embedded)

        return embedded

    def _load_embedding(
        self, embedding: torch.Tensor, emb_weight: "np.ndarray", freeze: bool = False
    ) -> None:
        """
        load pretrained embedding, which is trained by TextPreprocessor

        ;param embedding: nn.Embedding
        :param emb_weight: embedding weight
        :param freeze: freeze loaded weight
        """
        msg = "shape of both embedding and emb_weight mush be same"
        assert embedding.weight.shape == emb_weight.shape, msg

        if isinstance(embedding, torch.nn.modules.sparse.Embedding):
            embedding.weight = nn.Parameter(torch.from_numpy(emb_weight))
        else:
            raise TypeError("embedding has to be torch.Tensor")

        if freeze:
            embedding.weight.requires_grad = False

    def load_vocab_embedding(
        self, vocab_emb_weight: "np.ndarray", freeze: bool = False
    ) -> None:
        """
        load pretrained vocabulary embedding.

        :param vocab_emb_weight: pre-trained embedding weight
        :param freeze: freeze loaded weight
        """
        self._load_embedding(self.vocab_embedding, vocab_emb_weight, freeze)

    def load_pos_embedding(
        self, pos_emb_weight: "np.ndarray", freeze: bool = False
    ) -> None:
        """
        load pretrained positionall embedding.

        :param pos_emb_weight: pre-trained embedding weight
        :param freeze: freeze loaded weight
        """
        self._load_embedding(self.pos_embedding, pos_emb_weight, freeze)


class MultiHeadAttention(nn.Module):
    """
    Multihead Attention used by Transformer
    """

    __slot__ = [
        "hidden_dim",
        "num_heads",
        "dropout",
        "head_dim",
        "w_query",
        "w_key",
        "w_value",
        "scaled_attention",
        "output_projection",
    ]

    def __init__(
        self, hidden_dim: int = 256, num_heads: int = 8, dropout: float = 0.1
    ) -> None:
        """
        Multihead attention

        :param hidden_dim: size of dimension in hidden layer.
        :param num_heads: The number of heads in multi-head attention.
        :param dropout: dropout ratio
        """
        super(MultiHeadAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = hidden_dim // num_heads
        msg = "embed_dim must be divisible by num_heads"
        assert self.head_dim * num_heads == self.hidden_dim, msg

        self.w_query = nn.Linear(hidden_dim, num_heads * self.head_dim)
        self.w_key = nn.Linear(hidden_dim, num_heads * self.head_dim)
        self.w_value = nn.Linear(hidden_dim, num_heads * self.head_dim)

        self.scaled_attention = ScaledDotProductAttention(
            temperature=np.power(self.head_dim, 0.5), dropout=dropout
        )
        self.output_projection = nn.Linear(num_heads * self.head_dim, hidden_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """
        :param query: [batch_size, max_seq_len, hidden_dim]
        :param key: [batch_size, max_seq_len, hidden_dim]
        :param value: [batch_size, max_seq_len, hidden_dim]
        :param mask: [batch_size, max_seq_len]
        """
        query_batch_size, query_seq_len, _ = query.size()
        key_batch_size, key_seq_len, _ = key.size()
        value_batch_size, value_seq_len, _ = key.size()

        msg = "size of query, key and value is different"
        assert query_batch_size == key_batch_size == value_batch_size, msg

        # [batch_size, seq_len, num_heads*head_dim] -> [batch_size, num_heads, seq_len, head_dim]
        query = (
            self.w_query(query)
            .view(query_batch_size, query_seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key = (
            self.w_key(key)
            .view(key_batch_size, key_seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        value = (
            self.w_value(value)
            .view(value_batch_size, value_seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # attn_feature: [batch_size, num_heads, seq_len, head_dim]
        output, _ = self.scaled_attention(query, key, value, mask)

        # [batch_size, num_heads, seq_len, head_dim] -> [batch_size, seq_len, num_heads, head_dim]
        # -> [batch_size, seq_len, num_hads * head_dim]
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(query_batch_size, query_seq_len, self.num_heads * self.head_dim)
        )
        output = self.output_projection(output)

        return output


class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention
    """

    __slot__ = ["temperature", "dropout", "softmax"]

    def __init__(self, temperature: float, dropout: float = 0.1) -> None:
        """
        scaled dot-product attention

        :param temperature: softmax temperature
        :param dropout: dropout ratio
        """
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """
        forward pass for scaled dot-product attention

        :param query: query: [batch_size, num_heads, seq_len, head_dim]
        :param key: key: [batch_size, num_heads, seq_len, head_dim]
        :param value:: [batch_size, num_heads, seq_len, head_dim]
        :param mask: padding mask and self-attention mask
            mask: [num_heads * batch_size, 1, seq_len]
        :return: output: [num_heads * batch_size, seq_len, head_dim]
            attn: [num_heads * batch_size, seq_len, seq_len]
        """
        # attn: [batch_size, num_heads, seq_len, seq_len]
        attn = query.matmul(key.transpose(-2, -1))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill_(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        # attn: [batch_size, num_heads, seq_len, seq_len]
        # value: [batch_size, num_heads, seq_len, head_dim]
        # output: [batch_size, num_heads, seq_len, head_dim]
        output = attn.matmul(value)

        return output, attn


class FeedForwardNetwork(nn.Module):
    """
    Feed Forward Network used by Transformer
    """

    __slot__ = ["hidden_dim", "linear_1", "linear_2", "dropout"]

    def __init__(self, hidden_dim: int = 512, dropout: float = 0.1) -> None:
        """
        Feed Forward Network of two layers

        :param hidden_dim: dimention of hidden layer
        :param dropout: dropout ratio
        """
        super(FeedForwardNetwork, self).__init__()
        self.hidden_dim = hidden_dim

        self.linear_1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(hidden_dim * 4, hidden_dim)

        self._reset_parameters()

    def forward(self, attention: torch.Tensor) -> torch.Tensor:
        attention = self.dropout(F.relu(self.linear_1(attention)))
        attention = self.linear_2(attention)

        return attention

    def _reset_parameters(self) -> None:
        """
        initialize parameters.

        :param bias: use bias or not
        :param uniform: use uniform distribution as initialization method
        """
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.constant_(self.linear_1.bias, 0.0)
        nn.init.constant_(self.linear_2.bias, 0.0)


class ResidualNormalizationWrapper(nn.Module):
    """
    Residual Normalization Wrapper for specified layer
    """

    __slot__ = ["layer_normalization", "dropout"]

    def __init__(self, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        """
        Create residual normalization wrapper any layer specified

        NOTE
        ----
        input -> nn.LayerNorm -> layer -> nn.Dropout -> output -> output + input

        :param hidden_dim: dimention of hidden layer
        :param dropout: dropout ratio
        """
        super(ResidualNormalizationWrapper, self).__init__()
        self.layer_normalization = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input: torch.Tensor, layer: nn.Module) -> torch.Tensor:
        """
        apply residual normalization wrapper to specified layer

        :param input: input tensor to apply residual normalization.
            [max_seq_len, batch_size, hidden_dim]
        :param layer: layer to apply residual normalization
        :return: [batch_size, seq_len, hidden_dim]
        """
        output = self.layer_normalization(input)
        output = layer(output)
        output = self.dropout(output)

        return output + input  # residual connection

    def incremental_forward(
        self, input: torch.Tensor, layer: nn.Module
    ) -> torch.Tensor:
        """
        apply residual normalization wrapper to specified layer and incremental forwading,
        which apply forwarding just for one step

        :param input: input tensor to apply residual normalization. [batch_size, 1, hidden_dim]
        :param layer: layer to apply residual normalization
        :return: [batch_size, 1, hidden_dim]
        """
        output = self.layer_normalization(input)
        output = layer(output)
        output = self.dropout(output)

        return output + input[:, -1:]  # residual connection


class EncoderLayer(nn.Module):
    """
    One Encoder layer of multi-head attention and feed-forward layer with residual wrapper
    """

    __slot__ = [
        "self_attention",
        "feed_forward",
        "self_attention_res_wrapper",
        "feed_forward_res_wrapper",
    ]

    def __init__(
        self, hidden_dim: int = 512, num_heads: int = 4, dropout: float = 0.1
    ) -> None:
        """
        :param hidden_dim: dimention of hidden layer
        :param num_heads: the number of heads of multi-head attention
        :param dropout: dropout ratio
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(hidden_dim, num_heads, dropout=dropout)
        self.feed_forward = FeedForwardNetwork(hidden_dim, dropout)

        # wrapper with residual connection and layer normalization
        self.self_attn_res_wrapper = ResidualNormalizationWrapper(hidden_dim, dropout)
        self.feed_forward_res_wrapper = ResidualNormalizationWrapper(
            hidden_dim, dropout
        )

    def forward(
        self, encoder_input: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, ...]:
        """
        :paran encoder_input: input for encoder [batch_size, max_seq_len, hidden_dim]
        :param mask: attention mask, which masks padding,
            padding mask: [batch_size, 1, max_seq_len]
        :return: encoder_output: [batch_size, max_seq_len, hidden_dim]
            attention: [batch_size, max_seq_len, max_seq_len]
        """
        # self attention with residualnormalization
        encoder_input = self.self_attn_res_wrapper(
            encoder_input,
            lambda encoder_input: self.self_attn(
                encoder_input, encoder_input, encoder_input, mask=mask
            ),
        )
        # feed-forward with residualnormalization
        encoder_output = self.feed_forward_res_wrapper(encoder_input, self.feed_forward)

        return encoder_output


class DecoderLayer(nn.Module):
    """
    One Decoder layer of multi-head attention and feed-forward layer with residual wrapper
    """

    __slot__ = [
        "self_attn",
        "src_attn",
        "feed_forward",
        "self_attn_res_wrapper",
        "src_attn_res_wrapper",
        "feed_forward_res_wrapper",
    ]

    def __init__(
        self, hidden_dim: int = 512, num_heads: int = 8, dropout: float = 0.1
    ) -> None:
        """
        :param hidden_dim: dimention of hidden layer
        :param num_heads: the number of heads of multi-head attention
        :param dropout: dropout ratio
        """
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.src_attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.feed_forward = FeedForwardNetwork(hidden_dim, dropout)

        self.self_attn_res_wrapper = ResidualNormalizationWrapper(hidden_dim, dropout)
        self.src_attn_res_wrapper = ResidualNormalizationWrapper(hidden_dim, dropout)
        self.feed_forward_res_wrapper = ResidualNormalizationWrapper(
            hidden_dim, dropout
        )

    def forward(
        self,
        decoder_input: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """
        :paran decoder_input: input for decoder [batch_size, max_seq_len, hidden_dim]
        :paran encoder_output:: output of encoder [batch_size, max_seq_len, hidden_dim]
        :param src_mask: matrix of padding mask, which is denoted as 1 if PAD_ID is there
            [batch_size, 1, max_seq_len]
        :param tgt_mask: self-attention mask to prevent future information from leaking
            It is usually not necessary for Encoder part [batch_size, max_seq_len, max_seq_len]
        :return: [batch_size, max_seq_len, hidden_dim]
        """
        # self attention with residualnormalization
        decoder_input = self.self_attn_res_wrapper(
            decoder_input,
            lambda decoder_input: self.self_attn(
                decoder_input, decoder_input, decoder_input, mask=tgt_mask
            ),
        )
        # src-tgt attention with residualnormalization
        decoder_input = self.src_attn_res_wrapper(
            decoder_input,
            lambda decoder_input: self.src_attn(
                decoder_input, encoder_output, encoder_output, mask=src_mask
            ),
        )
        # feed-forward with residualnormalization
        decoder_output = self.feed_forward_res_wrapper(decoder_input, self.feed_forward)

        return decoder_output

    def incremental_forward(
        self,
        decoder_input: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
        prev_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """
        incremental forward which output just final states

        :param decoder_input: input for decoder [batch_size, max_seq_len, hidden_dim]
        :paran encoder_output:: output of encoder [batch_size, max_seq_len, hidden_dim]
        :param src_mask: matrix of padding mask, which is denoted as 1 if PAD_ID is there
            [batch_size, 1, max_seq_len]
        :param tgt_mask: self-attention mask to prevent future information from leaking
            It is usually not necessary for Encoder part [batch_size, max_seq_len, max_seq_len]
        :param prev_states: Tuple[
            [batch_size, prev_seq_len, hidden_dim],
            [batch_size, prev_seq_len, hidden_dim],
            [batch_size, prev_seq_len, hidden_dim]
            ]
        :return: [batch_size, 1, hidden_dim]
        """
        assert src_mask is None or encoder_output.size(1) == src_mask.size(-1)
        assert tgt_mask is None or decoder_input.size(1) == tgt_mask.size(1)

        new_states = []

        decoder_input = (
            torch.cat([prev_states[0], decoder_input], 1)
            if prev_states
            else decoder_input
        )
        new_states.append(decoder_input)

        # self attention with residualnormalization
        decoder_input = self.self_attn_res_wrapper.incremental_forward(
            decoder_input,
            lambda decoder_input: self.self_attn(
                decoder_input[:, -1:], decoder_input, decoder_input, mask=tgt_mask
            ),
        )
        decoder_input = (
            torch.cat((prev_states[1], decoder_input), 1)
            if prev_states
            else decoder_input
        )
        new_states.append(decoder_input)

        # src-tgt attention with residualnormalization
        decoder_input = self.src_attn_res_wrapper.incremental_forward(
            decoder_input,
            lambda decoder_input: self.src_attn(
                decoder_input[:, -1:], encoder_output, encoder_output, mask=src_mask
            ),
        )
        decoder_input = (
            torch.cat((prev_states[2], decoder_input), 1)
            if prev_states
            else decoder_input
        )
        new_states.append(decoder_input)

        # feed-forward with residualnormalization
        decoder_output = self.feed_forward_res_wrapper.incremental_forward(
            decoder_input,
            lambda decoder_input: self.feed_forward(decoder_input[:, -1:])
        )

        return decoder_output, new_states
