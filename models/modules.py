import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingLayer(nn.Module):
    def __init__(
        self,
        vocab,
        hidden_dim: int,
        max_seq_len: int,
        pad_idx: int,
        load_pretrained_embed: bool,
    ):
        super(EmbeddingLayer, self).__init__()
        self.token_embed = Embedding(len(vocab), hidden_dim)
        self.pos_embed = Embedding(max_seq_len, hidden_dim)
        self.vocab_size = len(vocab)
        if load_pretrained_embed:
            self.token_embed = nn.Embedding.from_pretrained(vocab.vectors)
            print("embed loaded.")

    def forward(self, x, pos):
        if len(x.size()) == 2:
            y = self.token_embed(x) + self.pos_embed(pos)
        else:
            y = torch.matmul(x, self.token_embed.weight) + self.pos_embed(pos)

        return y


class EncoderLayer(nn.Module):
    def __init__(self, hidden_dim, h, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(hidden_dim, h, dropout)
        self.pw_ffn = FeedForwardNetwork(hidden_dim, dropout)
        self.sublayer = nn.ModuleList(
            [SublayerConnection(hidden_dim, dropout) for _ in range(2)]
        )

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.pw_ffn)


class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim, h, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(hidden_dim, h, dropout)
        self.src_attn = MultiHeadAttention(hidden_dim, h, dropout)
        self.pw_ffn = FeedForwardNetwork(hidden_dim, dropout)
        self.sublayer = nn.ModuleList(
            [SublayerConnection(hidden_dim, dropout) for _ in range(3)]
        )

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.pw_ffn)

    def incremental_forward(self, x, memory, src_mask, tgt_mask, prev_states=None):
        new_states = []
        m = memory

        x = torch.cat((prev_states[0], x), 1) if prev_states else x
        new_states.append(x)
        x = self.sublayer[0].incremental_forward(
            x, lambda x: self.self_attn(x[:, -1:], x, x, tgt_mask)
        )
        x = torch.cat((prev_states[1], x), 1) if prev_states else x
        new_states.append(x)
        x = self.sublayer[1].incremental_forward(
            x, lambda x: self.src_attn(x[:, -1:], m, m, src_mask)
        )
        x = torch.cat((prev_states[2], x), 1) if prev_states else x
        new_states.append(x)
        x = self.sublayer[2].incremental_forward(
            x, lambda x: self.pw_ffn(x[:, -1:]))
        return x, new_states


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, h, dropout):
        super(MultiHeadAttention, self).__init__()
        assert hidden_dim % h == 0
        self.d_k = hidden_dim // h
        self.h = h
        self.head_projs = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(3)])
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask):
        batch_size = query.size(0)

        # [batch_size, seq_len, num_heads, head_dim] -> [batch_size, num_heads, seq_len, head_dim]
        query, key, value = [
            l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for x, l in zip((query, key, value), self.head_projs)
        ]

        attn_feature, _ = scaled_attention(query, key, value, mask)

        attn_concated = (
            attn_feature.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.h * self.d_k)
        )

        return self.fc(attn_concated)


def scaled_attention(query, key, value, mask):
    d_k = query.size(-1)
    scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(d_k)
    scores.masked_fill_(mask, float("-inf"))
    attn_weight = F.softmax(scores, -1)
    attn_feature = attn_weight.matmul(value)

    return attn_feature, attn_weight


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


class SublayerConnection(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super(SublayerConnection, self).__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        y = sublayer(self.layer_norm(x))
        return x + self.dropout(y)

    def incremental_forward(self, x, sublayer):
        y = sublayer(self.layer_norm(x))
        return x[:, -1:] + self.dropout(y)


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.xavier_uniform_(m.weight)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m
