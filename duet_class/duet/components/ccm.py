
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    """
    Классическая multi-head self-attention реализация
    """
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, N, d_model]
        B, N, D = x.shape
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.n_heads, self.d_k)
        q, k, v = qkv.unbind(dim=2)  # [B, N, H, d_k]

        q = q.transpose(1, 2)  # [B, H, N, d_k]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)  # [B, H, N, d_k]

        context = context.transpose(1, 2).contiguous().reshape(B, N, D)
        return self.out_proj(context)

class EncoderLayer(nn.Module):
    """
    Один слой encoder'а: attention + FFN
    """
    def __init__(self, d_model, n_heads, d_ff, dropout, activation="gelu"):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        if activation.lower() == "gelu":
            self.activation = nn.GELU()
        elif activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Attention
        x2 = self.attn(x)
        x = self.norm1(x + self.dropout1(x2))

        # FFN
        x2 = self.ffn[0](x)
        x2 = self.ffn[1](x2)
        x2 = self.ffn[2](x2)
        x2 = self.ffn[3](x2)
        x = self.norm2(x + self.dropout2(x2))
        return x

class CCM(nn.Module):
    """
    Channel Clustering Module: стек encoder слоев
    """
    def __init__(self, config):
        super().__init__()
        self.enc_layers = nn.ModuleList([
            EncoderLayer(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
                activation=config.activation
            )
            for _ in range(config.e_layers)
        ])

    def forward(self, x):
        for layer in self.enc_layers:
            x = layer(x)
        return x
