
import torch
import torch.nn as nn


class CCM(nn.Module):
    """
    Channel Clustering Module: FFT + channel clustering + sparse mask.
    """
    def __init__(self, config):
        super().__init__()
        self.seq_len = config.seq_len
        self.d_c = config.d_c
        self.num_clusters = config.K_c
        self.top_k = config.top_k
        self.eps = 1e-6

        freq_len = self.seq_len // 2 + 1
        self.embedding = nn.Linear(freq_len, self.d_c)
        self.metric = nn.Linear(self.d_c, self.d_c, bias=False)
        self.centroids = nn.Parameter(torch.randn(self.num_clusters, self.d_c))

    def forward(self, x):
        # x: [B, L, C]
        spectrum = torch.fft.rfft(x, dim=1)
        amplitude = torch.abs(spectrum)  # [B, F, C]
        amplitude = amplitude.transpose(1, 2)  # [B, C, F]

        embeddings = self.embedding(amplitude)  # [B, C, d_c]
        projected = self.metric(embeddings)     # [B, C, d_c]

        centers = self.centroids.unsqueeze(0)   # [1, K_c, d_c]
        diff = projected.unsqueeze(2) - centers  # [B, C, K_c, d_c]
        dist = (diff ** 2).sum(dim=-1)          # [B, C, K_c]
        assignments = torch.softmax(-dist, dim=-1)

        mask = torch.matmul(assignments, assignments.transpose(1, 2))  # [B, C, C]
        mask = (mask + mask.transpose(1, 2)) / 2

        topk = min(self.top_k, mask.size(-1))
        values, indices = torch.topk(mask, k=topk, dim=-1)
        sparse_mask = torch.zeros_like(mask)
        sparse_mask.scatter_(-1, indices, values)
        sparse_mask = torch.maximum(sparse_mask, sparse_mask.transpose(1, 2))

        return sparse_mask
