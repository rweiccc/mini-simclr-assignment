import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):

    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):

        batch_size = z1.size(0)
        device = z1.device

        z = torch.cat([z1, z2], dim=0)

        similarity = F.cosine_similarity(
            z.unsqueeze(1),
            z.unsqueeze(0),
            dim=2
        )

        similarity = similarity / self.temperature

        mask = torch.eye(
            2 * batch_size,
            dtype=torch.bool,
            device=device
        )

        similarity = similarity.masked_fill(mask, -1e9)

        positives = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(0, batch_size)
        ]).to(device)

        loss = F.cross_entropy(similarity, positives)

        return loss