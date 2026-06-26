import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class ProjectionHead(nn.Module):

    def __init__(self, in_dim=512, hidden_dim=512, out_dim=128):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.layers(x)


class SimCLR(nn.Module):


    def __init__(self, feature_dim=128, pretrained=False):
        super().__init__()

        if pretrained:
            from torchvision.models import ResNet18_Weights
            encoder = resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            encoder = resnet18(weights=None)

        self.encoder = nn.Sequential(
            *list(encoder.children())[:-1]
        )

        self.projector = ProjectionHead(
            in_dim=512,
            hidden_dim=512,
            out_dim=feature_dim
        )

    def encode(self, x):
 
        h = self.encoder(x)
        h = torch.flatten(h, start_dim=1)
        return h

    def forward(self, x):


        h = self.encode(x)
  
        z = self.projector(h)

        z = F.normalize(z, p=2, dim=1)

        return z