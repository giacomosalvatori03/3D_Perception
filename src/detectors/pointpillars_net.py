import torch
import torch.nn as nn
import torch.nn.functional as F

class PillarFeatureNet(nn.Module):
    """Pillar Feature Network (PFN) per generare la pseudo-immagine BEV (1, 64, 496, 432)."""
    def __init__(self, in_channels=9, feat_channels=64):
        super().__init__()
        self.linear = nn.Linear(in_channels, feat_channels, bias=False)
        self.norm = nn.BatchNorm1d(feat_channels, eps=1e-3, momentum=0.01)

    def forward(self, features, coords, grid_shape=(496, 432)):
        P, N, C = features.shape
        x = self.linear(features)
        x = self.norm(x.view(-1, 64)).view(P, N, 64)
        x = F.relu(x)
        x = torch.max(x, dim=1)[0]

        H, W = grid_shape
        canvas = torch.zeros((1, 64, H, W), dtype=x.dtype, device=x.device)
        y_indices = coords[:, 1].long()
        x_indices = coords[:, 2].long()
        canvas[0, :, y_indices, x_indices] = x.t()
        return canvas

class SECFPNBackbone(nn.Module):
    """Backbone SECOND + SECFPN Neck allineata alla struttura OpenMMLab (126 chiavi)."""
    def __init__(self):
        super().__init__()
        # Backbone SECOND (blocks.0, blocks.1, blocks.2)
        block0 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )
        block1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )
        block2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )
        self.blocks = nn.ModuleList([block0, block1, block2])

        # SECFPN Neck (deblocks.0, deblocks.1, deblocks.2)
        deblock0 = nn.Sequential(
            nn.Conv2d(64, 128, 1, stride=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )
        deblock1 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 2, stride=2, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )
        deblock2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=4, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )
        self.deblocks = nn.ModuleList([deblock0, deblock1, deblock2])

    def forward(self, x):
        x0 = self.blocks[0](x)   # (1, 64, 248, 216)
        x1 = self.blocks[1](x0)  # (1, 128, 124, 108)
        x2 = self.blocks[2](x1)  # (1, 256, 62, 54)

        u0 = self.deblocks[0](x0) # (1, 128, 248, 216)
        u1 = self.deblocks[1](x1) # (1, 128, 248, 216)
        u2 = self.deblocks[2](x2) # (1, 128, 248, 216)

        return torch.cat([u0, u1, u2], dim=1) # (1, 384, 248, 216)

class PointPillarsNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pfn = PillarFeatureNet(in_channels=9, feat_channels=64)
        self.backbone = SECFPNBackbone()
        self.conv_cls = nn.Conv2d(384, 18, kernel_size=1)
        self.conv_box = nn.Conv2d(384, 42, kernel_size=1)

    def forward(self, pillar_features, pillar_coords):
        spatial_features = self.pfn(pillar_features, pillar_coords)
        x = self.backbone(spatial_features)
        cls_preds = self.conv_cls(x)
        box_preds = self.conv_box(x)
        return cls_preds, box_preds