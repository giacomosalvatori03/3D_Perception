import torch
import torch.nn as nn
import torch.nn.functional as F

class PillarFeatureNet(nn.Module):
    """
    Pillar Feature Network (PFN): converte i pilastri di forma (P, N, 9)
    in feature (P, 64) e le dispone nella pseudo-immagine 2D vista dall'alto (BEV).
    """
    def __init__(self, in_channels=9, feat_channels=64):
        super().__init__()
        self.linear = nn.Linear(in_channels, feat_channels, bias=False)
        self.norm = nn.BatchNorm1d(feat_channels, eps=1e-3, momentum=0.01)

    def forward(self, features, coords, grid_shape=(496, 432)):
        P, N, C = features.shape
        x = self.linear(features)  # (P, N, 64)
        x = self.norm(x.view(-1, 64)).view(P, N, 64)
        x = F.relu(x)
        x = torch.max(x, dim=1)[0]  # Max-pooling lungo gli N punti del pilastro -> (P, 64)

        # Disposizione dei pilastri sulla griglia BEV 2D (Batch=1, Channels=64, H=496, W=432)
        H, W = grid_shape
        canvas = torch.zeros((1, 64, H, W), dtype=x.dtype, device=x.device)
        y_indices = coords[:, 1].long()
        x_indices = coords[:, 2].long()
        canvas[0, :, y_indices, x_indices] = x.t()
        return canvas

class SECFPNBackbone(nn.Module):
    """Backbone convoluzionale 2D con estrazione multi-scala (SECFPN)."""
    def __init__(self):
        super().__init__()
        # Blocco 1: Downsampling x2
        self.block1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )
        # Blocco 2: Downsampling x2
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )
        # Upsampling / Fusion
        self.deblock1 = nn.Sequential(
            nn.ConvTranspose2d(64, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )
        self.deblock2 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)

        u1 = self.deblock1(x1)
        u2 = self.deblock2(x2)
        
        # Concatena le feature a risoluzione unificata -> 256 canali
        return torch.cat([u1, u2], dim=1)

class PointPillarsNet(nn.Module):
    """Architettura PointPillars completa per 3 classi (Car, Pedestrian, Cyclist)."""
    def __init__(self):
        super().__init__()
        self.pfn = PillarFeatureNet(in_channels=9, feat_channels=64)
        self.backbone = SECFPNBackbone()
        
        # Head per la predizione di Classificazione e Box 3D
        # 3 classi x 2 orientamenti anchor = 6 anchor totali per cella
        num_anchors = 6 
        self.conv_cls = nn.Conv2d(256, num_anchors * 3, kernel_size=1)
        self.conv_box = nn.Conv2d(256, num_anchors * 7, kernel_size=1)

    def forward(self, pillar_features, pillar_coords):
        # 1. Generazione Pseudo-Immagine BEV (1, 64, 496, 432)
        spatial_features = self.pfn(pillar_features, pillar_coords)
        
        # 2. Estrazione Feature 2D (1, 256, 248, 216)
        x = self.backbone(spatial_features)
        
        # 3. Predizione Classificazione e Box 3D (7-DoF)
        cls_preds = self.conv_cls(x)
        box_preds = self.conv_box(x)
        
        return cls_preds, box_preds