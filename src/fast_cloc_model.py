import torch
import torch.nn as nn


class FastCLOCsAligner(nn.Module):
    """
    Fast-CLOCs Neural Score Alignment Module.
    Uses LayerNorm for stability across variable batch sizes and internal
    feature scaling for depth and volume dimensions.
    """
    def __init__(self, input_dim: int = 8, hidden_dim: int = 64, dropout_rate: float = 0.1):
        super(FastCLOCsAligner, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.norm2 = nn.LayerNorm(hidden_dim // 2)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.out_proj = nn.Linear(hidden_dim // 2, 1)

    def _scale_features(self, x: torch.Tensor) -> torch.Tensor:
        """Internally normalizes heterogeneous features (Depth and Volume) 
        to prevent them from dominating the gradients of the first linear layer.
        Input layout: [S_3d, S_2d, IoU_2d, Depth, Vol, Class_0, Class_1, Class_2]
     """
        x_scaled = x.clone()
        x_scaled[:, 3] = x_scaled[:, 3] / 80.0  # Normalize Depth (0-80m -> 0-1)
        x_scaled[:, 4] = x_scaled[:, 4] / 20.0  # Normalize Volume (0-20m^3 -> 0-1)
        return x_scaled

    def forward(self, x: torch.Tensor, return_logits: bool = True) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x (torch.Tensor): Feature tensor of shape (N, input_dim).
            return_logits (bool): If True, returns raw logits (for BCEWithLogitsLoss).
                                  If False, returns sigmoided probabilities [0, 1].
        """
        x_norm = self._scale_features(x)
        
        out = self.fc1(x_norm)
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.norm2(out)
        out = self.relu2(out)
        
        logits = self.out_proj(out)
        
        if return_logits:
            return logits
        return torch.sigmoid(logits)


if __name__ == "__main__":
    batch_size = 16
    feature_dim = 8
    dummy_features = torch.randn(batch_size, feature_dim)

    model = FastCLOCsAligner(input_dim=feature_dim)
    model.eval()

    with torch.no_grad():
        logits = model(dummy_features, return_logits=True)
        probs = model(dummy_features, return_logits=False)

    print("FastCLOCsAligner initialized successfully.")
    print(f"Logits shape:        {logits.shape}")
    print(f"Probabilities shape: {probs.shape}")
    print(f"Sample probabilities:\n{probs[:3].squeeze()}")