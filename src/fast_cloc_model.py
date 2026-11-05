import torch
import torch.nn as nn


class FastCLOCsAligner(nn.Module):
    """
    Fast-CLOCs Neural Score Alignment Module.
    Outputs raw logits to support BCEWithLogitsLoss with pos_weight class balancing.
    """
    def __init__(self, input_dim: int = 8, hidden_dim: int = 64, dropout_rate: float = 0.1):
        super(FastCLOCsAligner, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


if __name__ == "__main__":
    batch_size = 16
    feature_dim = 8
    dummy_features = torch.randn(batch_size, feature_dim)

    model = FastCLOCsAligner(input_dim=feature_dim)
    model.eval()

    with torch.no_grad():
        logits = model(dummy_features)
        probs = torch.sigmoid(logits)

    print("FastCLOCsAligner initialized successfully.")
    print(f"Logits shape:      {logits.shape}")
    print(f"Sample probabilities:\n{probs[:3].squeeze()}")