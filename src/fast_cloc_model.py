import torch
import torch.nn as nn


class FastCLOCsAligner(nn.Module):
    """
    Fast-CLOCs Neural Score Alignment Module.
    
    Maps 2D-3D candidate feature vectors (confidence scores, 2D IoU, depth, 
    box volume, and one-hot class encodings) to a learned multi-modal confidence score.
    """
    def __init__(self, input_dim: int = 8, hidden_dim: int = 64, dropout_rate: float = 0.1):
        """
        Args:
            input_dim (int): Dimension of the input feature vector (default: 8).
            hidden_dim (int): Number of units in the first hidden layer (default: 64).
            dropout_rate (float): Dropout probability for regularization (default: 0.1).
        """
        super(FastCLOCsAligner, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the score aligner.

        Args:
            x (torch.Tensor): Input tensor of shape (N, input_dim).

        Returns:
            torch.Tensor: Learned confidence scores of shape (N, 1) in range [0, 1].
        """
        return self.net(x)


if __name__ == "__main__":
    # Sanity check with dummy tensor
    batch_size = 16
    feature_dim = 8
    dummy_features = torch.randn(batch_size, feature_dim)

    model = FastCLOCsAligner(input_dim=feature_dim)
    model.eval()

    with torch.no_grad():
        predicted_scores = model(dummy_features)

    print("FastCLOCsAligner initialized successfully.")
    print(f"Input tensor shape:  {dummy_features.shape}")
    print(f"Output tensor shape: {predicted_scores.shape}")
    print(f"Sample output scores:\n{predicted_scores[:3].squeeze()}")