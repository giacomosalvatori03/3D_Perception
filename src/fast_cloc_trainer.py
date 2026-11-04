import os
import glob
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.fast_cloc_dataset import FastCLOCsDataset
from src.fast_cloc_model import FastCLOCsAligner


class FastCLOCsTrainer:
    """
    Trainer pipeline for the Fast-CLOCs score aligner network.
    Uses a strict frame-level split (e.g., 400 train frames, 100 validation frames)
    to prevent data leakage and enable rigorous mAP40 evaluation.
    """
    def __init__(
        self,
        pred_3d_dir: str,
        pred_2d_dir: str,
        gt_3d_dir: str,
        output_checkpoint_dir: str = "models_checkpoints",
        batch_size: int = 256,
        epochs: int = 30,
        lr: float = 0.001,
        val_split: float = 0.2,
        seed: int = 42,
        device: str = None
    ):
        """
        Args:
            pred_3d_dir (str): Path to 3D candidate predictions.
            pred_2d_dir (str): Path to 2D candidate predictions.
            gt_3d_dir (str): Path to 3D Ground Truth annotations.
            output_checkpoint_dir (str): Directory to save model checkpoints and frame split logs.
            batch_size (int): Mini-batch size for training.
            epochs (int): Number of training epochs.
            lr (float): Learning rate for AdamW optimizer.
            val_split (float): Ratio of frames reserved for validation (e.g., 0.2 -> 100/500 frames).
            seed (int): Random seed for reproducible frame splitting.
            device (str, optional): Computation device ('cuda' or 'cpu').
        """
        self.output_checkpoint_dir = output_checkpoint_dir
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        
        os.makedirs(self.output_checkpoint_dir, exist_ok=True)
        
        # Set computation device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f"Device selected: {self.device}")
        
        # Set random seeds for reproducibility
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        # Retrieve all available frame IDs
        all_frame_files = sorted(glob.glob(os.path.join(pred_3d_dir, "*.txt")))
        all_frame_ids = [
            os.path.splitext(os.path.basename(f))[0] for f in all_frame_files
        ]
        
        if not all_frame_ids:
            raise ValueError(f"No prediction text files found in {pred_3d_dir}")

        # Shuffle frame IDs deterministically
        shuffled_frame_ids = all_frame_ids.copy()
        random.shuffle(shuffled_frame_ids)

        # Split frames into Train and Validation sets
        split_idx = int(len(shuffled_frame_ids) * (1.0 - val_split))
        train_frame_ids = shuffled_frame_ids[:split_idx]
        val_frame_ids = shuffled_frame_ids[split_idx:]

        print(f"Frame-Level Split -> Total Frames: {len(all_frame_ids)} | "
              f"Train Frames: {len(train_frame_ids)} | Val Frames: {len(val_frame_ids)}")

        # Save validation frame IDs to text file for evaluator tracking
        val_frames_file = os.path.join(self.output_checkpoint_dir, "val_frames.txt")
        with open(val_frames_file, "w") as f:
            for frame_id in sorted(val_frame_ids):
                f.write(f"{frame_id}\n")
        print(f"Validation frame list exported to: {val_frames_file}")

        # Instantiate separate Datasets based on frame splits
        print("Extracting training candidate pairs...")
        train_dataset = FastCLOCsDataset(
            pred_3d_dir=pred_3d_dir,
            pred_2d_dir=pred_2d_dir,
            gt_3d_dir=gt_3d_dir,
            frame_ids=train_frame_ids
        )

        print("Extracting validation candidate pairs...")
        val_dataset = FastCLOCsDataset(
            pred_3d_dir=pred_3d_dir,
            pred_2d_dir=pred_2d_dir,
            gt_3d_dir=gt_3d_dir,
            frame_ids=val_frame_ids
        )

        print(f"Candidate pairs extracted -> Train pairs: {len(train_dataset)} | Val pairs: {len(val_dataset)}")

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Model, Loss, Optimizer setup
        self.model = FastCLOCsAligner(input_dim=8).to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

    def train_epoch(self) -> float:
        """Executes one training epoch."""
        self.model.train()
        total_loss = 0.0
        
        for features, labels in self.train_loader:
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            predictions = self.model(features)
            loss = self.criterion(predictions, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * features.size(0)
            
        return total_loss / max(1, len(self.train_loader.dataset))

    def validate(self) -> tuple:
        """Evaluates model performance on the frame-level validation split."""
        self.model.eval()
        total_loss = 0.0
        correct_preds = 0
        total_preds = 0
        
        with torch.no_grad():
            for features, labels in self.val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                predictions = self.model(features)
                loss = self.criterion(predictions, labels)
                
                total_loss += loss.item() * features.size(0)
                
                predicted_classes = (predictions >= 0.5).float()
                correct_preds += (predicted_classes == labels).sum().item()
                total_preds += labels.size(0)
                
        val_loss = total_loss / max(1, len(self.val_loader.dataset))
        val_acc = correct_preds / max(1, total_preds)
        return val_loss, val_acc

    def fit(self):
        """Main training loop with checkpointing based on best validation loss."""
        best_val_loss = float("inf")
        best_model_path = os.path.join(self.output_checkpoint_dir, "fast_cloc_best.pth")
        
        print("\n--- Starting Fast-CLOCs Training Loop ---")
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), best_model_path)
                saved_status = "[BEST SAVED]"
            else:
                saved_status = ""
                
            if epoch % 5 == 0 or epoch == 1 or saved_status:
                print(f"Epoch {epoch:02d}/{self.epochs:02d} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val Acc: {val_acc * 100:.2f}% {saved_status}")
                
        print(f"\nTraining complete. Best model checkpoint saved to: {best_model_path}")


if __name__ == "__main__":
    # Example execution script
    trainer = FastCLOCsTrainer(
        pred_3d_dir="experiments/baseline_64beams",
        pred_2d_dir="experiments_camera/yolov8m_baseline_2d",
        gt_3d_dir="data/kitti_validation/label_2",
        epochs=30,
        batch_size=128,
        lr=0.001,
        val_split=0.2,
        seed=42
    )
    trainer.fit()