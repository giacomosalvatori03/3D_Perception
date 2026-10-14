# src/detectors/__init__.py
from .base_detector import BaseDetector

# Safe import for LidarDetector (requires mmdet3d and PyTorch CUDA)
try:
    from .lidar_detector import LidarDetector
except (ImportError, ModuleNotFoundError):
    LidarDetector = None

# Safe import for CameraDetector (requires ultralytics and huggingface_hub)
try:
    from .camera_detector import CameraDetector
except (ImportError, ModuleNotFoundError):
    CameraDetector = None

__all__ = ["BaseDetector", "LidarDetector", "CameraDetector"]