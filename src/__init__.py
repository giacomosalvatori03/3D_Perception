# src/__init__.py
from src.calibration import Calibration
from src.detection import Detection3D
from src.kitti_dataset import KittiDataset
from src.visualizer import Visualizer

# Safe optional import for detectors requiring mmdet3d and CUDA dependencies
try:
    from src.detectors import BaseDetector, LidarDetector
except (ImportError, ModuleNotFoundError):
    BaseDetector = None
    LidarDetector = None

__all__ = [
    'KittiDataset',
    'Calibration',
    'Detection3D',
    'BaseDetector',
    'LidarDetector',
    'LidarSparsifier',
    'Visualizer',
    'KittiEvaluator',
    'SampleVisualizer'
]

