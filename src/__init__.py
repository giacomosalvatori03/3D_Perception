# src/__init__.py
from src.calibration import Calibration
from src.detection import Detection3D
from src.kitti_dataset import KittiDataset
from src.visualizer import Visualizer
from scripts.lidar_evaluator import KittiEvaluator
from scripts.camera_evaluator import CameraEvaluator

# Project 1: Safe optional import for detectors requiring mmdet3d and CUDA dependencies
try:
    from src.detectors import BaseDetector, LidarDetector
except (ImportError, ModuleNotFoundError):
    BaseDetector = None
    LidarDetector = None

# Project 2: 2D Camera Detector (YOLOv8)
try:
    from src.detectors.camera_detector import CameraDetector
except (ImportError, ModuleNotFoundError):
    CameraDetector = None

__all__ = [
    'KittiDataset',
    'Calibration',
    'Detection3D',
    'BaseDetector',
    'LidarDetector',
    'CameraDetector',
    'LidarSparsifier',
    'Visualizer',
    'KittiEvaluator',
    'SampleVisualizer',
    'KittiEvaluator',
    'CameraEvaluator'
]

