from abc import ABC, abstractmethod
from typing import List
from src.detection import Detection3D

class BaseDetector(ABC):
    """
    Abstract base class for 3D object detectors. 
    """
    def __init__(self, conf_threshold: float = 0.5):
        self.conf_threshold = conf_threshold

    @abstractmethod
    def detect(self, sample: dict) -> List[Detection3D]:
        """
        Abstract method to be implemented in subclasses.
        
        Parameters:
            sample (dict): Element returned by the KittiDataset containing 'image', 'points', 'calib', etc.
            
        Returns:
            List[Detection3D]: List of Detection3D objects representing the detected 3D objects.
        """
        pass