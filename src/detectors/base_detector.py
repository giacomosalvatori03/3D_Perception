from abc import ABC, abstractmethod
from typing import List
from src.detection import Detection3D

class BaseDetector(ABC):
    """
    Classe base astratta per tutti i detector del progetto.
    """
    def __init__(self, conf_threshold: float = 0.5):
        self.conf_threshold = conf_threshold

    @abstractmethod
    def detect(self, sample: dict) -> List[Detection3D]:
        """
        Metodo astratto da implementare nelle sottoclassi.
        
        Parametri:
            sample (dict): Elemento restituito dal KittiDataset contenente 'image', 'points', 'calib', etc.
            
        Restituisce:
            List[Detection3D]: Lista delle predizioni 3D rilevate.
        """
        pass