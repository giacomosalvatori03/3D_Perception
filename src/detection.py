import numpy as np

class Detection3D:
    """
    Struttura dati standardizzata per rappresentare una Bounding Box 3D predetta.
    Tutti i detector dei 3 progetti restituiranno istanze di questa classe.
    """
    def __init__(
        self,
        obj_type: str,
        dimensions_3d: np.ndarray, # [h, w, l] in metri
        location_3d: np.ndarray,   # [x, y, z] nel sistema di riferimento fotocamera
        rotation_y: float,         # Angolo Yaw in radianti [-pi, pi]
        score: float = 1.0,        # Punteggio di confidenza [0.0 - 1.0]
        bbox_2d: np.ndarray = None # [xmin, ymin, xmax, ymax] opzionale o proiettato
    ):
        self.type = obj_type
        self.dimensions_3d = np.array(dimensions_3d, dtype=np.float32)
        self.location_3d = np.array(location_3d, dtype=np.float32)
        self.rotation_y = float(rotation_y)
        self.score = float(score)
        self.bbox_2d = np.array(bbox_2d, dtype=np.float32) if bbox_2d is not None else None

    def __repr__(self):
        return (f"Detection3D(type='{self.type}', score={self.score:.2f}, "
                f"loc={self.location_3d.round(2).tolist()}, dim={self.dimensions_3d.round(2).tolist()}, "
                f"rot_y={self.rotation_y:.2f})")