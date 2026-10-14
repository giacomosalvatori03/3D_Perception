import os
from huggingface_hub import hf_hub_download
from ultralytics import YOLO


class CameraDetector:
    """2D Object Detector wrapper using KITTI fine-tuned YOLOv8 from HuggingFace."""

    # Explicit mapping from 2D KITTI classes to the 3 target 3D evaluation classes
    CLASS_MAPPING = {
        "Car": "Car",
        "Van": "Car",  # Van mapped to Car (standard KITTI convention)
        "Pedestrian": "Pedestrian",
        "Person_sitting": "Pedestrian",  # Person_sitting mapped to Pedestrian
        "Cyclist": "Cyclist",
    }

    def __init__(
        self,
        repo_id: str = "dronefreak/kitti-yolov8m",
        filename: str = "model.pt",
        conf_thresh: float = 0.25,
        device: str = "cuda",
    ):
        """Downloads fine-tuned KITTI weights from Hugging Face and initializes YOLOv8."""
        self.conf_thresh = conf_thresh
        self.device = device

        print(
            f" Downloading/Loading KITTI YOLOv8 model from HF: {repo_id}/{filename}..."
        )
        weights_path = hf_hub_download(repo_id=repo_id, filename=filename)
        self.model = YOLO(weights_path)

        # Retrieve dictionary of class names directly from model metadata
        self.class_names = self.model.names
        print(
            f" Loaded KITTI 2D Detector. Registered classes: {self.class_names}"
        )

    def detect(self, image_path: str) -> list[dict]:
        """Runs 2D object detection on an RGB image and returns predictions

        mapped strictly to the 3 target evaluation classes (Car, Pedestrian,
        Cyclist).
        """
        results = self.model.predict(
            source=image_path,
            conf=self.conf_thresh,
            device=self.device,
            verbose=False,
        )[0]

        detections_2d = []
        boxes = results.boxes

        if boxes is None or len(boxes) == 0:
            return detections_2d

        for box in boxes:
            cls_id = int(box.cls[0].cpu().numpy())
            score = float(box.conf[0].cpu().numpy())
            raw_class_name = self.class_names.get(cls_id, "Unknown")

            # Map class name using dictionary; returns None if class is not target (e.g. Tram, Truck, Misc)
            target_class = self.CLASS_MAPPING.get(raw_class_name)

            if target_class is not None:
                xyxy = box.xyxy[0].cpu().numpy().tolist()  # [x1, y1, x2, y2]

                detections_2d.append({
                    "bbox_2d": xyxy,
                    "score": score,
                    "type": target_class,
                })

        return detections_2d