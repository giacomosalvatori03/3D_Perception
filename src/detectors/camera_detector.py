import os
from huggingface_hub import hf_hub_download
from ultralytics import YOLO


class CameraDetector:
    """2D Object Detector wrapper using KITTI fine-tuned YOLOv8 from HuggingFace."""

    # Strictly target classes evaluated in the KITTI 3D Benchmark
    EVAL_TARGET_CLASSES = {"Car", "Pedestrian", "Cyclist"}

    def __init__(
        self,
        repo_id: str = "keremberke/yolov8s-kitti",
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
        print(f" Loaded KITTI 2D Detector. Registered classes: {self.class_names}")

    def detect(self, image_path: str) -> list[dict]:
        """
        Runs 2D object detection on an RGB image and returns predictions 
        filtered strictly to the 3 target evaluation classes (Car, Pedestrian,Cyclist).
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
            class_name = self.class_names.get(cls_id, "Unknown")

            # Optional mapping: Van -> Car (common in KITTI evaluation pipeline)
            if class_name == "Van":
                class_name = "Car"

            # Filter strictly for the 3 target classes
            if class_name in self.EVAL_TARGET_CLASSES:
                xyxy = box.xyxy[0].cpu().numpy().tolist()  # [x1, y1, x2, y2]

                detections_2d.append({
                    "bbox_2d": xyxy,
                    "score": score,
                    "type": class_name,
                })

        return detections_2d