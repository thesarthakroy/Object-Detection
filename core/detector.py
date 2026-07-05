"""
Detector Module.
Wraps OpenCV DNN and YOLOv8 model inference.
"""
import os
import logging
from typing import List, Tuple, Any
import cv2
import numpy as np

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class ObjectDetector:
    """Class handling model instantiations, preprocessing, and inference checks."""

    def __init__(self, model_type: str = "ssd", config: Any = None):
        self.model_type = model_type.lower()
        self.config = config
        self.labels: List[str] = []
        self.model = None
        self.yolo_model = None

        self._load_labels()

        if self.model_type == "ssd":
            self._init_ssd()
        elif self.model_type == "yolo":
            self._init_yolo()
        else:
            logging.error("Unsupported model type: %s. Defaulting to SSD.", self.model_type)
            self.model_type = "ssd"
            self._init_ssd()

    def _load_labels(self) -> None:
        """Loads categories names list from the text configuration file."""
        labels_path = self.config.get("labels_path")
        if os.path.exists(labels_path):
            try:
                with open(labels_path, "r", encoding="utf-8") as f:
                    self.labels = [line.strip() for line in f.readlines() if line.strip()]
                logging.info("Successfully loaded %d class labels.", len(self.labels))
            except Exception as e:
                logging.error("Error reading labels file: %s. Error: %s", labels_path, e)
        else:
            logging.warning("Labels path not found: %s. Using default numerical IDs.", labels_path)

    def _init_ssd(self) -> None:
        """Initializes SSD MobileNet v3 DNN model."""
        weights = self.config.get("ssd_weights")
        config_file = self.config.get("ssd_config")

        if not os.path.exists(weights) or not os.path.exists(config_file):
            err_msg = f"SSD weights or configuration files missing: {weights}, {config_file}"
            logging.error(err_msg)
            raise FileNotFoundError(err_msg)

        logging.info("Loading SSD MobileNet v3: %s", weights)
        self.model = cv2.dnn_DetectionModel(weights, config_file)

        input_size = self.config.get("input_size", [320, 320])
        self.model.setInputSize(input_size[0], input_size[1])
        self.model.setInputScale(1.0 / 127.5)
        self.model.setInputMean((127.5, 127.5, 127.5))
        self.model.setInputSwapRB(True)

        # Check for CUDA acceleration backend preference
        device = self.config.get("device", "cpu").lower()
        if device == "gpu":
            try:
                self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                logging.info("DNN GPU/CUDA Backend initialized successfully.")
            except Exception as e:
                logging.warning("CUDA Backend failed. Defaulting to CPU backend. Error: %s", e)
                self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
                self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        else:
            self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            logging.info("DNN CPU Backend initialized.")

    def _init_yolo(self) -> None:
        """Initializes YOLOv8 model weights."""
        if not YOLO_AVAILABLE:
            logging.warning("YOLOv8 package (ultralytics) is not installed. Defaulting to SSD.")
            self.model_type = "ssd"
            self._init_ssd()
            return

        model_name = self.config.get("yolo_model", "yolov8n.pt")
        logging.info("Loading YOLOv8 weights: %s", model_name)
        
        device = "cuda" if self.config.get("device", "cpu").lower() == "gpu" else "cpu"
        try:
            self.yolo_model = YOLO(model_name)
            self.yolo_model.to(device)
            logging.info("YOLOv8 successfully initialized on device: %s", device)
        except Exception as e:
            logging.error("Failed to load YOLOv8 model: %s", e)
            raise e

    def detect(self, frame: np.ndarray, conf_threshold: float = None, nms_threshold: float = None) -> Tuple[List[int], List[float], List[List[int]]]:
        """Detects objects on the provided frame."""
        if frame is None or frame.size == 0:
            return [], [], []

        if conf_threshold is None:
            conf_threshold = self.config.get("conf_threshold", 0.5)
        if nms_threshold is None:
            nms_threshold = self.config.get("nms_threshold", 0.4)

        class_ids = []
        confidences = []
        bboxes = []

        if self.model_type == "ssd" and self.model is not None:
            try:
                raw_class_ids, raw_confidences, raw_bboxes = self.model.detect(
                    frame, confThreshold=conf_threshold, nmsThreshold=nms_threshold
                )
                if len(raw_class_ids) > 0:
                    class_ids = list(raw_class_ids.flatten())
                    confidences = list(raw_confidences.flatten())
                    bboxes = [list(box) for box in raw_bboxes]
            except Exception as e:
                logging.error("SSD detection error: %s", e)
                
        elif self.model_type == "yolo" and self.yolo_model is not None:
            try:
                results = self.yolo_model(frame, conf=conf_threshold, iou=nms_threshold, verbose=False)
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        w = x2 - x1
                        h = y2 - y1
                        bboxes.append([int(x1), int(y1), int(w), int(h)])
                        confidences.append(float(box.conf[0]))
                        class_ids.append(int(box.cls[0]))
            except Exception as e:
                logging.error("YOLO detection error: %s", e)

        return class_ids, confidences, bboxes

    def get_label_name(self, class_id: int) -> str:
        """Resolves class index to label string."""
        offset = 1 if self.model_type == "ssd" else 0
        idx = class_id - offset
        
        if 0 <= idx < len(self.labels):
            return self.labels[idx]
        return f"ID {class_id}"
