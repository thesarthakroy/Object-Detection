"""
Configuration Management Module.
Defines system default parameters, file path constants, and JSON storage interface.
"""
import os
import json
import logging

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DEFAULT_CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

DEFAULT_SETTINGS = {
    "model_type": "yolo",  # Default is YOLOv8 as requested by the user
    "ssd_weights": os.path.join(MODELS_DIR, "frozen_inference_graph.pb"),
    "ssd_config": os.path.join(MODELS_DIR, "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"),
    "labels_path": os.path.join(MODELS_DIR, "Text.txt"),
    "yolo_model": "yolov8n.pt",
    "conf_threshold": 0.55,
    "nms_threshold": 0.45,
    "input_size": [320, 320],
    "device": "cpu",
    "output_dir": os.path.join(BASE_DIR, "output"),
    "tracking_max_age": 15,
    "tracking_min_iou": 0.15,
    "enable_alerts": False,
    "alert_classes": ["person"],
    "alert_email": "",
    "enable_recording": False,
    "snapshot_on_detection": False,
    "save_path": os.path.join(BASE_DIR, "output", "recordings")
}

class AppConfig:
    """Class to manage loading, modifying, and saving VMS system parameters."""

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        self.config_path = config_path
        self.settings = DEFAULT_SETTINGS.copy()
        self.load()

    def load(self) -> None:
        """Loads configuration settings from JSON file. Defaults back if file is missing."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    user_settings = json.load(f)
                    for k, v in user_settings.items():
                        if k in ["ssd_weights", "ssd_config", "labels_path", "output_dir", "save_path"] and not os.path.isabs(v):
                            user_settings[k] = os.path.join(BASE_DIR, v)
                    self.settings.update(user_settings)
                logging.info("Configuration loaded from %s", self.config_path)
            except Exception as e:
                logging.warning("Failed to parse configuration file. Default settings loaded. Error: %s", e)
        else:
            self.save()

    def save(self) -> None:
        """Saves current settings back to config.json."""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=4)
            logging.info("Configuration saved successfully to %s", self.config_path)
        except Exception as e:
            logging.error("Failed to save configuration. Error: %s", e)

    def get(self, key: str, default=None):
        """Retrieve key value."""
        return self.settings.get(key, default)

    def set(self, key: str, value) -> None:
        """Modify key value and save configuration file."""
        self.settings[key] = value
        self.save()
