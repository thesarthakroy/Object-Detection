"""
Helper Utilities Module.
Contains CSS injectors, hardware stats monitors, CSV exporters, and annotation utilities.
"""
import os
import logging
import csv
from datetime import datetime
from typing import List, Tuple, Any, Dict
import cv2
import numpy as np
import psutil

# Detect GPU memory or availability using torch/GPUtil if possible
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def inject_custom_css() -> None:
    """Injects custom CSS to style Streamlit elements like Datadog/Grafana VMS dashboards."""
    import streamlit as st
    st.markdown("""
        <style>
        /* Primary App backgrounds resembling Datadog/Grafana dark theme */
        .stApp {
            background-color: #0b0f19 !important;
            color: #f8fafc !important;
        }
        
        /* Sidebar styling override */
        section[data-testid="stSidebar"] {
            background-color: #111625 !important;
            border-right: 1px solid #1e293b !important;
        }
        
        /* Expander headers styling */
        .st-emotion-cache-p5msec {
            background-color: #1e293b !important;
            border-radius: 4px;
            color: #cbd5e1 !important;
        }

        /* Metric cards styling */
        .metric-card {
            background-color: #111625;
            padding: 15px;
            border-radius: 6px;
            border: 1px solid #1e293b;
            margin-bottom: 12px;
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        }
        
        .metric-label {
            font-size: 0.85rem;
            color: #94a3b8;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .metric-value {
            font-size: 1.75rem;
            font-weight: 700;
            color: #38bdf8;
            margin-top: 4px;
        }

        /* Video feed frame card wrapper */
        .video-card {
            border: 1px solid #1e293b;
            border-radius: 8px;
            overflow: hidden;
            background-color: #0b0f19;
            box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.3);
        }
        
        /* Styled buttons */
        div.stButton > button {
            background-color: #1e293b !important;
            color: #e2e8f0 !important;
            border: 1px solid #334155 !important;
            transition: all 0.2s ease-in-out;
        }
        div.stButton > button:hover {
            border-color: #38bdf8 !important;
            color: #38bdf8 !important;
        }
        
        /* Scrollable console event log */
        .log-container {
            background-color: #090d16;
            border: 1px solid #1e293b;
            border-radius: 6px;
            padding: 10px;
            height: 180px;
            overflow-y: scroll;
            font-family: 'Courier New', Courier, monospace;
            font-size: 0.8rem;
            color: #10b981;
        }
        .log-item {
            margin-bottom: 4px;
            border-bottom: 1px solid #111625;
            padding-bottom: 2px;
        }
        </style>
    """, unsafe_allow_html=True)


def get_system_stats() -> Dict[str, str]:
    """Computes CPU, RAM, and GPU status in real-time."""
    stats = {
        "cpu": f"{psutil.cpu_percent()}%",
        "ram": f"{psutil.virtual_memory().percent}%",
        "gpu": "0%",
        "gpu_mem": "0.0 GB"
    }
    
    if TORCH_AVAILABLE and torch.cuda.is_available():
        stats["gpu"] = "Active"
        try:
            device = torch.cuda.current_device()
            mem_allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
            stats["gpu_mem"] = f"{mem_allocated:.1f} GB"
        except Exception:
            stats["gpu_mem"] = "N/A"
    else:
        stats["gpu"] = "N/A"
        stats["gpu_mem"] = "N/A"

    return stats


def get_stable_color(track_id: int) -> Tuple[int, int, int]:
    """Generates a stable, visually pleasing bounding box color (BGR format) based on Track ID."""
    np.random.seed(track_id)
    # Exclude very dark colors to maintain visibility on dark backgrounds
    color = list(np.random.choice(range(80, 230), size=3))
    return int(color[0]), int(color[1]), int(color[2])


def draw_prediction_overlays(frame: np.ndarray, tracks: List[Any], detector: Any, font_scale: float = 0.5, thickness: int = 1) -> np.ndarray:
    """Draws sleek, non-intrusive bounding boxes and structured class information labels."""
    if frame is None or frame.size == 0:
        return frame

    for track in tracks:
        x, y, w, h = track.bbox
        color = get_stable_color(track.global_id)

        # Draw a clean, thin bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

        # Resolve category name
        class_name = detector.get_label_name(track.class_id)
        label = f"{class_name} #{track.class_specific_id} ({int(track.confidence * 100)}%)"

        # Calculate text dimensions to draw a matching label background card
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

        # Draw box header rectangle above prediction coordinates
        cv2.rectangle(
            frame, 
            (x, y - label_h - 6), 
            (x + label_w + 6, y), 
            color, 
            cv2.FILLED
        )

        # Print text overlays in high-contrast white
        cv2.putText(
            frame, 
            label, 
            (x + 3, y - 4), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            font_scale, 
            (255, 255, 255), 
            thickness=1, 
            lineType=cv2.LINE_AA
        )

    return frame


class CSVExporter:
    """Logs active targets history in real time into CSV tables inside output folder."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.csv_path = os.path.join(self.output_dir, f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        self._write_header()

    def _write_header(self) -> None:
        try:
            with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Track_ID", "Class_Name", "Class_ID", "Confidence", "BBox_xywh"])
        except Exception as e:
            logging.error("Failed to initialize CSV log writer. Error: %s", e)

    def append_tracks(self, tracks: List[Any], detector: Any) -> None:
        """Appends active tracking entries to the CSV file."""
        if not tracks:
            return
        
        timestamp = datetime.now().isoformat()
        try:
            with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for track in tracks:
                    class_name = detector.get_label_name(track.class_id)
                    bbox_str = f"[{track.bbox[0]},{track.bbox[1]},{track.bbox[2]},{track.bbox[3]}]"
                    writer.writerow([
                        timestamp,
                        f"{class_name} #{track.class_specific_id}",
                        class_name,
                        track.class_id,
                        f"{track.confidence:.4f}",
                        bbox_str
                    ])
        except Exception as e:
            logging.error("Failed to append track logs to CSV file: %s", e)


def send_alert_notification(class_name: str, object_id: str, email_target: str = "") -> None:
    """Simulates sending an alert event via log stream / emails when targeted classes are found."""
    alert_msg = f"ALERT EVENT: Target object '{class_name}' with ID '{object_id}' detected at {datetime.now().strftime('%H:%M:%S')}"
    logging.warning(alert_msg)
    
    if email_target:
        # Placeholder representing real email alert dispatch
        logging.info("DISPATCHING Alert email notification to: %s", email_target)
