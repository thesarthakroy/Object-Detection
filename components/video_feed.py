"""
Video Feed Component.
Manages VMS overlays rendering (model, latency, resolution) and offline stream placeholders.
"""
import streamlit as st
import cv2
import numpy as np
from typing import List, Any
from utils.helpers import draw_prediction_overlays

def draw_vms_metadata(frame: np.ndarray, model_name: str, fps: float, latency_ms: int) -> np.ndarray:
    """Overlays professional system status info on the frame boundaries."""
    if frame is None or frame.size == 0:
        return frame

    h, w, _ = frame.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    text_color = (255, 255, 255)
    shadow_color = (0, 0, 0)
    bg_green = (0, 185, 0)  # Green for LIVE badge

    # 1. Top Right Overlay: LIVE, FPS, Latency
    live_label = "LIVE"
    metrics_label = f"FPS: {fps:.1f} | LATENCY: {latency_ms}ms"
    
    # Calculate text sizes
    (live_w, live_h), _ = cv2.getTextSize(live_label, font, font_scale, thickness)
    (met_w, met_h), _ = cv2.getTextSize(metrics_label, font, font_scale, thickness)

    # Draw a green filled box for LIVE
    live_x = w - met_w - live_w - 30
    cv2.rectangle(frame, (live_x - 5, 10), (live_x + live_w + 5, 10 + live_h + 10), bg_green, cv2.FILLED)
    cv2.putText(frame, live_label, (live_x, 25), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # Draw black shadow for metrics
    cv2.putText(frame, metrics_label, (w - met_w - 14, 26), font, font_scale, shadow_color, thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, metrics_label, (w - met_w - 15, 25), font, font_scale, text_color, thickness, cv2.LINE_AA)

    # 2. Bottom Left Overlay: Current active model type
    model_label = f"MODEL: {model_name.upper()}"
    cv2.putText(frame, model_label, (16, h - 14), font, font_scale, shadow_color, thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, model_label, (15, h - 15), font, font_scale, text_color, thickness, cv2.LINE_AA)

    # 3. Bottom Right Overlay: Output resolution
    res_label = f"RESOLUTION: {w}x{h}"
    (res_w, res_h), _ = cv2.getTextSize(res_label, font, font_scale, thickness)
    cv2.putText(frame, res_label, (w - res_w - 14, h - 14), font, font_scale, shadow_color, thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, res_label, (w - res_w - 15, h - 15), font, font_scale, text_color, thickness, cv2.LINE_AA)

    return frame


def render_video_feed(placeholder, frame: np.ndarray, config: Any, detector: Any = None, tracks: List[Any] = None, fps: float = 0.0, latency_ms: int = 0) -> None:
    """Renders either the annotated video frames or the custom offline placeholder card."""
    if st.session_state.running and frame is not None:
        # Clone frame to prevent side-effects on stream buffer
        processed_frame = frame.copy()

        # Draw predictions
        if detector is not None and tracks is not None:
            processed_frame = draw_prediction_overlays(processed_frame, tracks, detector)

        # Draw professional metadata overlays
        model_name = config.get("model_type")
        processed_frame = draw_vms_metadata(processed_frame, model_name, fps, latency_ms)

        # Render image
        placeholder.image(processed_frame, channels="BGR", use_container_width=True)
    else:
        # Render a professional, clean camera offline placeholder
        placeholder.markdown("""
            <div class="video-card" style="padding: 100px 20px; text-align: center; color: #94a3b8;">
                <div style="font-size: 5rem; margin-bottom: 20px;">📷</div>
                <h3 style="color: #cbd5e1; font-weight: 600; margin-bottom: 8px;">VMS Video Stream Offline</h3>
                <p style="font-size: 0.95rem; color: #64748b; margin-top: 0px;">
                    Control Panel state: <strong>System Idle</strong>.<br>
                    Select inputs and click <strong>Start Stream</strong> to initialize sensor feeds.
                </p>
            </div>
        """, unsafe_allow_html=True)
