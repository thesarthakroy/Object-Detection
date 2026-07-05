"""
Unified VMS Platform Entrypoint.
Orchestrates CLI mode and the enterprise Streamlit VMS Web Dashboard.
"""
import os
import sys
import argparse
import logging
import gc
import cv2
import pandas as pd
import numpy as np
from datetime import datetime

from utils.config import AppConfig
from utils.logger import setup_logger
from utils.helpers import (
    inject_custom_css, 
    get_system_stats, 
    draw_prediction_overlays, 
    CSVExporter, 
    send_alert_notification
)
from core.camera import VideoStream, FPSCalculator
from core.detector import ObjectDetector
from core.tracker import ObjectTracker

# Detect Streamlit context
def check_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except ImportError:
        return False

IS_STREAMLIT = check_streamlit()

if IS_STREAMLIT:
    import streamlit as st
    from components.sidebar import render_sidebar
    from components.video_feed import render_video_feed
    from components.analytics import render_analytics_panel


def run_cli(args, config: AppConfig) -> None:
    """Executes VMS pipeline in Command Line Interface mode."""
    # CLI Overrides
    if args.confidence:
        config.set("conf_threshold", args.confidence)
    if args.device:
        config.set("device", args.device)
    if args.model_type:
        config.set("model_type", args.model_type)

    setup_logger("cli_app.log")
    logging.info("Starting CLI VMS Dashboard...")

    # Load resources
    detector = ObjectDetector(model_type=config.get("model_type"), config=config)
    tracker = ObjectTracker(max_age=config.get("tracking_max_age"), min_iou=config.get("tracking_min_iou"))
    exporter = CSVExporter(config.get("output_dir"))
    fps_calc = FPSCalculator()

    width = args.width if args.width else config.get("input_size")[0]
    height = args.height if args.height else config.get("input_size")[1]

    try:
        stream = VideoStream(src=args.source, width=width, height=height).start()
    except Exception as e:
        logging.critical("Failed to start CLI VideoStream. Error: %s", e)
        sys.exit(1)

    logging.info("CLI Stream active. Press 'q' on the OpenCV GUI to close.")
    
    try:
        while not stream.stopped:
            start_time = datetime.now()
            grabbed, frame = stream.read()
            if not grabbed or frame is None:
                logging.info("VideoStream source finished.")
                break

            fps_calc.tick()

            # Detection & Tracking
            class_ids, confidences, bboxes = detector.detect(frame)
            active_tracks = tracker.update(bboxes, class_ids, confidences)
            
            # Export logs
            exporter.append_tracks(active_tracks, detector)
            
            # Alert dispatcher
            if config.get("enable_alerts"):
                alert_classes = config.get("alert_classes", [])
                for track in active_tracks:
                    c_name = detector.get_label_name(track.class_id)
                    if c_name in alert_classes:
                        send_alert_notification(c_name, f"#{track.class_specific_id}", config.get("alert_email"))

            # Draw visual bounding boxes
            annotated_frame = draw_prediction_overlays(frame, active_tracks, detector)
            
            # Performance overlay
            fps_val = fps_calc.get_fps()
            latency = int((datetime.now() - start_time).total_seconds() * 1000)
            cv2.putText(
                annotated_frame, 
                f"FPS: {fps_val:.1f} | Latency: {latency}ms", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (0, 255, 0), 
                2, 
                cv2.LINE_AA
            )

            cv2.imshow("VMS CLI Object Tracker", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        logging.info("Session cancelled.")
    finally:
        stream.stop()
        cv2.destroyAllWindows()
        logging.info("CLI execution closed cleanly.")


def run_streamlit_dashboard(config: AppConfig) -> None:
    """Executes the Streamlit VMS Web Dashboard."""
    # Inject Custom Grafana Styles
    inject_custom_css()

    st.title("🛡️ Enterprise VMS Monitoring Station")
    st.write("Professional object detection, tracking, and telemetry dashboard.")

    # Render Left Control Sidebar (Sidebar Component)
    source, start_clicked, stop_clicked = render_sidebar(config)

    # Initialize layout columns: Center (70%) and Right (30%)
    col_center, col_right = st.columns([7, 3])

    with col_center:
        st.subheader("📺 Camera Live Stream")
        video_container = st.empty()

    with col_right:
        st.subheader("📊 Operational Telemetry")
        stats_container = st.empty()

    # Handle mutual Stop Button logic and release camera resources
    if stop_clicked:
        st.session_state.running = False
        if "stream" in st.session_state and st.session_state.stream is not None:
            st.session_state.stream.stop()
            st.session_state.stream = None
            st.session_state.detector = None
            st.session_state.tracker = None
            st.session_state.exporter = None
            gc.collect()
        st.success("Webcam stream released. VMS Station is now offline.")

    # Handle Start Button logic
    if start_clicked:
        if source is not None:
            # Release any stale streams first
            if "stream" in st.session_state and st.session_state.stream is not None:
                st.session_state.stream.stop()
                st.session_state.stream = None
                gc.collect()

            st.session_state.running = True
            
            # Start new resources in session state
            try:
                st.session_state.stream = VideoStream(src=source, width=640, height=480).start()
                st.session_state.detector = ObjectDetector(model_type=config.get("model_type"), config=config)
                st.session_state.tracker = ObjectTracker(
                    max_age=config.get("tracking_max_age"), 
                    min_iou=config.get("tracking_min_iou")
                )
                st.session_state.exporter = CSVExporter(config.get("output_dir"))
                st.session_state.fps_calc = FPSCalculator()
                st.session_state.unique_tracks = set()
                st.session_state.recent_events = []
            except Exception as e:
                st.error(f"Failed to start video source: {e}")
                st.session_state.running = False
                gc.collect()
        else:
            st.error("No source provided. Upload a video file or enter an RTSP URL.")

    # Stream rendering loop
    if st.session_state.running and "stream" in st.session_state and st.session_state.stream is not None:
        setup_logger("web_app.log")
        logging.info("VMS Live Session started.")
        
        while st.session_state.running:
            start_time = datetime.now()
            grabbed, frame = st.session_state.stream.read()
            
            if not grabbed or frame is None:
                st.session_state.running = False
                break

            # Tick FPS Calculator
            st.session_state.fps_calc.tick()

            # Inference & Tracking
            class_ids, confidences, bboxes = st.session_state.detector.detect(frame)
            active_tracks = st.session_state.tracker.update(bboxes, class_ids, confidences)
            
            # Append CSV log
            st.session_state.exporter.append_tracks(active_tracks, st.session_state.detector)

            # Record Unique IDs & logs events
            for track in active_tracks:
                c_name = st.session_state.detector.get_label_name(track.class_id)
                track_uid = f"{c_name} #{track.class_specific_id}"
                
                if track_uid not in st.session_state.unique_tracks:
                    st.session_state.unique_tracks.add(track_uid)
                    event_str = f"{datetime.now().strftime('%H:%M:%S')} - {c_name.capitalize()} #{track.class_specific_id} detected"
                    st.session_state.recent_events.append(event_str)
                    
                    # Dispatch alerts if enabled
                    if config.get("enable_alerts") and c_name in config.get("alert_classes", []):
                        send_alert_notification(c_name, f"#{track.class_specific_id}", config.get("alert_email"))

            # Calculate FPS and latency
            fps_val = st.session_state.fps_calc.get_fps()
            latency_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            # Render Center Video Feed with custom metadata overlays
            render_video_feed(
                video_container, 
                frame, 
                config, 
                st.session_state.detector, 
                active_tracks, 
                fps_val, 
                latency_ms
            )

            # Render Right Analytics Telemetry
            sys_stats = get_system_stats()
            with col_right:
                render_analytics_panel(
                    stats_container, 
                    sys_stats, 
                    active_tracks, 
                    st.session_state.detector, 
                    len(st.session_state.unique_tracks), 
                    st.session_state.recent_events
                )

        # Cleanup if loop exits naturally
        if "stream" in st.session_state and st.session_state.stream is not None:
            st.session_state.stream.stop()
            st.session_state.stream = None
            st.session_state.detector = None
            st.session_state.tracker = None
            st.session_state.exporter = None
            gc.collect()

    # Render inactive placeholders when stopped
    if not st.session_state.running:
        render_video_feed(video_container, None, config)
        sys_stats = get_system_stats()
        with col_right:
            render_analytics_panel(stats_container, sys_stats, [], None, 0, [])


def main() -> None:
    config = AppConfig()

    if IS_STREAMLIT:
        run_streamlit_dashboard(config)
    else:
        parser = argparse.ArgumentParser(description="VMS Modular CLI parser")
        parser.add_argument("--source", type=str, default="0", help="Camera index or local video path")
        parser.add_argument("--confidence", type=float, help="Confidence filter threshold")
        parser.add_argument("--device", type=str, choices=["cpu", "gpu"], help="Hardware backend")
        parser.add_argument("--model-type", type=str, choices=["ssd", "yolo"], help="Model type selector")
        parser.add_argument("--width", type=int, help="Frame width resizing")
        parser.add_argument("--height", type=int, help="Frame height resizing")
        
        args = parser.parse_args()
        run_cli(args, config)


if __name__ == "__main__":
    main()
