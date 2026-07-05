"""
Sidebar Component for VMS Control Panel.
Handles stream controllers, threshold sliders, device selectors, and mutual start/stop locks.
"""
import os
import streamlit as st
from utils.config import AppConfig
from core.detector import YOLO_AVAILABLE

def render_sidebar(config: AppConfig):
    """Renders VMS parameters sidebars and start/stop controls.

    Returns:
        tuple: (source_path, start_clicked, stop_clicked)
    """
    st.sidebar.markdown("<h2 style='text-align: center; color: #38bdf8;'>🛠️ VMS Control Panel</h2>", unsafe_allow_html=True)
    st.sidebar.markdown("---")

    # Group 1: Detection Settings
    with st.sidebar.expander("🔍 Detection Settings", expanded=True):
        model_options = ["YOLOv8 (Default)", "SSD MobileNet v3"] if YOLO_AVAILABLE else ["SSD MobileNet v3"]
        model_choice = st.selectbox(
            "Detection Model", 
            model_options,
            index=0 if YOLO_AVAILABLE else 0
        )
        model_type = "yolo" if "YOLO" in model_choice else "ssd"

        conf_threshold = st.slider(
            "Confidence Threshold", 
            min_value=0.1, max_value=1.0, 
            value=float(config.get("conf_threshold")), step=0.05
        )
        nms_threshold = st.slider(
            "NMS Threshold", 
            min_value=0.1, max_value=1.0, 
            value=float(config.get("nms_threshold")), step=0.05
        )

        source_choice = st.selectbox(
            "Input Source", 
            ["Webcam", "Video File Upload", "IP / RTSP Camera"]
        )

        source = 0
        if source_choice == "Video File Upload":
            uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])
            if uploaded_file is not None:
                # Save locally to output folder
                temp_dir = os.path.join(config.get("output_dir"), "temp")
                os.makedirs(temp_dir, exist_ok=True)
                temp_path = os.path.join(temp_dir, "streamlit_temp.mp4")
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.read())
                source = temp_path
            else:
                source = None
                st.info("Upload a video to activate.")
        elif source_choice == "IP / RTSP Camera":
            source = st.text_input("RTSP URL", value="rtsp://192.168.1.100:554/stream1")

    # Group 2: Tracking Parameters
    with st.sidebar.expander("📍 Tracking Configuration", expanded=False):
        st.selectbox("Tracking Algorithm", ["Centroid-IoU (Standard)"])
        max_age = st.number_input(
            "Max Tracking Age (Frames)", 
            min_value=1, max_value=200, 
            value=int(config.get("tracking_max_age"))
        )
        min_iou = st.slider(
            "Min IoU Association", 
            min_value=0.05, max_value=0.9, 
            value=float(config.get("tracking_min_iou")), step=0.05
        )

    # Group 3: Performance Target
    with st.sidebar.expander("⚡ Performance Options", expanded=False):
        device_opt = st.selectbox(
            "Execution Device", 
            ["CPU", "GPU"],
            index=1 if config.get("device") == "gpu" else 0
        )
        device = "gpu" if device_opt == "GPU" else "cpu"
        
        # Display CUDA Acceleration Status
        import torch
        cuda_status = "🟢 CUDA Available" if torch.cuda.is_available() else "🔴 CUDA Unavailable (CPU only)"
        st.caption(f"System Check: {cuda_status}")

    # Group 4: Alerts Settings
    with st.sidebar.expander("🔔 Alarm System", expanded=False):
        enable_alerts = st.checkbox("Enable Alerts", value=config.get("enable_alerts"))
        alert_classes = st.text_input("Alert Classes (comma separated)", value=",".join(config.get("alert_classes")))
        alert_email = st.text_input("Alert Email Target", value=config.get("alert_email"))

    # Group 5: Recording Settings
    with st.sidebar.expander("🎥 Recording & Snapshots", expanded=False):
        enable_recording = st.checkbox("Enable Recording", value=config.get("enable_recording"))
        snapshot_on_detection = st.checkbox("Snapshot on Detection", value=config.get("snapshot_on_detection"))
        save_path = st.text_input("Recordings Save Folder", value=config.get("save_path"))

    # Apply widget settings changes to runtime config
    config.settings["model_type"] = model_type
    config.settings["conf_threshold"] = conf_threshold
    config.settings["nms_threshold"] = nms_threshold
    config.settings["tracking_max_age"] = max_age
    config.settings["tracking_min_iou"] = min_iou
    config.settings["device"] = device
    config.settings["enable_alerts"] = enable_alerts
    config.settings["alert_classes"] = [c.strip() for c in alert_classes.split(",") if c.strip()]
    config.settings["alert_email"] = alert_email
    config.settings["enable_recording"] = enable_recording
    config.settings["snapshot_on_detection"] = snapshot_on_detection
    config.settings["save_path"] = save_path

    st.sidebar.markdown("---")

    # Start & Stop Control buttons with mutual lock mechanisms
    # Running state is stored in st.session_state.running
    if "running" not in st.session_state:
        st.session_state.running = False

    col_start, col_stop = st.sidebar.columns(2)

    with col_start:
        start_btn = st.button(
            "▶️ START", 
            key="start_stream_btn", 
            disabled=st.session_state.running,
            use_container_width=True
        )

    with col_stop:
        stop_btn = st.button(
            "⏹️ STOP", 
            key="stop_stream_btn", 
            disabled=not st.session_state.running,
            use_container_width=True
        )

    return source, start_btn, stop_btn
