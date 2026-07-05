"""
Analytics Component for VMS Right Panel.
Renders hardware metric cards, category breakdown lists, and a scrollable alarm event logger.
"""
import streamlit as st
from typing import List, Dict, Any

def render_metric_card(label: str, value: str) -> str:
    """Helper to return styled HTML for a single system card."""
    return f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
    """


def render_analytics_panel(stats_placeholder, stats: Dict[str, str], active_tracks: List[Any], detector: Any, unique_objects_count: int, recent_events: List[str]) -> None:
    """Renders Grafana-style system metrics, object breakdown list, and scrollable logs."""
    # Build complete HTML string for the right panel
    panel_html = ""

    # 1. System Metrics Cards Grid
    is_running = st.session_state.running
    cpu_val = stats["cpu"] if is_running else "System Idle"
    ram_val = stats["ram"] if is_running else "System Idle"
    gpu_val = stats["gpu_mem"] if is_running else "System Idle"
    tracking_status = "ACTIVE" if is_running else "System Idle"

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(render_metric_card("CPU Load", cpu_val), unsafe_allow_html=True)
        st.markdown(render_metric_card("GPU Memory", gpu_val), unsafe_allow_html=True)
        st.markdown(render_metric_card("Active Objects", str(len(active_tracks)) if is_running else "System Idle"), unsafe_allow_html=True)

    with col_b:
        st.markdown(render_metric_card("RAM Usage", ram_val), unsafe_allow_html=True)
        st.markdown(render_metric_card("Tracker Status", tracking_status), unsafe_allow_html=True)
        st.markdown(render_metric_card("Unique Targets", str(unique_objects_count) if is_running else "System Idle"), unsafe_allow_html=True)

    # 2. Object Categories Breakdown List
    st.markdown("<h4 style='color: #cbd5e1; border-bottom: 1px solid #1e293b; padding-bottom: 6px; margin-top: 15px;'>📋 Active Object Counts</h4>", unsafe_allow_html=True)
    
    breakdown_html = ""
    if is_running and active_tracks:
        # Group counts by class name
        counts: Dict[str, int] = {}
        for track in active_tracks:
            name = detector.get_label_name(track.class_id)
            counts[name] = counts.get(name, 0) + 1

        breakdown_html += "<ul style='list-style-type: none; padding-left: 0; color: #94a3b8; font-size: 0.95rem;'>"
        for name, cnt in counts.items():
            breakdown_html += f"<li style='margin-bottom: 6px;'><strong style='color: #f1f5f9;'>{name.capitalize()}</strong>: {cnt} active</li>"
        breakdown_html += "</ul>"
    else:
        breakdown_html += "<p style='color: #475569; font-style: italic; font-size: 0.9rem;'>No active targets.</p>"
    st.markdown(breakdown_html, unsafe_allow_html=True)

    # 3. Recent Events Console Log List
    st.markdown("<h4 style='color: #cbd5e1; border-bottom: 1px solid #1e293b; padding-bottom: 6px; margin-top: 15px;'>🛡️ VMS Recent Events Log</h4>", unsafe_allow_html=True)
    
    log_html = '<div class="log-container">'
    if is_running and recent_events:
        # Show newest logs at the top
        for event in reversed(recent_events):
            log_html += f'<div class="log-item">{event}</div>'
    else:
        log_html += '<div style="color: #475569; font-style: italic; text-align: center; margin-top: 60px;">No event log active.</div>'
    log_html += '</div>'
    
    st.markdown(log_html, unsafe_allow_html=True)
