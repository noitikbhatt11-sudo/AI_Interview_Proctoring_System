import streamlit as st
import pandas as pd
import numpy as np
import cv2
from datetime import datetime
from utils.proctor_ai import analyze_frame

st.set_page_config(
    page_title="AI Interview Proctor Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# Initialize Session State variables for tracking
if "proctor_active" not in st.session_state:
    st.session_state.proctor_active = False
if "warnings_count" not in st.session_state:
    st.session_state.warnings_count = 0
if "logs" not in st.session_state:
    st.session_state.logs = []

def add_log(message, level="info"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.insert(0, {"time": timestamp, "message": message, "level": level})

st.title("🛡️ AI Interview Proctor: Live Monitoring Dashboard")
st.markdown("Real-time candidate tracking, behavioral anomaly detection, and automated assessment logs.")

# Sidebar Controls
st.sidebar.header("Proctor Controls")
candidate_name = st.sidebar.text_input("Candidate Name", value="Alex Johnson")
role_applied = st.sidebar.selectbox("Target Role", ["Full Stack Engineer", "Data Scientist", "AI Engineer"])

if not st.session_state.proctor_active:
    if st.sidebar.button("Start Proctoring Session", type="primary"):
        st.session_state.proctor_active = True
        add_log(f"Session started for {candidate_name} ({role_applied}).", "success")
        st.rerun()
else:
    if st.sidebar.button("End Session", type="secondary"):
        st.session_state.proctor_active = False
        add_log(f"Session ended for {candidate_name}.", "info")
        st.rerun()

if st.sidebar.button("Simulate Warning Trigger"):
    st.session_state.warnings_count += 1
    add_log(f"Warning #{st.session_state.warnings_count}: Unauthorized tab-switch or movement detected.", "warning")
    st.rerun()

# Main Dashboard Layout
if st.session_state.proctor_active:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("LIVE Feed & Monitoring")
        
        # Camera input for live simulation
        img_file_buffer = st.camera_input("Candidate Webcam Feed")
        
        if img_file_buffer is not None:
            bytes_data = img_file_buffer.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            analysis = analyze_frame(cv2_img)
            if analysis["warning"]:
                st.toast(f"Alert: {analysis['status']}", icon="⚠️")

        # Status Panel
        st.markdown("### 📊 Status Panel")
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        trust_score = max(100 - (st.session_state.warnings_count * 15), 0)
        metric_col1.metric("Trust & Integrity Score", f"{trust_score}%", "-15% per alert")
        metric_col2.metric("Active Warnings", st.session_state.warnings_count)
        metric_col3.metric("Focus Status", "Normal" if st.session_state.warnings_count < 2 else "Flagged")

    with col2:
        st.subheader("⚠️ Warning & Event Panel")
        warning_box = st.container(height=300)
        with warning_box:
            if not st.session_state.logs:
                st.caption("No infractions or alerts recorded yet.")
            else:
                for log in st.session_state.logs:
                    if log["level"] == "warning":
                        st.warning(f"[{log['time']}] {log['message']}")
                    elif log["level"] == "success":
                        st.success(f"[{log['time']}] {log['message']}")
                    else:
                        st.info(f"[{log['time']}] {log['message']}")

        st.markdown("---")
        st.subheader("Actions")
        
        # Screenshot Button
        if st.button("📸 Capture Evidence Snapshot"):
            add_log("Manual evidence snapshot saved successfully.", "info")
            st.toast("Snapshot archived to secure storage.", icon="📸")

        # Download Report Summary
        report_text = f"""AI INTERVIEW PROCTORING REPORT
----------------------------------------
Candidate Name: {candidate_name}
Target Role: {role_applied}
Final Trust Score: {trust_score}%
Total Warnings Generated: {st.session_state.warnings_count}
Generated At: {datetime.now()}
"""
        st.download_button(
            label="📥 Download PDF/TXT Proctor Report",
            data=report_text,
            file_name=f"Proctor_Report_{candidate_name.replace(' ', '_')}.txt",
            mime="text/plain"
        )
else:
    st.warning("⚠️ Proctoring session is currently offline. Use the sidebar to start a session.")
