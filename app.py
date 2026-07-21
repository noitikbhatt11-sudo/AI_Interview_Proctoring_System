import time
from datetime import datetime
import os
import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from ultralytics import YOLO

# Initialize application directories for saving violation artifacts
os.makedirs("violations/snapshots", exist_ok=True)
os.makedirs("violations/crops", exist_ok=True)

st.set_page_config(page_title="Advanced AI Violation & Zone Analytics Hub", layout="wide")
st.title("🛡️ Next-Gen AI Real-Time Proctoring & Zone Violation Hub")

# Advanced Sidebar Controls
st.sidebar.title("System Controls & AI Tuning")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.45, 0.05)
selected_model = st.sidebar.selectbox("Select YOLO Architecture", ["yolov8n.pt", "yolov8s.pt"])
enable_heatmap = st.sidebar.toggle("Enable Real-time Motion Heatmap", value=True)
enable_zone_check = st.sidebar.toggle("Enable Restricted Polygon Zone", value=True)

# Load YOLO model with Streamlit caching
@st.cache_resource
def load_yolo_model(model_path):
    return YOLO(model_path)

model = load_yolo_model(selected_model)

# Initialize Session State structures
if "violations_log" not in st.session_state:
    st.session_state.violations_log = []
if "heatmap_canvas" not in st.session_state:
    st.session_state.heatmap_canvas = None

class AdvancedVideoProcessor:
    def __init__(self):
        self.prev_time = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        
        # Calculate real-time FPS
        current_time = time.time()
        fps = 1 / (current_time - self.prev_time) if self.prev_time > 0 else 0
        self.prev_time = current_time

        # Initialize heatmap accumulator layer if dimensions change
        if st.session_state.heatmap_canvas is None or st.session_state.heatmap_canvas.shape[:2] != (h, w):
            st.session_state.heatmap_canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # Run YOLO inference with ByteTrack object tracking enabled
        results = model.track(img, persist=True, conf=confidence_threshold, verbose=False)
        annotated_img = results[0].plot()

        # Define a dynamic restricted polygon zone (e.g., center security box)
        # Format: Normalized coordinates scaled to frame resolution
        zone_pts = np.array([
            [int(w * 0.3), int(h * 0.3)],
            [int(w * 0.7), int(h * 0.3)],
            [int(w * 0.7), int(h * 0.8)],
            [int(w * 0.3), int(h * 0.8)]
        ], np.int32)

        if enable_zone_check:
            # Draw boundary overlay for restricted zone
            cv2.polylines(annotated_img, [zone_pts], isClosed=True, color=(0, 0, 255), thickness=2)
            cv2.putText(annotated_img, "RESTRICTED ZONE", (int(w * 0.3), int(h * 0.3) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Scan detected objects for violations and zone intrusions
        for idx, box in enumerate(results[0].boxes):
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf_score = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy() # Bounding box coordinates [x1, y1, x2, y2]
            
            # Center point of bounding box for zone checking
            center_x = int((xyxy[0] + xyxy[2]) / 2)
            center_y = int((xyxy[1] + xyxy[3]) / 2)

            # Update heatmap tracking trail
            if enable_heatmap:
                cv2.circle(st.session_state.heatmap_canvas, (center_x, center_y), 15, (0, 255, 255), -1)

            # Check if object is unauthorized OR intruding into the restriction zone
            unauthorized_classes = ["cell phone", "laptop", "book", "tv", "person"]
            in_zone = cv2.pointPolygonTest(zone_pts, (center_x, center_y), False) >= 0 if enable_zone_check else False

            if cls_name in unauthorized_classes or (enable_zone_check and in_zone):
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                violation_reason = f"Zone Intrusion ({cls_name})" if in_zone else f"Unauthorized Object ({cls_name})"
                
                violation_entry = {
                    "time": timestamp,
                    "violation": violation_reason,
                    "confidence": round(conf_score, 2)
                }
                
                # Prevent duplicate logging spam for the same object instance within short frames
                recent_logs = st.session_state.violations_log
                is_duplicate = any(
                    v['violation'] == violation_reason and 
                    (datetime.now() - datetime.strptime(v['time'], "%Y-%m-%d_%H-%M-%S")).seconds < 3
                    for v in recent_logs
                )

                if not is_duplicate:
                    st.session_state.violations_log.append(violation_entry)
                    
                    # 1. Save Full Frame Snapshot Evidence
                    snapshot_path = f"violations/snapshots/{timestamp}_{cls_name}.jpg"
                    cv2.imwrite(snapshot_path, annotated_img)

                    # 2. Extract and Save Cropped Evidence Thumbnail of Target Violation
                    x1, y1, x2, y2 = map(int, xyxy)
                    # Clip bounds safely inside image frame dimensions
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    if x2 > x1 and y2 > y1:
                        crop_img = img[y1:y2, x1:x2]
                        crop_path = f"violations/crops/crop_{timestamp}_{cls_name}.jpg"
                        cv2.imwrite(crop_path, crop_img)

        # Blend cumulative heatmap over frame if toggled active
        if enable_heatmap and st.session_state.heatmap_canvas is not None:
            # Apply color map and slight fading effect
            heatmap_gray = cv2.cvtColor(st.session_state.heatmap_canvas, cv2.COLOR_BGR2GRAY)
            heatmap_colored = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)
            annotated_img = cv2.addWeighted(annotated_img, 0.7, heatmap_colored, 0.3, 0)
            # Slowly decay heatmap over time to represent active traffic flow
            st.session_state.heatmap_canvas = cv2.addWeighted(st.session_state.heatmap_canvas, 0.98, np.zeros_like(st.session_state.heatmap_canvas), 0.02, 0)

        # Overlay real-time HUD telemetry counters
        cv2.putText(annotated_img, f"FPS: {int(fps)}", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_img, f"Active Logs: {len(st.session_state.violations_log)}", (25, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

# Streamlit WebRTC Component Layout
col_stream, col_analytics = st.columns([1.5, 1])

with col_stream:
    st.subheader("🔴 Live AI Camera Feed & Spatial Engine")
    webrtc_streamer(
        key="advanced-violation-stream",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=AdvancedVideoProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False}
    )

with col_analytics:
    st.subheader("📊 Live Threat Analytics & Frequency")
    if st.session_state.violations_log:
        # Extract violation types for native visual plotting
        violation_types = [v["violation"] for v in st.session_state.violations_log]
        unique_types, counts = np.unique(violation_types, return_counts=True)
        chart_data = {t: c for t, c in zip(unique_types, counts)}
        st.bar_chart(chart_data)
    else:
        st.info("Waiting for first violation event to compile spatial analytics...")

st.markdown("---")
st.subheader("📋 Recorded Violation Incidents & Evidence Vault")

if st.session_state.violations_log:
    st.dataframe(st.session_state.violations_log, use_container_width=True)
    
    # Export CSV Report
    csv_string = "Timestamp,Violation Type,Confidence Score\n"
    for row in st.session_state.violations_log:
        csv_string += f"{row['time']},{row['violation']},{row['confidence']}\n"

    st.download_button(
        label="📥 Download Certified Incident Compliance Report (CSV)",
        data=csv_string,
        file_name=f"compliance_audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    if st.button("🧹 Clear Incident Audit Logs"):
        st.session_state.violations_log = []
        st.session_state.heatmap_canvas = None
        st.rerun()
else:
    st.success("System Secure: Zero active security or zone violations recorded.")
