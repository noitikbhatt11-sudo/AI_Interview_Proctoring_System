import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import os
from datetime import datetime

from utils.proctor_ai import (
    analyze_frame,
    get_proctor_instance
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Interview Proctor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.block-container{
    padding-top:2rem;
}

.metric-card{
    background:#1F2937;
    border-radius:10px;
    padding:12px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD PROCTOR ENGINE
# ============================================================

proctor = get_proctor_instance()

# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "running": False,
    "candidate_name": "Alex Johnson",
    "role": "AI Engineer",
    "mode": "Webcam",
    "result": None
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🛡 AI Interview Proctor")

st.sidebar.markdown("### Candidate Details")

st.session_state.candidate_name = st.sidebar.text_input(
    "Candidate Name",
    value=st.session_state.candidate_name
)

st.session_state.role = st.sidebar.selectbox(
    "Applied Role",
    [
        "AI Engineer",
        "Machine Learning Engineer",
        "Data Scientist",
        "Software Engineer",
        "Backend Developer",
        "Frontend Developer",
        "Full Stack Developer"
    ],
    index=[
        "AI Engineer",
        "Machine Learning Engineer",
        "Data Scientist",
        "Software Engineer",
        "Backend Developer",
        "Frontend Developer",
        "Full Stack Developer"
    ].index(st.session_state.role)
    if st.session_state.role in [
        "AI Engineer",
        "Machine Learning Engineer",
        "Data Scientist",
        "Software Engineer",
        "Backend Developer",
        "Frontend Developer",
        "Full Stack Developer"
    ] else 0
)

st.session_state.mode = st.sidebar.radio(
    "Monitoring Mode",
    ["Webcam", "Upload Video"],
    index=0 if st.session_state.mode == "Webcam" else 1
)

st.sidebar.divider()

# ============================================================
# SESSION CONTROLS
# ============================================================

col1, col2 = st.sidebar.columns(2)

with col1:

    if st.button(
        "▶ Start",
        use_container_width=True
    ):

        proctor.reset_session()

        st.session_state.running = True

        st.session_state.result = None

        st.success("Session Started")

with col2:

    if st.button(
        "■ Stop",
        use_container_width=True
    ):

        st.session_state.running = False

        st.success("Session Stopped")

st.sidebar.divider()

if st.sidebar.button(
    "🔄 Reset Session",
    use_container_width=True
):

    proctor.reset_session()

    st.session_state.running = False

    st.session_state.result = None

    st.success("Session Reset Successfully")

# ============================================================
# HEADER
# ============================================================

st.title("🛡 AI Interview Proctoring System")

st.caption(
    "YOLOv8 • MediaPipe • OpenCV • Streamlit"
)

status = "🟢 RUNNING" if st.session_state.running else "🔴 STOPPED"

st.markdown(
    f"""
**Candidate:** {st.session_state.candidate_name}

**Role:** {st.session_state.role}

**Status:** {status}
"""
)

st.divider()

# ============================================================
# LIVE MONITORING
# ============================================================

left_col, right_col = st.columns([2.3, 1])

frame_placeholder = left_col.empty()

with right_col:

    st.subheader("📊 Live Statistics")

    trust_metric = st.empty()
    cheat_metric = st.empty()
    warning_metric = st.empty()
    fps_metric = st.empty()
    timer_metric = st.empty()

    st.divider()

    st.subheader("👤 Face Analysis")

    head_metric = st.empty()
    face_metric = st.empty()

    st.divider()

    st.subheader("📦 Objects")

    object_placeholder = st.empty()

    st.divider()

    st.subheader("⚠ Live Logs")

    log_placeholder = st.empty()


# ============================================================
# WEBCAM MODE
# ============================================================

if st.session_state.mode == "Webcam":

    image = st.camera_input(
        "Interview Camera",
        disabled=not st.session_state.running
    )

    if image is not None and st.session_state.running:

        image_bytes = image.getvalue()

        frame = cv2.imdecode(
            np.frombuffer(image_bytes, np.uint8),
            cv2.IMREAD_COLOR
        )

        result = analyze_frame(frame)

        st.session_state.result = result

        rgb = cv2.cvtColor(
            result["frame"],
            cv2.COLOR_BGR2RGB
        )

        frame_placeholder.image(
            rgb,
            channels="RGB",
            use_container_width=True
        )


# ============================================================
# VIDEO MODE
# ============================================================

else:

    uploaded_video = st.file_uploader(
        "Upload Interview Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None and st.session_state.running:

        temp_video = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        temp_video.write(uploaded_video.read())
        temp_video.close()

        cap = cv2.VideoCapture(temp_video.name)

        progress = st.progress(0)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        current = 0

        while cap.isOpened():

            success, frame = cap.read()

            if not success:
                break

            current += 1

            result = analyze_frame(frame)

            st.session_state.result = result

            rgb = cv2.cvtColor(
                result["frame"],
                cv2.COLOR_BGR2RGB
            )

            frame_placeholder.image(
                rgb,
                channels="RGB",
                use_container_width=True
            )

            if total_frames > 0:

                progress.progress(
                    min(current / total_frames, 1.0)
                )

        cap.release()

        os.remove(temp_video.name)


# ============================================================
# DASHBOARD UPDATE
# ============================================================

if st.session_state.result is not None:

    result = st.session_state.result

    trust_metric.metric(
        "Trust Score",
        f"{result['trust_score']}%"
    )

    cheat_metric.metric(
        "Cheating Score",
        result["cheating_score"]
    )

    warning_metric.metric(
        "Warnings",
        result["warning_count"]
    )

    fps_metric.metric(
        "FPS",
        f"{result['fps']:.2f}"
    )

    timer_metric.metric(
        "Timer",
        result["timer"]
    )

    head_metric.metric(
        "Direction",
        result["head"]["direction"]
    )

    face_metric.metric(
        "Faces",
        result["head"]["face_count"]
    )

    object_df = pd.DataFrame({

        "Object":[
            "Phone",
            "Book",
            "Headphone",
            "Laptop",
            "Person",
            "TV"
        ],

        "Count":[
            result["objects"]["phone"],
            result["objects"]["book"],
            result["objects"]["headphone"],
            result["objects"]["laptop"],
            result["objects"]["person"],
            result["objects"]["tv"]
        ]

    })

    object_placeholder.dataframe(
        object_df,
        hide_index=True,
        use_container_width=True
    )

    if result["logs"]:

        log_text = ""

        for log in reversed(result["logs"][-15:]):

            timestamp = log.get("timestamp", "--:--:--")
            severity = log.get("severity", "INFO")
            message = log.get("message", "")

            log_text += (
                f"[{timestamp}] "
                f"{severity}: "
                f"{message}\n"
            )

        log_placeholder.text(log_text)

    else:

        log_placeholder.success(
            "No violations detected."
        )

else:

    frame_placeholder.info(
        "Start the session and capture/upload a frame."
    )

# ============================================================
# SESSION SUMMARY
# ============================================================

st.divider()

if proctor.total_frames > 0:

    summary = proctor.get_session_summary()

    st.header("📑 Session Summary")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Final Trust Score",
            f"{summary['final_trust_score']}%"
        )

        st.metric(
            "Warnings",
            summary["total_warnings"]
        )

    with c2:

        st.metric(
            "Cheating Score",
            summary["final_cheating_score"]
        )

        st.metric(
            "Average FPS",
            summary["average_fps"]
        )

    with c3:

        st.metric(
            "Duration",
            f"{summary['duration_seconds']} sec"
        )

        st.metric(
            "Status",
            summary["status"]
        )

    # ============================================================
    # VIOLATION BREAKDOWN
    # ============================================================

    st.divider()

    st.subheader("📊 Violation Breakdown")

    violation_df = pd.DataFrame({

        "Violation": list(summary["violation_breakdown"].keys()),

        "Count": list(summary["violation_breakdown"].values())

    })

    st.bar_chart(
        violation_df.set_index("Violation")
    )

    # ============================================================
    # SCREENSHOTS
    # ============================================================

    st.divider()

    st.subheader("📸 Violation Screenshots")

    if len(proctor.captured_screenshots) == 0:

        st.info("No screenshots captured.")

    else:

        cols = st.columns(3)

        for i, shot in enumerate(proctor.captured_screenshots):

            with cols[i % 3]:

                if os.path.exists(shot["path"]):

                    st.image(
                        shot["path"],
                        caption=f"{shot['event']} ({shot['timestamp']})",
                        use_container_width=True
                    )

    # ============================================================
    # REPORT DOWNLOADS
    # ============================================================

    st.divider()

    st.subheader("📥 Export Reports")

    col_csv, col_pdf = st.columns(2)

    csv_path = proctor.export_csv_report()

    with open(csv_path, "rb") as f:

        col_csv.download_button(

            "⬇ Download CSV",

            data=f,

            file_name="Interview_Report.csv",

            mime="text/csv",

            use_container_width=True

        )

    try:

        pdf_path = proctor.export_pdf_report()

        with open(pdf_path, "rb") as f:

            col_pdf.download_button(

                "⬇ Download PDF",

                data=f,

                file_name="Interview_Report.pdf",

                mime="application/pdf",

                use_container_width=True

            )

    except Exception as e:

        col_pdf.warning(str(e))

    # ============================================================
    # EVENT HISTORY
    # ============================================================

    st.divider()

    st.subheader("📜 Event History")

    if len(proctor.logs) == 0:

        st.success("No violations recorded.")

    else:

        log_df = pd.DataFrame(proctor.logs)

        st.dataframe(

            log_df,

            use_container_width=True,

            hide_index=True

        )

    # ============================================================
    # FINAL OBJECT COUNTS
    # ============================================================

    if st.session_state.result is not None:

        result = st.session_state.result

        st.divider()

        st.subheader("📦 Final Object Statistics")

        object_df = pd.DataFrame({

            "Object":[

                "Phone",

                "Book",

                "Headphone",

                "Laptop",

                "Person",

                "TV"

            ],

            "Count":[

                result["objects"]["phone"],

                result["objects"]["book"],

                result["objects"]["headphone"],

                result["objects"]["laptop"],

                result["objects"]["person"],

                result["objects"]["tv"]

            ]

        })

        st.table(object_df)

else:

    st.info("Start a session to view reports.")

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
"""
<center>

### 🛡 AI Interview Proctoring System

YOLOv8 • MediaPipe • OpenCV • Streamlit

Developed by Noitik Bhattacharya....

</center>
""",
unsafe_allow_html=True
)
