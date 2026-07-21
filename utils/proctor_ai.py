import csv
from datetime import datetime
import os
import time
import cv2
import numpy as np
from ultralytics import YOLO

from utils.head_pose import HeadPoseEstimator

# Optional import for PDF report generation
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False


class InterviewProctor:
    """
    AI Interview Proctor

    Performs:
    1. YOLO Object Detection
    2. Face Detection & Head Pose Tracking
    3. Event Cooldown & Penalty Scoring
    4. Screenshot Capture on Violations
    5. Enhanced HUD Dashboard (FPS, Session Timer, Color Coding)
    6. CSV & PDF Session Reporting
    """

    def __init__(self, model_path="model/best.pt", confidence_threshold=0.45):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model not found:\n{model_path}")

        print("Loading YOLO Model...")
        self.model = YOLO(model_path)
        print("YOLO Model Loaded Successfully")

        self.confidence = confidence_threshold
        self.head_pose = HeadPoseEstimator()

        self.class_names = self.model.names

        # Class counts
        self.phone_count = 0
        self.book_count = 0
        self.headphone_count = 0
        self.laptop_count = 0
        self.person_count = 0
        self.tv_count = 0

        # Performance & Timing
        self.total_frames = 0
        self.session_start_time = time.time()
        self.prev_frame_time = time.time()
        self.fps = 0.0

        # Scoring
        self.warning_count = 0
        self.cheating_score = 0
        self.trust_score = 100

        # Logs & Screenshots
        self.logs = []
        self.captured_screenshots = []
        self.screenshot_dir = "violations_screenshots"
        os.makedirs(self.screenshot_dir, exist_ok=True)

        # 1. EVENT COOLDOWN LOGIC (seconds between penalizations)
        self.cooldowns = {
            "cell_phone": 3.0,
            "book": 3.0,
            "headphone": 3.0,
            "multiple_persons": 3.0,
            "tv": 5.0,
            "no_face": 2.0,
            "multiple_faces": 3.0,
            "looking_away": 2.0,
        }
        self.last_triggered = {key: 0.0 for key in self.cooldowns}

        # Session aggregate counters
        self.violation_counts = {key: 0 for key in self.cooldowns}

        # Colors for Object Bounding Boxes
        self.colors = {
            "person": (0, 255, 0),
            "cell phone": (0, 0, 255),
            "book": (255, 0, 0),
            "headphone": (255, 255, 0),
            "laptop": (255, 0, 255),
            "tv": (0, 255, 255),
            "default": (200, 200, 200),
        }

        print("Interview Proctor Initialized Successfully")

    def reset_counters(self):
        self.phone_count = 0
        self.book_count = 0
        self.headphone_count = 0
        self.laptop_count = 0
        self.person_count = 0
        self.tv_count = 0

    def _can_trigger_event(self, event_type, current_time):
        """Checks whether the event cooldown period has passed."""
        cooldown = self.cooldowns.get(event_type, 2.0)
        last_time = self.last_triggered.get(event_type, 0.0)
        if (current_time - last_time) >= cooldown:
            self.last_triggered[event_type] = current_time
            return True
        return False

    def capture_screenshot(self, frame, event_name):
        """2. SCREENSHOT CAPTURE FOR VIOLATIONS"""
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        sanitized_event = event_name.lower().replace(" ", "_")
        filepath = os.path.join(
            self.screenshot_dir, f"violation_{sanitized_event}_{now_str}.jpg"
        )
        cv2.imwrite(filepath, frame)

        screenshot_meta = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "event": event_name,
            "path": filepath,
        }
        self.captured_screenshots.append(screenshot_meta)
        return filepath

    def add_event(self, message, severity="INFO", frame=None):
        timestamp = datetime.now().strftime("%H:%M:%S")
        event_entry = {
            "timestamp": timestamp,
            "severity": severity,
            "message": message,
        }
        self.logs.append(event_entry)

        # Capture screenshot for high/medium severity events
        if severity in ["HIGH", "MEDIUM"] and frame is not None:
            self.capture_screenshot(frame, message)

    def draw_box(self, frame, box, class_name, confidence):
        x1, y1, x2, y2 = map(int, box)
        color = self.colors.get(class_name, self.colors["default"])

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} {confidence:.2f}"
        (w, h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )

        label_y = max(y1 - 10, h + 5)
        cv2.rectangle(
            frame,
            (x1, label_y - h - 5),
            (x1 + w + 10, label_y + baseline),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x1 + 5, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    def detect_objects(self, frame):
        self.reset_counters()
        detections = []

        results = self.model.predict(
            frame, conf=self.confidence, verbose=False
        )

        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                class_name = self.class_names[cls]

                detections.append(
                    {"class": class_name, "confidence": conf, "box": xyxy}
                )

                if class_name == "cell phone":
                    self.phone_count += 1
                elif class_name == "book":
                    self.book_count += 1
                elif class_name == "headphone":
                    self.headphone_count += 1
                elif class_name == "laptop":
                    self.laptop_count += 1
                elif class_name == "person":
                    self.person_count += 1
                elif class_name == "tv":
                    self.tv_count += 1

                self.draw_box(frame, xyxy, class_name, conf)

        return frame, detections

    def update_scores(self):
        self.trust_score = max(100 - self.cheating_score, 0)

    def analyze_face(self, frame):
        current_time = time.time()
        result = self.head_pose.estimate(frame)

        face_count = result.get("face_count", 0)
        direction = result.get("head_direction", "Unknown")

        if face_count == 0:
            if self._can_trigger_event("no_face", current_time):
                self.cheating_score += 20
                self.warning_count += 1
                self.violation_counts["no_face"] += 1
                self.add_event("No face detected", "HIGH", frame)

        elif face_count > 1:
            if self._can_trigger_event("multiple_faces", current_time):
                self.cheating_score += 30
                self.warning_count += 1
                self.violation_counts["multiple_faces"] += 1
                self.add_event("Multiple faces detected", "HIGH", frame)

        if direction not in ["Center", "No Face", "Unknown"]:
            if self._can_trigger_event("looking_away", current_time):
                self.cheating_score += 5
                self.warning_count += 1
                self.violation_counts["looking_away"] += 1
                self.add_event(
                    f"Candidate looking {direction}", "MEDIUM", frame
                )

        if "mesh" in result and result["mesh"] is not None:
            frame = self.head_pose.draw(frame, result["mesh"])

        return frame, result

    def analyze_objects(self, frame):
        current_time = time.time()

        if self.phone_count > 0 and self._can_trigger_event(
            "cell_phone", current_time
        ):
            self.cheating_score += 40
            self.warning_count += 1
            self.violation_counts["cell_phone"] += 1
            self.add_event("Mobile Phone Detected", "HIGH", frame)

        if self.book_count > 0 and self._can_trigger_event("book", current_time):
            self.cheating_score += 20
            self.warning_count += 1
            self.violation_counts["book"] += 1
            self.add_event("Book Detected", "MEDIUM", frame)

        if self.headphone_count > 0 and self._can_trigger_event(
            "headphone", current_time
        ):
            self.cheating_score += 25
            self.warning_count += 1
            self.violation_counts["headphone"] += 1
            self.add_event("Headphone Detected", "MEDIUM", frame)

        if self.person_count > 1 and self._can_trigger_event(
            "multiple_persons", current_time
        ):
            self.cheating_score += 35
            self.warning_count += 1
            self.violation_counts["multiple_persons"] += 1
            self.add_event("Multiple Persons Detected", "HIGH", frame)

        if self.tv_count > 0 and self._can_trigger_event("tv", current_time):
            self.cheating_score += 10
            self.warning_count += 1
            self.violation_counts["tv"] += 1
            self.add_event("TV Detected", "LOW", frame)

    def draw_dashboard(self, frame, head_result):
        """4. ENHANCED DASHBOARD (FPS, Timer, HUD, Color Coding)"""
        self.update_scores()
        current_time = time.time()

        # Calculate FPS
        time_diff = current_time - self.prev_frame_time
        if time_diff > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / time_diff)
        self.prev_frame_time = current_time

        # Calculate Session Timer (HH:MM:SS)
        elapsed = int(current_time - self.session_start_time)
        hrs, remainder = divmod(elapsed, 3600)
        mins, secs = divmod(remainder, 60)
        timer_str = f"{hrs:02d}:{mins:02d}:{secs:02d}"

        # Color coding logic based on Trust Score
        if self.trust_score >= 80:
            status_color = (0, 255, 0)  # Green
        elif self.trust_score >= 50:
            status_color = (0, 255, 255)  # Yellow
        else:
            status_color = (0, 0, 255)  # Red

        # Draw semi-transparent HUD overlay panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (320, 190), (20, 20, 20), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        cv2.rectangle(frame, (10, 10), (320, 190), status_color, 2)

        # Dashboard HUD Stats
        cv2.putText(
            frame,
            f"Trust Score : {self.trust_score}%",
            (25, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            status_color,
            2,
        )
        cv2.putText(
            frame,
            f"Cheating Score: {self.cheating_score}",
            (25, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 255),
            1,
        )
        cv2.putText(
            frame,
            f"Warnings : {self.warning_count}",
            (25, 98),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            frame,
            f"Timer : {timer_str}",
            (25, 128),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            1,
        )
        cv2.putText(
            frame,
            f"FPS : {int(self.fps)}",
            (25, 158),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            1,
        )

        return frame

    def analyze_frame(self, frame):
        if frame is None:
            return None
        self.total_frames += 1
        frame, detections = self.detect_objects(frame)
        frame, head_result = self.analyze_face(frame)

        self.analyze_objects(frame)
        frame = self.draw_dashboard(frame, head_result)

        return {
            "frame": frame,
            "detections": detections,
            "logs": self.logs,
            "trust_score": self.trust_score,
            "cheating_score": self.cheating_score,
            "warning_count": self.warning_count,
            "fps": round(self.fps, 1),
            "timer": str(
                datetime.fromtimestamp(
                    time.time() - self.session_start_time
                ).strftime("%M:%S")
            ),
            "objects": {
                "phone": self.phone_count,
                "book": self.book_count,
                "headphone": self.headphone_count,
                "laptop": self.laptop_count,
                "person": self.person_count,
                "tv": self.tv_count,
            },
            "head": {
                "face_count": head_result.get("face_count", 0),
                "direction": head_result.get("head_direction", "Unknown"),
            },
        }

    # 5. SESSION STATISTICS AND ALERT SUMMARIES
    def get_session_summary(self):
        duration_sec = int(time.time() - self.session_start_time)
        return {
            "total_frames": self.total_frames,
            "duration_seconds": duration_sec,
            "final_trust_score": self.trust_score,
            "final_cheating_score": self.cheating_score,
            "total_warnings": self.warning_count,
            "average_fps": round(self.fps, 1),
            "violation_breakdown": self.violation_counts,
            "screenshots_captured": len(self.captured_screenshots),
            "status": "PASSED"
            if self.trust_score >= 70
            else ("NEEDS REVIEW" if self.trust_score >= 40 else "FAILED"),
        }

    # 3. REPORT GENERATION (CSV)
    def export_csv_report(self, filepath="session_report.csv"):
        summary = self.get_session_summary()

        with open(filepath, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)

            # Summary Section
            writer.writerow(["=== INTERVIEW PROCTOR SESSION REPORT ==="])
            writer.writerow(["Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            writer.writerow(["Duration (seconds)", summary["duration_seconds"]])
            writer.writerow(["Final Trust Score", f"{summary['final_trust_score']}%"])
            writer.writerow(["Final Cheating Score", summary["final_cheating_score"]])
            writer.writerow(["Status", summary["status"]])
            writer.writerow(["Total Warnings", summary["total_warnings"]])
            writer.writerow([])

            # Violation Breakdown
            writer.writerow(["=== VIOLATION BREAKDOWN ==="])
            writer.writerow(["Violation Type", "Occurrences"])
            for event, count in summary["violation_breakdown"].items():
                writer.writerow([event, count])
            writer.writerow([])

            # Detailed Event Logs
            writer.writerow(["=== DETAILED EVENT LOGS ==="])
            writer.writerow(["Time", "Severity", "Event Description"])
            for log in self.logs:
                writer.writerow([log["timestamp"], log["severity"], log["message"]])

        return filepath

    # 3. REPORT GENERATION (PDF)
    def export_pdf_report(self, filepath="session_report.pdf"):
        if not HAS_REPORTLAB:
            raise ImportError(
                "ReportLab library not found. Install it via 'pip install reportlab'."
            )

        summary = self.get_session_summary()
        doc = SimpleDocTemplate(filepath, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []

        # Title
        title = Paragraph("<b>AI Interview Proctoring Report</b>", styles["Title"])
        elements.append(title)
        elements.append(Spacer(1, 12))

        # Overview Table
        overview_data = [
            ["Metric", "Value"],
            ["Session Status", summary["status"]],
            ["Final Trust Score", f"{summary['final_trust_score']}%"],
            ["Total Cheating Penalty", summary["final_cheating_score"]],
            ["Total Warnings Issued", summary["total_warnings"]],
            ["Total Duration", f"{summary['duration_seconds']} seconds"],
            ["Screenshots Taken", summary["screenshots_captured"]],
        ]
        t = Table(overview_data, colWidths=[200, 200])
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.navy),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ]
            )
        )
        elements.append(t)
        elements.append(Spacer(1, 18))

        # Log Section
        elements.append(
            Paragraph("<b>Detailed Incident Logs</b>", styles["Heading2"])
        )
        elements.append(Spacer(1, 8))

        log_data = [["Time", "Severity", "Message"]]
        for log in self.logs:
            log_data.append([log["timestamp"], log["severity"], log["message"]])

        if len(log_data) == 1:
            log_data.append(["N/A", "INFO", "No violations recorded."])

        log_table = Table(log_data, colWidths=[80, 80, 280])
        log_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ]
            )
        )
        elements.append(log_table)

        doc.build(elements)
        return filepath

    def reset_session(self):
    self.total_frames = 0
    self.session_start_time = time.time()
    self.prev_frame_time = time.time()
    self.fps = 0.0

    self.warning_count = 0
    self.cheating_score = 0
    self.trust_score = 100

    self.logs = []
    self.captured_screenshots = []

    self.last_triggered = {
        key: 0.0 for key in self.cooldowns
    }

    self.violation_counts = {
        key: 0 for key in self.cooldowns
    }

    self.reset_counters()


# Streamlit / Web UI Global Integration
proctor_instance = None


def get_proctor_instance():
    global proctor_instance
    if proctor_instance is None:
        proctor_instance = InterviewProctor()
    return proctor_instance


def analyze_frame(frame):
    """Wrapper function for Streamlit stream processing."""
    proctor = get_proctor_instance()
    return proctor.analyze_frame(frame)
