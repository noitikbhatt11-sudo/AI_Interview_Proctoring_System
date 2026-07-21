# 🛡️ AI Interview Proctoring System

## 📖 Project Introduction

The **AI Interview Proctoring System** is an intelligent web-based application designed to automate the monitoring of online technical interviews and remote assessments using Artificial Intelligence and Computer Vision. It continuously analyzes live webcam feeds or uploaded interview videos to identify suspicious activities that may indicate unfair practices during an interview.

The system combines **YOLOv8** for real-time object detection and **MediaPipe Face Mesh** for face detection and head pose estimation to monitor candidate behavior. It detects prohibited objects such as mobile phones, books, headphones, televisions, and multiple persons, while also tracking head movement, face visibility, and looking-away events. Based on these observations, the application dynamically calculates a **Trust Score** and **Cheating Score**, providing interviewers with an objective assessment of candidate integrity.

To enhance transparency and post-interview analysis, the system automatically records violations, captures screenshots of suspicious events, and generates detailed **CSV** and **PDF** reports containing session statistics, violation history, and performance metrics.

Developed using **Python**, **Streamlit**, **OpenCV**, **YOLOv8**, and **MediaPipe**, this project demonstrates the practical application of Artificial Intelligence in online recruitment, remote examinations, and secure digital assessments.
---

## 🚀 Features

- 🎥 Real-time webcam monitoring
- 📹 Interview video upload support
- 🤖 YOLOv8 object detection
- 👤 MediaPipe face and head pose estimation
- 📱 Mobile phone detection
- 📚 Book detection
- 🎧 Headphone detection
- 👥 Multiple person detection
- 📺 TV/Monitor detection
- 👀 Head direction tracking
- ❌ No-face detection
- ⚠️ Automatic cheating score calculation
- ✅ Dynamic trust score calculation
- 📸 Automatic screenshot capture during violations
- 📊 Live dashboard with metrics
- 📝 Detailed violation logs
- 📈 Session analytics
- 📄 CSV report generation
- 📑 PDF report generation
- 🎯 FPS monitoring
- ⏱ Session timer

---

# 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| Language | Python 3.11 |
| Framework | Streamlit |
| Computer Vision | OpenCV |
| Object Detection | YOLOv8 (Ultralytics) |
| Face Tracking | MediaPipe |
| Machine Learning | PyTorch |
| Data Handling | Pandas, NumPy |
| Visualization | Plotly |
| Video Streaming | streamlit-webrtc |
| Report Generation | ReportLab |

---

# 📂 Project Structure

```
AI_Interview_Proctoring_System/
│
├── app.py
├── README.md
├── requirements.txt
├── runtime.txt
├── test_head_pose.py
│
├── model/
│   └── best.pt
│
├── utils/
│   ├── __init__.py
│   ├── head_pose.py
│   └── proctor_ai.py
│
└── violations_screenshots/
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/noitikbhatt11-sudo/AI_Interview_Proctoring_System.git
```

Move into the project directory

```bash
cd AI_Interview_Proctoring_System
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

# 📦 Required Python Version

Python 3.11+

---

# ▶️ How It Works

1. Start the Streamlit application.
2. Enter candidate details.
3. Choose Webcam or Upload Video mode.
4. Start monitoring.
5. The system continuously:
   - Detects prohibited objects.
   - Tracks face and head movement.
   - Calculates trust score.
   - Detects suspicious behaviour.
   - Captures screenshots on violations.
6. At the end of the session:
   - Download CSV Report
   - Download PDF Report
   - View complete violation history
   - Review session statistics

---

# 📊 Dashboard Metrics

The dashboard displays:

- Trust Score
- Cheating Score
- Warning Count
- FPS
- Session Timer
- Face Count
- Head Direction
- Object Counts
- Live Logs

---

# 🚨 Detected Violations

The system currently detects:

- Mobile Phone
- Book
- Headphones
- Multiple Persons
- TV/Monitor
- No Face
- Multiple Faces
- Looking Away

---

# 📈 Reports Generated

### CSV Report

Contains:

- Session Summary
- Trust Score
- Cheating Score
- Duration
- Warnings
- Violation Breakdown
- Event Logs

### PDF Report

Contains:

- Session Overview
- Final Status
- Trust Score
- Cheating Score
- Screenshots Count
- Complete Incident Logs

---

# 📸 Automatic Screenshot Capture

Whenever a HIGH or MEDIUM severity violation occurs, the system automatically captures and stores a screenshot inside:

```
violations_screenshots/
```

---

# 🧠 AI Models Used

### YOLOv8

Used for detecting:

- Person
- Cell Phone
- Book
- Laptop
- Headphones
- TV

### MediaPipe Face Mesh

Used for:

- Face Detection
- Face Count
- Head Pose Estimation
- Looking Direction

---

# 🎯 Trust Score Calculation

The system starts with:

```
Trust Score = 100
```

Each violation deducts points based on severity.

Examples:

| Violation | Penalty |
|------------|----------|
| Mobile Phone | -40 |
| Multiple Persons | -35 |
| Multiple Faces | -30 |
| Headphones | -25 |
| Book | -20 |
| No Face | -20 |
| Looking Away | -5 |
| TV | -10 |

---

# 📸 Sample Screens

- Live Monitoring Dashboard
- Violation Detection
- Session Summary
- PDF Report
- CSV Report

(Add screenshots here.)

---

# 💡 Future Enhancements

- Voice activity detection
- Eye gaze tracking
- Lip movement analysis
- Browser tab monitoring
- Multi-camera support
- Cloud database integration
- Admin dashboard
- Candidate authentication
- Face recognition
- AI interview analytics

---

# 📚 Applications

- Online Recruitment
- Campus Placements
- Remote Hiring
- Certification Exams
- University Online Assessments
- Skill Assessments
- Coding Interviews

---

# 👨‍💻 Author

**Noitik Bhattacharya**

MCA (Hons.) Data Science

Lovely Professional University

GitHub:
https://github.com/noitikbhatt11-sudo

---

# 📜 License

This project is developed for educational and research purposes as part of an MCA Major Project.

---

# ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub.

---

# 🙏 Acknowledgements

- Ultralytics YOLOv8
- MediaPipe
- OpenCV
- Streamlit
- PyTorch
- ReportLab
- Plotly
