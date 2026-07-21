import cv2
import numpy as np

# Load OpenCV's pre-trained face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load OpenCV's pre-trained full body cascade for person/movement tracking
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

def analyze_frame(frame):
    """
    Executes advanced computer vision checks for face count, multiple people, 
    positional drifting, and device/phone orientation heuristics.
    """
    if frame is None:
        return {
            "status": "No frame received",
            "warning": True,
            "faces": 0,
            "persons": 0
        }

    # Convert frame to grayscale for cascade detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 1. Detect Faces
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(30, 30)
    )
    
    # 2. Detect Full Bodies (Additional Person check)
    bodies = body_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=3, 
        minSize=(50, 50)
    )
    
    faces_detected = len(faces)
    persons_detected = max(faces_detected, len(bodies))
    gaze_centered = True 
    phone_suspected = False

    # Frame dimensions
    frame_height, frame_width = frame.shape[:2]

    # Heuristic 1: Check if the face is centered or drifting away horizontally
    if faces_detected == 1:
        (x, y, w, h) = faces[0]
        face_center_x = x + (w / 2)
        
        # If face is too far to the left or right edge of the frame
        if face_center_x < (frame_width * 0.2) or face_center_x > (frame_width * 0.8):
            gaze_centered = False
            
        # Heuristic 2: Phone/Secondary Device detection approximation via lower-frame object contour/motion blocks
        # Looking for abrupt brightness/edge clusters near the lower desk area where candidates usually hold a phone
        roi_lower = gray[int(frame_height * 0.6):frame_height, int(frame_width * 0.2):int(frame_width * 0.8)]
        edges = cv2.Canny(roi_lower, 100, 200)
        edge_density = np.sum(edges > 0) / (roi_lower.shape[0] * roi_lower.shape[1])
        
        # If sudden high-density edges appear in the lower desk zone, flag a potential device distraction
        if edge_density > 0.15:
            phone_suspected = True

    status_msg = "Normal"
    warning_flag = False
    
    if faces_detected == 0:
        status_msg = "No face detected!"
        warning_flag = True
    elif persons_detected > 1 or faces_detected > 1:
        status_msg = "Multiple persons/faces detected!"
        warning_flag = True
    elif phone_suspected:
        status_msg = "Potential secondary device (phone) detected!"
        warning_flag = True
    elif not gaze_centered:
        status_msg = "Candidate position/gaze drifted from center."
        warning_flag = True
        
    return {
        "status": status_msg,
        "warning": warning_flag,
        "faces": faces_detected,
        "persons": persons_detected
    }
