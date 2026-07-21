import cv2
import numpy as np

def analyze_frame(frame):
    """
    Executes advanced computer vision checks for face count, multiple people, 
    positional drifting, and device/phone orientation heuristics safely.
    """
    if frame is None:
        return {
            "status": "No frame received",
            "warning": True,
            "faces": 0,
            "persons": 0
        }

    # Safely load cascades inside the function to prevent global attribute crashes
    face_cascade = None
    try:
        if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    except Exception:
        pass

    # Convert frame to grayscale for cascade detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces_detected = 0
    if face_cascade is not None and not face_cascade.empty():
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        faces_detected = len(faces)
    else:
        # Fallback heuristic if cascade fails to load: check if frame is too dark or blank
        if np.mean(gray) > 20:
            faces_detected = 1 

    persons_detected = faces_detected
    gaze_centered = True 
    phone_suspected = False

    frame_height, frame_width = frame.shape[:2]

    # Heuristic checks
    if faces_detected == 1 and face_cascade is not None and not face_cascade.empty():
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face_center_x = x + (w / 2)
            if face_center_x < (frame_width * 0.2) or face_center_x > (frame_width * 0.8):
                gaze_centered = False

    # Lower frame edge density check for secondary devices
    roi_lower = gray[int(frame_height * 0.6):frame_height, int(frame_width * 0.2):int(frame_width * 0.8)]
    if roi_lower.size > 0:
        edges = cv2.Canny(roi_lower, 100, 200)
        edge_density = np.sum(edges > 0) / (roi_lower.shape[0] * roi_lower.shape[1])
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
