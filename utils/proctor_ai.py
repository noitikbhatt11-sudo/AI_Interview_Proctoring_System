import cv2
import numpy as np

def analyze_frame(frame):
    """
    Simulates or executes computer vision checks (e.g., face count, gaze direction).
    """
    faces_detected = 1 
    gaze_centered = True
    
    status_msg = "Normal"
    warning_flag = False
    
    if faces_detected == 0:
        status_msg = "No face detected!"
        warning_flag = True
    elif faces_detected > 1:
        status_msg = "Multiple faces detected!"
        warning_flag = True
    elif not gaze_centered:
        status_msg = "Gaze diverted from screen."
        warning_flag = True
        
    return {
        "status": status_msg,
        "warning": warning_flag,
        "faces": faces_detected
    }
