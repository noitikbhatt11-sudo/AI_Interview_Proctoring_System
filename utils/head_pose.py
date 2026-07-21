import cv2
import mediapipe as mp
import numpy as np


class HeadPoseEstimator:
    """
    Detects face count and estimates head direction
    using MediaPipe Face Mesh.
    """

    def __init__(self):

        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.drawer = mp.solutions.drawing_utils

    def estimate(self, frame):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(rgb)

        output = {
            "face_count": 0,
            "head_direction": "No Face",
            "warning": False,
            "mesh": results.multi_face_landmarks
        }

        if not results.multi_face_landmarks:
            output["warning"] = True
            return output

        faces = results.multi_face_landmarks

        output["face_count"] = len(faces)

        if len(faces) > 1:
            output["warning"] = True

        face = faces[0]

        h, w, _ = frame.shape

        nose = face.landmark[1]
        left_cheek = face.landmark[234]
        right_cheek = face.landmark[454]
        forehead = face.landmark[10]
        chin = face.landmark[152]

        nose_x = nose.x * w
        nose_y = nose.y * h

        left_x = left_cheek.x * w
        right_x = right_cheek.x * w

        forehead_y = forehead.y * h
        chin_y = chin.y * h

        center_x = (left_x + right_x) / 2
        center_y = (forehead_y + chin_y) / 2

        dx = nose_x - center_x
        dy = nose_y - center_y

        horizontal_threshold = 25
        vertical_threshold = 20

        if dx > horizontal_threshold:
            direction = "Right"

        elif dx < -horizontal_threshold:
            direction = "Left"

        elif dy > vertical_threshold:
            direction = "Down"

        elif dy < -vertical_threshold:
            direction = "Up"

        else:
            direction = "Center"

        output["head_direction"] = direction

        if direction != "Center":
            output["warning"] = True

        return output

    def draw(self, frame, mesh):

        if mesh is None:
            return frame

        for face in mesh:

            self.drawer.draw_landmarks(
                frame,
                face,
                self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.drawer.DrawingSpec(
                    color=(0,255,0),
                    thickness=1,
                    circle_radius=1
                )
            )

        return frame
