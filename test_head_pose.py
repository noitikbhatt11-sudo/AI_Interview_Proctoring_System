import cv2
from utils.head_pose import HeadPoseEstimator

detector = HeadPoseEstimator()

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    result = detector.estimate(frame)

    detector.draw(frame, result["mesh"])

    cv2.putText(
        frame,
        f'Faces : {result["face_count"]}',
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,0),
        2
    )

    cv2.putText(
        frame,
        f'Direction : {result["head_direction"]}',
        (20,80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,0),
        2
    )

    cv2.imshow("Head Pose", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
