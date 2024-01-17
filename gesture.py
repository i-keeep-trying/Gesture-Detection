import numpy as np
import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
poses = mp_pose.Pose(model_complexity=0)

vid = cv2.VideoCapture(0)

while (True):
    ret, frame = vid.read()
    frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = poses.process(frame1)
    lm = results.pose_landmarks

    if lm:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing_styles.get_default_pose_landmarks_style())
        cv2.imshow("Pose", cv2.flip(frame, 1))
        k = cv2.waitKey(5)
        if (k==ord("q")):
            break
cv2.destroyAllWindows()