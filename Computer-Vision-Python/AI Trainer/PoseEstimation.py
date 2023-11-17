import cv2.cv2 as cv2
import mediapipe as mp


class PoseEstimation:
    def __init__(self, min_detection_confidence: float = 0.8, min_tracking_confidence: float = 0.8):
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=min_detection_confidence,
                                      min_tracking_confidence=min_tracking_confidence)
        self.results = None

    def detect_pose(self, image):
        new_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(new_image)

        # if self.results.pose_landmarks:
        #     self.mp_draw.draw_landmarks(image, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

    def find_landmarks(self, image):
        landmarks = []

        if self.results.pose_landmarks:
            for index, landmark in enumerate(self.results.pose_landmarks.landmark):
                height, width, channel = image.shape
                center_x, center_y = int(landmark.x * width), int(landmark.y * height)
                landmarks.append([index, center_x, center_y])

        return landmarks
