import cv2.cv2 as cv2
import mediapipe as mp


class HandTracking:
    def __init__(self, max_num_hands: int = 1, min_detection_confidence: float = 0.9,
                 min_tracking_confidence: float = 0.9):
        self.max_num_hands: int = max_num_hands
        self.min_detection_confidence: float = min_detection_confidence
        self.min_tracking_confidence: float = min_tracking_confidence
        self.mp_hand: mp.solutions = mp.solutions.hands
        self.mp_draw: mp.solutions = mp.solutions.drawing_utils
        self.hand = self.mp_hand.Hands(max_num_hands=self.max_num_hands,
                                       min_detection_confidence=self.min_detection_confidence,
                                       min_tracking_confidence=self.min_tracking_confidence)
        self.results = None

    def detect_hand(self, image: cv2) -> cv2:
        self.results = self.hand.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if self.results.multi_hand_landmarks:
            for i in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(image, i, self.mp_hand.HAND_CONNECTIONS)

        return image

    def get_hand_landmarks(self, image: cv2) -> list:
        land_marks: list = []

        if self.results.multi_hand_landmarks:
            for index, landmark in enumerate(self.results.multi_hand_landmarks[0].landmark):
                height, width, channel = image.shape
                center_x, center_y = int(landmark.x * width), int(landmark.y * height)
                land_marks.append([index, center_x, center_y])

        return land_marks
