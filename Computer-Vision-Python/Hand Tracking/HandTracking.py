import cv2
import mediapipe as mp
import time


class HandTracking:
    def __init__(self, static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.static_image_mode, self.max_num_hands, self.min_detection_confidence,
                                         self.min_tracking_confidence)
        self.mp_draw = mp.solutions.drawing_utils

    def detect_video(self):
        video = cv2.VideoCapture(0)
        current_time, previous_time = 0, 0

        while video.isOpened():
            flag, image = video.read()

            if flag:
                self.find_hands(image)
                landmarks = self.find_landmarks(image)

                if landmarks:
                    print(landmarks[5])

                current_time = time.time()
                fps = 1 / (current_time - previous_time)
                previous_time = current_time

                cv2.putText(image, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)
                cv2.imshow("Video", image)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

    def find_hands(self, image):
        new_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(new_image)

        if self.results.multi_hand_landmarks:
            for hand_landmark in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(image, hand_landmark, self.mp_hands.HAND_CONNECTIONS)

    def find_landmarks(self, image, hand_number=0):
        landmarks = []

        if self.results.multi_hand_landmarks:
            for index, landmark in enumerate(self.results.multi_hand_landmarks[hand_number].landmark):
                height, width, channel = image.shape
                center_x, center_y = int(landmark.x * width), int(landmark.y * height)
                landmarks.append([index, center_x, center_y])

        return landmarks


if __name__ == '__main__':
    HandTracking().detect_video()
