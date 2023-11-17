import cv2
import mediapipe as mp
import time


class PoseEstimation:
    def __init__(self):
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()

    def start(self):
        video = cv2.VideoCapture("Resources/Woman - 63241.mp4")
        current_time, previous_time = 0, 0

        while video.isOpened():
            flag, image = video.read()

            if flag:
                self.detect_pose(image)
                landmarks = self.find_landmarks(image)

                print(landmarks)

                current_time = time.time()
                fps = 1 / (current_time - previous_time)
                previous_time = current_time

                cv2.putText(img=image, text=str(int(fps)), fontFace=5, fontScale=5, thickness=5, color=(255, 0, 255),
                            org=(10, 70))
                cv2.imshow("Video", image)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

    def detect_pose(self, image):
        new_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(new_image)

        if self.results.pose_landmarks:
            self.mp_draw.draw_landmarks(image, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

    def find_landmarks(self, image):
        landmarks = []

        if self.results.pose_landmarks:
            for index, landmark in enumerate(self.results.pose_landmarks.landmark):
                height, width, channel = image.shape
                center_x, center_y = int(landmark.x * width), int(landmark.y * height)
                landmarks.append([index, center_x, center_y])

        return landmarks


if __name__ == '__main__':
    PoseEstimation().start()
