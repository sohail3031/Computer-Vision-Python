import cv2.cv2 as cv2
import ctypes
import numpy as np
import PoseEstimation as pe
import math


class AITrainer:
    def __init__(self):
        self.width: ctypes.windll.user32 = ctypes.windll.user32.GetSystemMetrics(0)
        self.height: ctypes.windll.user32 = ctypes.windll.user32.GetSystemMetrics(1)
        self.pose_estimation = pe.PoseEstimation()
        self.count = 0
        self.direction = 0

    def start(self):
        video = cv2.VideoCapture(0)

        while video.isOpened():
            # image = cv2.imread("Asserts/2.jpg")
            flag, image = video.read()

            if flag:
                self.pose_estimation.detect_pose(image)
                land_marks = self.pose_estimation.find_landmarks(image)

                if land_marks:
                    # angle = self.find_angle(image, land_marks, 11, 13, 15)
                    angle = self.find_angle(image, land_marks, 12, 14, 16)
                    percentage = np.interp(angle, (130, 210), (0, 100))

                    if percentage == 100:
                        if self.direction == 0:
                            self.count += 0.5
                            self.direction = 1
                    if percentage == 0:
                        if self.direction == 1:
                            self.count += 0.5
                            self.direction = 0

                    cv2.putText(image, f"{int(self.count)}", (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 255), 5)

                image = cv2.resize(image, (self.width, self.height - 50))
                cv2.imshow("Image", image)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

    @staticmethod
    def find_angle(image, landmarks, p1, p2, p3):
        x1, y1 = landmarks[p1][1:]
        x2, y2 = landmarks[p2][1:]
        x3, y3 = landmarks[p3][1:]

        angle = math.degrees(math.atan2(y3 - y1, x3 - x1) - math.atan2(y1 - y2, x1 - x2))

        if angle < 0:
            angle += 360

        cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 3)
        cv2.line(image, (x2, y2), (x3, y3), (255, 255, 255), 3)
        cv2.circle(image, (x1, y1), 50, (0, 0, 255), cv2.FILLED)
        cv2.circle(image, (x2, y2), 50, (0, 0, 255), cv2.FILLED)
        cv2.circle(image, (x3, y3), 50, (0, 0, 255), cv2.FILLED)

        return angle


if __name__ == '__main__':
    AITrainer().start()
