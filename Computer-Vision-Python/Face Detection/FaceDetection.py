import cv2.cv2 as cv2
import mediapipe as mp
import time


class FaceDetection:
    def __init__(self, min_detection=0.5):
        self.min_detection = min_detection

        self.mp_face = mp.solutions.face_detection
        self.face = self.mp_face.FaceDetection(self.min_detection)
        self.mp_draw = mp.solutions.drawing_utils

    def display(self):
        video = cv2.VideoCapture(0)

        current_time, previous_time = 0, 0

        while video.isOpened():
            flag, frame = video.read()

            if flag:
                self.face_detection(frame)

                current_time = time.time()
                fps = 1 / (current_time - previous_time)
                previous_time = current_time

                cv2.putText(frame, f"FSP: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 255), 1)
                cv2.imshow("Video", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

    def face_detection(self, image):
        new_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face.process(new_image)

        if results.detections:
            for index, detection in enumerate(results.detections):
                height, width, channel = image.shape
                box = detection.location_data.relative_bounding_box
                data = int(box.xmin * width), int(box.ymin * height), int(box.width * width), int(box.height * height)

                cv2.putText(image, f"{int(detection.score[0] * 100)}%", (data[0], data[1] - 20), 6,
                            cv2.FONT_HERSHEY_PLAIN, (255, 0, 255), 1)
                self.draw_edges(image, data)

    def draw_edges(self, image, data):
        x, y, width, height = data
        x1, y1 = x + width, y + height

        cv2.rectangle(image, data, (255, 0, 255), 1)

        # Top Left x, y
        cv2.line(image, (x, y), (x + 30, y), (255, 0, 255), 5)
        cv2.line(image, (x, y), (x, y + 30), (255, 0, 255), 5)
        # Top Right x1, y
        cv2.line(image, (x1, y), (x1 - 30, y), (255, 0, 255), 5)
        cv2.line(image, (x1, y), (x1, y + 30), (255, 0, 255), 5)
        # Bottom Left x, y1
        cv2.line(image, (x, y1), (x + 30, y1), (255, 0, 255), 5)
        cv2.line(image, (x, y1), (x, y1 - 30), (255, 0, 255), 5)
        # Top Right x1, y1
        cv2.line(image, (x1, y1), (x1 - 30, y1), (255, 0, 255), 5)
        cv2.line(image, (x1, y1), (x1, y1 - 30), (255, 0, 255), 5)


if __name__ == '__main__':
    FaceDetection(0.75).display()
