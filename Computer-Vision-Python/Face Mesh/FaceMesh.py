import cv2.cv2 as cv2
import mediapipe as mp
import time


class FaceMesh:
    def __init__(self, max_num_faces=5):
        self.max_num_faces = max_num_faces
        self.current_time, self.previous_time = 0, 0
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=self.max_num_faces)
        self.spec = self.mp_draw.DrawingSpec(thickness=1, circle_radius=1)

    def display(self) -> None:
        video = cv2.VideoCapture(0)

        while video.isOpened():
            flag, image = video.read()

            if flag:
                new_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(new_image)

                if results.multi_face_landmarks:
                    for i in results.multi_face_landmarks:
                        self.mp_draw.draw_landmarks(image, i, self.mp_face_mesh.FACEMESH_CONTOURS, self.spec, self.spec)

                self.current_time = time.time()
                fps = 1 / (self.current_time - self.previous_time)
                self.previous_time = self.current_time

                cv2.putText(image, f"FPS: {int(fps)}", (10, 70), cv2.FONT_ITALIC, 2, (255, 0, 255), 2)
                cv2.imshow("Video", image)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break


if __name__ == '__main__':
    FaceMesh().display()
