import ctypes
import time
import math

import numpy as np
import HandTracking as ht
import cv2.cv2 as cv2

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


class VolumeController:
    def __init__(self):
        self.min_volume: int = 0
        self.max_volume: int = 0
        self.volume: int = 0
        self.width: ctypes.windll.user32 = ctypes.windll.user32.GetSystemMetrics(0)
        self.height: ctypes.windll.user32 = ctypes.windll.user32.GetSystemMetrics(1)
        self.hand_tracking: ht = ht.HandTracking()
        self.volume_controller()

    def display(self) -> None:
        previous_time: int = 0
        volume_bar: int = 400
        volume_percentage: int = 0

        video: cv2 = cv2.VideoCapture(0)
        video.set(3, self.width)
        video.set(4, self.height)

        while video.isOpened():
            flag, image = video.read()

            if flag:
                hand_image: cv2 = self.hand_tracking.detect_hand(image)
                hand_land_marks: list = self.hand_tracking.get_hand_landmarks(image)

                current_time: time = time.time()
                fps: int = int(1 / (current_time - previous_time))
                previous_time: float = current_time

                if hand_land_marks:
                    x1, y1 = hand_land_marks[4][1], hand_land_marks[4][2]
                    x2, y2 = hand_land_marks[8][1], hand_land_marks[8][2]
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    length = math.hypot(x2 - x1, y2 - y1)
                    volume = np.interp(length, [10, 300], [self.min_volume, self.max_volume])
                    volume_bar = np.interp(length, [10, 300], [400, 150])
                    volume_percentage = np.interp(length, [10, 300], [0, 100])

                    cv2.circle(image, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                    cv2.circle(image, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
                    cv2.line(image, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    cv2.circle(image, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

                    if length < 50:
                        cv2.circle(image, (cx, cy), 15, (255, 255, 255), cv2.FILLED)

                    self.volume.SetMasterVolumeLevel(volume, None)

                cv2.rectangle(image, (50, 150), (85, 400), (0, 255, 0), 3)
                cv2.rectangle(image, (50, int(volume_bar)), (85, 400), (0, 255, 0), cv2.FILLED)
                cv2.putText(image, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
                cv2.putText(image, f"{int(volume_percentage)} %", (40, 450), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                cv2.imshow("Video", hand_image)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

    def volume_controller(self) -> None:
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = cast(interface, POINTER(IAudioEndpointVolume))
        volume_range: list = self.volume.GetVolumeRange()
        self.min_volume, self.max_volume = volume_range[0], volume_range[1]


if __name__ == '__main__':
    VolumeController().display()
