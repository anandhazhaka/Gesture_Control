import cv2
import mediapipe as mp
import numpy as np
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List
from collections import deque
import signal


@dataclass
class GestureConfig:
    COOLDOWN_FRAMES: int = 15
    SWIPE_THRESHOLD: float = 0.15
    SWIPE_FRAMES: int = 5
    MIN_SCALE: float = 0.5
    MAX_SCALE: float = 3.0
    TARGET_SIZE: Tuple[int, int] = (800, 600)


class HandGestureDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.hand_history = deque(maxlen=5)

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.hands.process(rgb_frame)

    def draw_landmarks(self, frame, hand_landmarks):
        self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)


class ImageController:
    def __init__(self, image_directory: str, config: GestureConfig):
        self.config = config
        self.image_files = self._load_image_files(image_directory)
        self.current_index = 0
        self.scale = 1.0
        self.current_image = None
        self.original_image = None
        self._load_current_image()

    def _load_image_files(self, directory: str) -> List[str]:
        files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        files.sort()
        return files

    def _load_current_image(self):
        if not self.image_files:
            return
        img_path = os.path.join("radio", self.image_files[self.current_index])
        img = cv2.imread(img_path)
        if img is not None:
            self.original_image = cv2.resize(img, self.config.TARGET_SIZE)
            self.current_image = self.original_image.copy()

    def zoom_image(self, scale_change: float):
        """Zooms the image in/out based on pinch gesture."""
        self.scale = max(self.config.MIN_SCALE,
                         min(self.config.MAX_SCALE, self.scale + scale_change))
        if self.original_image is not None:
            height, width = self.original_image.shape[:2]
            new_width = int(width * self.scale)
            new_height = int(height * self.scale)
            self.current_image = cv2.resize(self.original_image, (new_width, new_height))

    def draw_info(self):
        if self.current_image is not None:
            cv2.putText(self.current_image,
                        f'Image {self.current_index + 1}/{len(self.image_files)}',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(self.current_image,
                        f'Zoom: {self.scale:.2f}x',
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


def calculate_distance(p1, p2) -> float:
    return math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)


def is_palm_open(landmarks) -> bool:
    finger_tips = [landmarks.landmark[tip] for tip in [8, 12, 16, 20]]
    distances = [calculate_distance(finger_tips[i], finger_tips[i + 1]) for i in range(len(finger_tips) - 1)]
    return all(d > 0.04 for d in distances)


def is_strict_thumb_up(hand_landmarks) -> bool:
    """
    Detects if the thumb is strictly pointing up.
    - Thumb tip must be significantly above the base.
    - Other fingers should be folded.
    """
    thumb_tip = hand_landmarks.landmark[4]
    thumb_mcp = hand_landmarks.landmark[2]

    # Ensure thumb tip is significantly above MCP joint
    is_thumb_up = thumb_tip.y < thumb_mcp.y - 0.05

    # Ensure all other fingers are curled
    other_fingers_folded = all(
        hand_landmarks.landmark[tip].y > hand_landmarks.landmark[tip - 2].y
        for tip in [8, 12, 16, 20]
    )

    return is_thumb_up and other_fingers_folded


# Initialize
config = GestureConfig()
detector = HandGestureDetector()
controller = ImageController("radio", config)
cap = cv2.VideoCapture(0)
thumbs_up_img = cv2.imread("thumbs_up.png")

previous_pinch_distance = None

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot access camera")
        break

    frame = cv2.flip(frame, 1)
    results = detector.process_frame(frame)

    if results.multi_hand_landmarks:
        right_hand = None

        for hand_landmarks in results.multi_hand_landmarks:
            detector.draw_landmarks(frame, hand_landmarks)

            # Strict thumbs-up detection
            if is_strict_thumb_up(hand_landmarks):
                if thumbs_up_img is not None:
                    frame[50:250, 50:250] = cv2.resize(thumbs_up_img, (200, 200))

                # Save the current displayed image only if thumb is strictly up
                if controller.current_image is not None:
                    cv2.imwrite("photo_capture.jpg", controller.current_image)

                cv2.putText(frame, "THUMBS UP DETECTED - PHOTO SAVED!", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Process right hand for zooming
            if is_palm_open(hand_landmarks):
                right_hand = hand_landmarks

        # If a right hand is detected, process pinch zoom
        if right_hand:
            thumb = right_hand.landmark[4]
            index = right_hand.landmark[8]
            pinch_distance = calculate_distance(thumb, index) * 1000

            if previous_pinch_distance is not None:
                scale_change = (pinch_distance - previous_pinch_distance) / 200
                controller.zoom_image(scale_change)

            previous_pinch_distance = pinch_distance

            cv2.putText(frame, "ZOOM MODE", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    controller.draw_info()
    cv2.imshow('Photo', controller.current_image)
    cv2.imshow('Controle gestuel', frame)

    cv2.imshow("Camera Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
     break

cap.release()
cv2.destroyAllWindows()
