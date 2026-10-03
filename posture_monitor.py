import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pygame
import time
import math
import sys
import os

# ==================== CONFIG ====================
CAMERA_INDEX = 0
STANDING_MIN, STANDING_MAX = 85, 95
SITTING_MIN, SITTING_MAX = 75, 105
GOOD_POSTURE_SECONDS = 5.0

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

ALARM_MP3 = resource_path("alarm.mp3")
MODEL_PATH = resource_path("pose_landmarker.task")

# ==================== INIT ====================
pygame.mixer.init()

# Create Pose Landmarker
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
landmarker = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

# State
mode = "posture"
is_alarming = False
good_posture_start = None
timer_seconds = 0
timer_start_time = None
alarm_playing = False

def play_alarm():
    global alarm_playing
    if not alarm_playing and os.path.exists(ALARM_MP3):
        try:
            pygame.mixer.music.load(ALARM_MP3)
            pygame.mixer.music.play(-1)
            alarm_playing = True
            print(">>> ALARM STARTED")
        except Exception as e:
            print("Could not play MP3:", e)

def stop_alarm():
    global alarm_playing, is_alarming, good_posture_start
    if alarm_playing:
        pygame.mixer.music.stop()
        alarm_playing = False
        print(">>> ALARM STOPPED")
    is_alarming = False
    good_posture_start = None

def get_landmark_point(landmarks, index, w, h):
    lm = landmarks[index]
    return int(lm.x * w), int(lm.y * h)

def calculate_spine_angle(landmarks, w, h):
    # MediaPipe Pose landmarks:
    # 11 = left shoulder, 12 = right shoulder
    # 23 = left hip, 24 = right hip
    ls = get_landmark_point(landmarks, 11, w, h)
    rs = get_landmark_point(landmarks, 12, w, h)
    lh = get_landmark_point(landmarks, 23, w, h)
    rh = get_landmark_point(landmarks, 24, w, h)

    mid_shoulder = ((ls[0] + rs[0]) // 2, (ls[1] + rs[1]) // 2)
    mid_hip = ((lh[0] + rh[0]) // 2, (lh[1] + rh[1]) // 2)

    dx = mid_shoulder[0] - mid_hip[0]
    dy = mid_shoulder[1] - mid_hip[1]

    angle = math.degrees(math.atan2(dy, dx))
    if angle < 0:
        angle += 180
    return angle, mid_hip, mid_shoulder

def draw_labeled_line(frame, p1, p2, color, label, thickness=3):
    cv2.line(frame, p1, p2, color, thickness)
    mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
    cv2.putText(frame, label, (mid[0] + 8, mid[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

def is_posture_ok(angle):
    return (SITTING_MIN <= angle <= SITTING_MAX) or (STANDING_MIN <= angle <= STANDING_MAX)

print("""
Controls:
  q     - Quit
  s     - Stop alarm
  m     - Switch mode (Posture <-> Alarm Clock)
  t     - Set timer
  c     - Cycle camera
""")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    detection_result = landmarker.detect(mp_image)

    spine_angle = None
    posture_status = "No person"
    color_status = (0, 0, 255)

    if detection_result.pose_landmarks:
        landmarks = detection_result.pose_landmarks[0]

        # Head
        nose = get_landmark_point(landmarks, 0, w, h)
        ls = get_landmark_point(landmarks, 11, w, h)
        rs = get_landmark_point(landmarks, 12, w, h)
        mid_s = ((ls[0] + rs[0]) // 2, (ls[1] + rs[1]) // 2)
        draw_labeled_line(frame, nose, mid_s, (255, 0, 255), "HEAD", 2)

        # Spine
        spine_angle, mid_hip, mid_shoulder = calculate_spine_angle(landmarks, w, h)
        spine_color = (0, 255, 0) if is_posture_ok(spine_angle) else (0, 0, 255)
        draw_labeled_line(frame, mid_hip, mid_shoulder, spine_color, "SPINE", 4)

        # Arms
        for side, color, sh_idx, el_idx, wr_idx in [
            ("LEFT", (255, 165, 0), 11, 13, 15),
            ("RIGHT", (0, 165, 255), 12, 14, 16)
        ]:
            sh = get_landmark_point(landmarks, sh_idx, w, h)
            el = get_landmark_point(landmarks, el_idx, w, h)
            wr = get_landmark_point(landmarks, wr_idx, w, h)
            draw_labeled_line(frame, sh, el, color, f"{side[0]} ARM", 2)
            draw_labeled_line(frame, el, wr, color, "", 2)

        # Legs
        for side, color, hip_idx, knee_idx, ank_idx in [
            ("LEFT", (0, 255, 255), 23, 25, 27),
            ("RIGHT", (255, 255, 0), 24, 26, 28)
        ]:
            hip = get_landmark_point(landmarks, hip_idx, w, h)
            knee = get_landmark_point(landmarks, knee_idx, w, h)
            ank = get_landmark_point(landmarks, ank_idx, w, h)
            draw_labeled_line(frame, hip, knee, color, f"{side[0]} LEG", 2)
            draw_labeled_line(frame, knee, ank, color, "", 2)

        # Posture logic
        if mode == "posture":
            if is_posture_ok(spine_angle):
                posture_status = f"OK  ({spine_angle:.1f}°)"
                color_status = (0, 255, 0)
                if is_alarming:
                    if good_posture_start is None:
                        good_posture_start = time.time()
                    elif time.time() - good_posture_start >= GOOD_POSTURE_SECONDS:
                        stop_alarm()
            else:
                posture_status = f"BAD / FALL RISK  ({spine_angle:.1f}°)"
                color_status = (0, 0, 255)
                good_posture_start = None
                if not is_alarming:
                    is_alarming = True
                    play_alarm()

    # Alarm Clock mode
    if mode == "alarm_clock":
        if timer_start_time is not None:
            remaining = max(0, timer_seconds - (time.time() - timer_start_time))
            if remaining <= 0:
                if not is_alarming:
                    is_alarming = True
                    play_alarm()
                posture_status = "TIMER EXPIRED - ALARM"
                color_status = (0, 0, 255)
            else:
                posture_status = f"Timer: {int(remaining)}s"
                color_status = (255, 255, 0)
        else:
            posture_status = "Timer not set (press 't')"

    # UI
    cv2.rectangle(frame, (0, 0), (w, 90), (30, 30, 30), -1)
    cv2.putText(frame, f"Mode: {mode.upper()}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, posture_status, (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color_status, 2)

    if is_alarming:
        cv2.putText(frame, "ALARM ACTIVE - Press 's' to stop", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(frame, "q:quit  s:stop  m:mode  t:timer  c:camera", (10, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    cv2.imshow("Posture & Fall Monitor + Alarm Clock", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        stop_alarm()
    elif key == ord('m'):
        mode = "alarm_clock" if mode == "posture" else "posture"
        stop_alarm()
        timer_start_time = None
        print(f"Switched to {mode} mode")
    elif key == ord('t') and mode == "alarm_clock":
        try:
            val = input("Enter timer seconds (e.g. 30): ")
            timer_seconds = int(val)
            timer_start_time = time.time()
            stop_alarm()
            print(f"Timer set to {timer_seconds} seconds")
        except:
            print("Invalid input")
    elif key == ord('c'):
        cap.release()
        CAMERA_INDEX = (CAMERA_INDEX + 1) % 4
        cap = cv2.VideoCapture(CAMERA_INDEX)
        print(f"Switched to camera {CAMERA_INDEX}")

# Cleanup
stop_alarm()
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
landmarker.close()
