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
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

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
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)

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

def draw_settings_box(frame, mode, spine_angle, is_alarming, timer_remaining=None):
    """Draw a nice settings panel on the top-right corner"""
    h, w = frame.shape[:2]
    box_w, box_h = 320, 210
    x1 = w - box_w - 20
    y1 = 20
    x2 = w - 20
    y2 = y1 + box_h

    # Semi-transparent dark background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Border
    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 80), 2)

    # Title
    cv2.putText(frame, "SETTINGS", (x1 + 15, y1 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Mode
    cv2.putText(frame, f"Mode: {mode.upper()}", (x1 + 15, y1 + 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 255), 1)

    # Spine Angle
    if spine_angle is not None:
        angle_color = (0, 255, 0) if is_posture_ok(spine_angle) else (0, 0, 255)
        cv2.putText(frame, f"Spine Angle: {spine_angle:.1f} deg", (x1 + 15, y1 + 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, angle_color, 1)
    else:
        cv2.putText(frame, "Spine Angle: --", (x1 + 15, y1 + 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)

    # Thresholds
    cv2.putText(frame, f"Standing: {STANDING_MIN}-{STANDING_MAX}", (x1 + 15, y1 + 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(frame, f"Sitting:  {SITTING_MIN}-{SITTING_MAX}", (x1 + 15, y1 + 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # Alarm status
    if is_alarming:
        cv2.putText(frame, "ALARM: ACTIVE", (x1 + 15, y1 + 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "ALARM: Off", (x1 + 15, y1 + 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 1)

    # Timer (if in alarm clock mode)
    if mode == "alarm_clock" and timer_remaining is not None:
        cv2.putText(frame, f"Timer: {timer_remaining}s", (x1 + 15, y1 + 205),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

print("""
Controls:
  q     - Quit
  s     - Stop alarm
  m     - Switch mode
  t     - Set timer
  c     - Cycle camera
""")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize to make sure it's big
    frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    detection_result = landmarker.detect(mp_image)

    spine_angle = None
    posture_status = "No person detected"
    color_status = (0, 0, 255)
    timer_remaining = None

    if detection_result.pose_landmarks:
        landmarks = detection_result.pose_landmarks[0]

        # Draw body parts
        nose = get_landmark_point(landmarks, 0, w, h)
        ls = get_landmark_point(landmarks, 11, w, h)
        rs = get_landmark_point(landmarks, 12, w, h)
        mid_s = ((ls[0] + rs[0]) // 2, (ls[1] + rs[1]) // 2)
        draw_labeled_line(frame, nose, mid_s, (255, 0, 255), "HEAD", 2)

        spine_angle, mid_hip, mid_shoulder = calculate_spine_angle(landmarks, w, h)
        spine_color = (0, 255, 0) if is_posture_ok(spine_angle) else (0, 0, 255)
        draw_labeled_line(frame, mid_hip, mid_shoulder, spine_color, "SPINE", 4)

        for side, color, sh_idx, el_idx, wr_idx in [
            ("LEFT", (255, 165, 0), 11, 13, 15),
            ("RIGHT", (0, 165, 255), 12, 14, 16)
        ]:
            sh = get_landmark_point(landmarks, sh_idx, w, h)
            el = get_landmark_point(landmarks, el_idx, w, h)
            wr = get_landmark_point(landmarks, wr_idx, w, h)
            draw_labeled_line(frame, sh, el, color, f"{side[0]} ARM", 2)
            draw_labeled_line(frame, el, wr, color, "", 2)

        for side, color, hip_idx, knee_idx, ank_idx in [
            ("LEFT", (0, 255, 255), 23, 25, 27),
            ("RIGHT", (255, 255, 0), 24, 26, 28)
        ]:
            hip = get_landmark_point(landmarks, hip_idx, w, h)
            knee = get_landmark_point(landmarks, knee_idx, w, h)
            ank = get_landmark_point(landmarks, ank_idx, w, h)
            draw_labeled_line(frame, hip, knee, color, f"{side[0]} LEG", 2)
            draw_labeled_line(frame, knee, ank, color, "", 2)

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
            timer_remaining = int(remaining)
            if remaining <= 0:
                if not is_alarming:
                    is_alarming = True
                    play_alarm()
                posture_status = "TIMER EXPIRED - ALARM"
                color_status = (0, 0, 255)
            else:
                posture_status = f"Timer: {timer_remaining}s"
                color_status = (255, 255, 0)
        else:
            posture_status = "Timer not set (press 't')"

    # ===== UI =====
    # Top left status bar
    cv2.rectangle(frame, (0, 0), (420, 90), (25, 25, 25), -1)
    cv2.putText(frame, f"Mode: {mode.upper()}", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, posture_status, (15, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_status, 2)

    # Settings box (top right)
    draw_settings_box(frame, mode, spine_angle, is_alarming, timer_remaining)

    # Bottom instructions
    cv2.rectangle(frame, (0, h - 50), (w, h), (25, 25, 25), -1)
    cv2.putText(frame, "q: Quit   |   s: Stop Alarm   |   m: Switch Mode   |   t: Set Timer   |   c: Change Camera",
                (20, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    if is_alarming:
        cv2.putText(frame, "ALARM ACTIVE!", (w//2 - 120, h - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

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
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)
        print(f"Switched to camera {CAMERA_INDEX}")

# Cleanup
stop_alarm()
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
landmarker.close()
