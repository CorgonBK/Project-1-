import cv2
import mediapipe as mp
import numpy as np
import pygame
import time
import math
import os

# ==================== CONFIG ====================
ALARM_MP3 = "alarm.mp3"          # Change this path if needed
CAMERA_INDEX = 0                 # Default camera (change with 'c' key)
STANDING_MIN, STANDING_MAX = 85, 95
SITTING_MIN, SITTING_MAX = 75, 105
GOOD_POSTURE_SECONDS = 5.0       # Must stay OK for this long to auto-stop alarm

# ==================== INIT ====================
pygame.mixer.init()
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
    model_complexity=1
)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

# State
mode = "posture"                 # "posture" or "alarm_clock"
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
            pygame.mixer.music.play(-1)  # loop forever
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

def get_landmark_point(landmarks, landmark_id, w, h):
    lm = landmarks[landmark_id]
    return int(lm.x * w), int(lm.y * h)

def calculate_spine_angle(landmarks, w, h):
    """Returns angle from horizontal (90° = perfectly vertical upright)."""
    ls = get_landmark_point(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER, w, h)
    rs = get_landmark_point(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER, w, h)
    lh = get_landmark_point(landmarks, mp_pose.PoseLandmark.LEFT_HIP, w, h)
    rh = get_landmark_point(landmarks, mp_pose.PoseLandmark.RIGHT_HIP, w, h)

    mid_shoulder = ((ls[0] + rs[0]) // 2, (ls[1] + rs[1]) // 2)
    mid_hip = ((lh[0] + rh[0]) // 2, (lh[1] + rh[1]) // 2)

    dx = mid_shoulder[0] - mid_hip[0]
    dy = mid_shoulder[1] - mid_hip[1]   # y increases downward in image

    # atan2(dy, dx) → normalize so 90° means upright (pointing up)
    angle = math.degrees(math.atan2(dy, dx))
    if angle < 0:
        angle += 180
    # Because person faces camera, upright spine is near 90°
    return angle, mid_hip, mid_shoulder

def draw_labeled_line(frame, p1, p2, color, label, thickness=3):
    cv2.line(frame, p1, p2, color, thickness)
    mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
    cv2.putText(frame, label, (mid[0] + 8, mid[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

def is_posture_ok(angle):
    # Accept either sitting or standing range
    return (SITTING_MIN <= angle <= SITTING_MAX) or (STANDING_MIN <= angle <= STANDING_MAX)

# ==================== MAIN LOOP ====================
print("""
Controls:
  q     - Quit
  s     - Stop alarm (manual)
  m     - Switch mode (Posture Monitor <-> Alarm Clock)
  t     - Set timer (Alarm Clock mode only)
  c     - Cycle camera
""")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    spine_angle = None
    posture_status = "No person"
    color_status = (0, 0, 255)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        # ----- Draw body parts with colored labeled lines -----
        # Head (nose → mid shoulder)
        nose = get_landmark_point(lm, mp_pose.PoseLandmark.NOSE, w, h)
        ls = get_landmark_point(lm, mp_pose.PoseLandmark.LEFT_SHOULDER, w, h)
        rs = get_landmark_point(lm, mp_pose.PoseLandmark.RIGHT_SHOULDER, w, h)
        mid_s = ((ls[0] + rs[0]) // 2, (ls[1] + rs[1]) // 2)
        draw_labeled_line(frame, nose, mid_s, (255, 0, 255), "HEAD", 2)

        # Spine
        spine_angle, mid_hip, mid_shoulder = calculate_spine_angle(lm, w, h)
        spine_color = (0, 255, 0) if is_posture_ok(spine_angle) else (0, 0, 255)
        draw_labeled_line(frame, mid_hip, mid_shoulder, spine_color, "SPINE", 4)

        # Arms
        for side, color in [("LEFT", (255, 165, 0)), ("RIGHT", (0, 165, 255))]:
            sh = get_landmark_point(lm, getattr(mp_pose.PoseLandmark, f"{side}_SHOULDER"), w, h)
            el = get_landmark_point(lm, getattr(mp_pose.PoseLandmark, f"{side}_ELBOW"), w, h)
            wr = get_landmark_point(lm, getattr(mp_pose.PoseLandmark, f"{side}_WRIST"), w, h)
            draw_labeled_line(frame, sh, el, color, f"{side[0]} ARM", 2)
            draw_labeled_line(frame, el, wr, color, "", 2)

        # Legs
        for side, color in [("LEFT", (0, 255, 255)), ("RIGHT", (255, 255, 0))]:
            hip = get_landmark_point(lm, getattr(mp_pose.PoseLandmark, f"{side}_HIP"), w, h)
            knee = get_landmark_point(lm, getattr(mp_pose.PoseLandmark, f"{side}_KNEE"), w, h)
            ank = get_landmark_point(lm, getattr(mp_pose.PoseLandmark, f"{side}_ANKLE"), w, h)
            draw_labeled_line(frame, hip, knee, color, f"{side[0]} LEG", 2)
            draw_labeled_line(frame, knee, ank, color, "", 2)

        # ----- Posture logic -----
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

    # ----- Alarm Clock mode -----
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

    # ----- On-screen UI -----
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

    # ----- Keyboard -----
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
        # Cycle cameras
        cap.release()
        CAMERA_INDEX = (CAMERA_INDEX + 1) % 4
        cap = cv2.VideoCapture(CAMERA_INDEX)
        print(f"Switched to camera {CAMERA_INDEX}")

# Cleanup
stop_alarm()
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
