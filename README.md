# Senior Fall Detection & Posture Monitor

A Python desktop / webcam application for real-time fall-risk and posture monitoring of seniors using MediaPipe Pose Estimation.  
The system draws labeled body landmarks, computes spine angle, and triggers an audible alarm when posture is unsafe. It also includes a simple Alarm Clock mode.

**Author:** Bryant Kha

---

## Status — read first

- Working application source is included (`posture_monitor.py`).
- Requires a local webcam and the MediaPipe pose landmarker model file.
- The companion GitHub Pages research tutorial lives in the `/docs` folder of this repository.
- No unit-test suite is included. Test by running the program with a webcam and verifying the on-screen skeleton, angle readout, and alarm behavior.

---

## 1. What this project does

| Feature | Description |
|---------|-------------|
| Live camera view | Continuous webcam feed |
| Body landmarks | Head, spine, arms, and legs drawn with colored labeled lines |
| Spine-angle logic | Standing OK ≈ 85–95°, Sitting OK ≈ 75–105° |
| Automatic alarm | Plays `alarm.mp3` when posture leaves the safe range |
| Auto-stop | Alarm stops after 5 continuous seconds of good posture (or manual stop) |
| Alarm Clock mode | Set a timer; same alarm file and stop behavior |
| Settings panel | Top-right overlay shows mode, live angle, thresholds, and alarm status |

---

## 2. Requirements

- **Python 3.11 or 3.12 recommended** (3.14 works with `pygame-ce` and recent MediaPipe)
- Webcam
- Windows, macOS, or Linux

Python packages (see `requirements.txt`):

```
opencv-python
mediapipe
numpy
pygame-ce          # use pygame-ce on Python 3.14; classic pygame lacks 3.14 wheels
```

---

## 3. Setup (Windows)

1. Install **Python 3.12** from https://www.python.org/downloads/  
   During installation check **“Add python.exe to PATH”**.

2. Clone or download this repository, then open PowerShell in the project folder:

   ```powershell
   cd path\to\Project-1-
   py -3.12 -m venv venv
   .\venv\Scripts\activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

   If you are on Python 3.14 and `pygame` fails to install, use:

   ```powershell
   pip install opencv-python mediapipe numpy pygame-ce
   ```

3. Download the MediaPipe pose model and place it in the project root:

   - URL: https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task  
   - Save / rename the file as: **`pose_landmarker.task`**

4. Place an MP3 file named **`alarm.mp3`** in the same folder (or the code will skip audio).

5. Run:

   ```powershell
   python posture_monitor.py
   ```

---

## 4. Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Stop alarm (manual) |
| `m` | Switch mode (Posture Monitor ↔ Alarm Clock) |
| `t` | Set timer (Alarm Clock mode only) |
| `c` | Cycle camera index |

---

## 5. How the posture logic works

1. MediaPipe Pose Landmarker returns body landmarks each frame.
2. Mid-shoulder and mid-hip points are computed.
3. Spine angle (degrees from horizontal) is calculated; ~90° = upright.
4. If the angle stays outside the sitting/standing safe ranges, the alarm starts.
5. The alarm stops automatically after 5 continuous seconds of good posture, or when you press `s`.

Privacy note: only skeleton lines are drawn; no raw video is uploaded.

---

## 6. Project files

| File / folder | Purpose |
|---------------|---------|
| `posture_monitor.py` | Main application |
| `requirements.txt` | Python dependencies |
| `pose_landmarker.task` | MediaPipe model (you must download) |
| `alarm.mp3` | Alarm sound |
| `docs/` | Full GitHub Pages research tutorial (HTML/CSS) |
| `README.md` | This file |

---

## 7. GitHub Pages tutorial

The research tutorial (problem, pipeline, algorithms, systems, challenges, quiz, bibliography) is in the **`docs/`** folder.

To publish it:

1. Repository → **Settings → Pages**
2. Source: Deploy from a branch
3. Branch: `main`, folder: `/docs`
4. Save

Site URL will be similar to:  
`https://YOUR_USERNAME.github.io/Project-1-/`

---

## 8. Demo video

YouTube: https://youtu.be/IB7hSkayhRE

---

## 9. Notes & limitations

- Best results when the full body is visible and the person faces the camera.
- Performance depends on lighting and camera quality.
- The simple angle + stillness rule is intentionally easy to understand; research systems often add temporal models (LSTM/Transformer) for fewer false alarms.
- This is an educational prototype, not a medical device.

---

## References

- MediaPipe Pose Landmarker – Google AI Edge  
- University of Alberta / Spxtrm AI fall-detection work (privacy-preserving vision)  
- Course materials and related computer-vision fall-detection literature (see `docs/bibliography.html`)
