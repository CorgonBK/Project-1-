# Senior Fall Detection & Posture Monitor

Mobile / Desktop Computer Vision system for real-time fall risk and posture monitoring of seniors using MediaPipe Pose Estimation.

**Author:** Bryant Kha

## Features
- Live camera feed with body landmark detection
- Color-coded and labeled skeleton (Spine, Head, Arms, Legs)
- Real-time spine angle calculation
- Automatic alarm when posture is unsafe (sitting/standing thresholds)
- Alarm auto-stops after 5 seconds of good posture or manual stop
- Alarm Clock mode with timer
- Works on ordinary webcams / mobile cameras

## Requirements
- Python 3.8+
- Webcam

## Installation

```bash
git clone https://github.com/CorgonBK/Project-1-.git
cd Project-1-
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
