# GestureSense – Simple Reflex Vision Agent

## Description
GestureSense is a real-time computer vision application built with OpenCV and MediaPipe. It detects hands, faces, and mouth expressions from a webcam stream, then responds instantly using a strict **Simple Reflex Agent** architecture.

## Architecture (Simple Reflex Agent)
The system follows a pure reflex model:
- No memory of past states
- No learning or model training
- No prediction
- Decision is made only from current frame input

Decision method:
- `SimpleReflexAgent.decide(finger_count, face_count, mouth_status)`

Rules:
- If `face_count == 0` → `No User Detected`
- If `finger_count == 0` → `Stop Interaction`
- If `finger_count == 5` → `Greeting User`
- If `mouth_status == "You are smiling"` → `User Happy`
- If `mouth_status == "You are laughing"` → `High Engagement`
- Otherwise → `Idle`

## Features
- Real-time hand detection and finger counting
- Real-time face detection
- Mouth state analysis (`Mouth open`, `You are smiling`, `You are laughing`)
- Optional teeth visibility estimation
- Professional dark semi-transparent UI overlay
- Color-coded system status and agent actions
- Action logging with timestamp (`actions.log`)
- FPS counter for performance feedback

## Project Structure
- `main.py` – application runtime loop and orchestration
- `reflex_agent.py` – strict rule-based decision logic
- `ui.py` – professional overlay rendering and layout
- `logger.py` – timestamped action logging
- `requirements.txt` – dependencies
- `src/detector.py` – hand, face, and face-mesh detectors
- `src/utils.py` – hand analysis and finger counting helpers

## How to Run
1. Create and activate a Python 3.10+ environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run:
   ```bash
   python main.py
   ```

## Controls
- `S` or `ESC` → Exit
- `R` → Reset state (manual reset event logged)
- `L` → Clear logs
