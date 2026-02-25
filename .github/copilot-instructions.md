# Copilot Instructions for VirtualMouse

## Project Overview
This project implements a hand-gesture-based virtual mouse and authentication system using computer vision. It is organized into modular components for detection, gesture interpretation, mouse control, authentication, and more. The main entry point is `src/main.py`.

## Architecture & Key Components
- **Detection**: `src/detection/hand_detector.py` uses MediaPipe for hand landmark detection.
- **Gestures**: `src/gestures/` interprets hand/finger states and mouse gestures (e.g., left click, movement).
- **Control**: `src/control/virtual_mouse.py` translates gestures into OS mouse actions using `pyautogui`.
- **Authentication**: `src/authentication/` handles user enrollment, liveness, session management, and verification.
- **ML**: `src/ml/` contains training scripts and classifiers for authentication and gesture recognition.
- **Preprocessing**: `src/preprocessing/` provides normalization, ROI extraction, and smoothing utilities.
- **System/UI**: `src/system/` and `src/ui/` manage access control, overlays, and feedback.
- **Utils**: `src/utils/` offers logging and FPS counting.

## Data Flow
- Webcam frames are processed in `main.py`.
- Hand landmarks are detected, then passed to gesture and authentication modules.
- Gestures trigger mouse actions or authentication events.

## Developer Workflows
- **Run Main App**: `python -m src.main` (ensure webcam is connected)
- **Dependencies**: Install via `pip install -r requirements.txt` (add `opencv-python`, `pyautogui`, `mediapipe` if missing)
- **Model Files**: Place `.task` models in `data/models/` or as referenced in code.
- **Testing**: No tests present; add to `tests/` using project structure conventions.

## Conventions & Patterns
- All modules use relative imports from `src/`.
- Hand landmarks are accessed as lists of objects with `.x`, `.y` attributes.
- Gesture logic expects finger state arrays (e.g., `[thumb, index, middle, ...]`).
- Mouse movement is mapped from hand coordinates to screen coordinates using `VirtualMouse.move()`.
- Authentication and ML modules are decoupled from main gesture logic.

## Integration Points
- **External**: `cv2`, `pyautogui`, `mediapipe` (install as needed)
- **Models**: Hand landmark models (`.task` files) required for detection
- **Data**: Enrollment and logs stored in `data/`

## Example: Mouse Movement
```python
if fingers[1] == 1 and fingers[2] == 0:
    index_tip = hand[8]
    cx, cy = int(index_tip.x * w), int(index_tip.y * h)
    mouse.move(cx, cy, w, h)
```

## Key Files & Directories
- `src/main.py`: App entry point
- `src/detection/hand_detector.py`: Hand detection
- `src/control/virtual_mouse.py`: Mouse control
- `src/gestures/`: Gesture logic
- `src/authentication/`: User auth
- `data/`: Models, logs, enrollment

---
**For AI agents:**
- Use modular structure for new features
- Follow existing data flow and import patterns
- Reference this file for project-specific conventions
