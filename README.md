# 🎮 Gesture-Controlled Volume System

Control your laptop using your hand seen in the camera.

## 🎬 Demo

Watch the demo video here ()

## ✨ Features

- **Peace Sign (✌️) to Unlock** - Hold for 0.5 seconds to activate
- **Hand Movement to Control Volume** - Move your hand up/down like a volume slider
- **Fist (✊) to Lock** - Hold for 0.5 seconds to deactivate
- **Smart Detection** - It won't accidentally trigger from random hand movements without unlocking
- **Volume Snapping** - Automatically rounds to nearest 5% for smooth control

## 🚀 Installation

1. Make sure you have Python 3.8+ installed
2. Install dependencies:
```bash
pip install opencv-python mediapipe pycaw comtypes
```

📖  Instructions
Run the script ```python main.py``` 
A webcam window will open
Show a peace sign - Hold it for half a second to unlock
Move your hand up/down - Your hand position controls the volume
Hand at top = 100% volume
Hand at bottom = 0% volume
Make a fist - Hold for half a second to lock the system

🛠️ Built With
Python - Programming language
OpenCV - Computer vision library
MediaPipe - Hand tracking by Google
PyCAW - Windows audio control

🔮 Future Improvements
Add brightness control
Add visual volume bar on screen
Add minimize windows gesture
Convert to .exe file for easy installation
Auto-lock after 5 seconds of inactivity
