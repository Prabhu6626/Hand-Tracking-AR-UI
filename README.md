# Hand Tracking AR UI - Advanced Edition

An advanced Augmented Reality Hand Tracking interface optimized for MacBook M1. This application uses computer vision to detect hand gestures and overlays futuristic holographic UI elements in real-time.

## Features

### Advanced AR Elements
- **Multi-layered Radial Interfaces**: Rotating concentric circles with animated ticks
- **Energy Core Visualization**: Pulsing hexagonal core with wave patterns
- **Holographic Grid Overlay**: 3D-style grid projection
- **Data Streams**: Animated connections to fingertips
- **Particle Systems**: Dynamic particle effects for interactions
- **HUD Panels**: Corner brackets and info bars
- **Scan Lines**: Sci-fi scanning effects

### Gesture Recognition
- **Open Hand**: Full AR interface with data visualization
- **Pinch**: Precision control interface with progress indicators
- **Fist**: Power mode with pulsing effects
- **Peace Sign**: Special particle burst effects
- **OK Sign**: Menu toggle (expandable)
- **Swipe Detection**: Left/right swipe recognition

### Technical Features
- Optimized for Apple M1 chip
- Real-time FPS monitoring
- Smooth 60 FPS performance
- Two-hand tracking support
- Gesture confidence indicators
- Dynamic energy level system
- Particle physics simulation

## Installation

### Prerequisites
- Python 3.8 or higher
- MacBook M1 (optimized for)
- Webcam access

### Setup

1. Install dependencies:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

2. Run the application:
\`\`\`bash
python main.py
\`\`\`

## Usage

### Controls
- **ESC**: Exit application
- **S**: Switch to scan mode
- **C**: Switch to control mode

### Gestures Guide

#### Open Hand (Full AR Interface)
- Spread all fingers wide
- Shows complete holographic UI with:
  - Multiple rotating rings
  - Energy core visualization
  - Data streams to fingertips
  - HUD panels and readouts
  - Angle measurements

#### Pinch (Precision Control)
- Touch thumb and index finger together
- Shows:
  - Pinch force indicator (0-100%)
  - Circular progress bar
  - Radial segments
  - Energy arcs

#### Fist (Power Mode)
- Close all fingers into fist
- Displays:
  - Pulsing red power rings
  - Hexagonal core
  - Energy indicators

#### Peace Sign
- Extend index and middle finger only
- Creates particle burst effects
- Green holographic elements

#### OK Sign
- Touch thumb and index in circle, extend other fingers
- Purple interface elements
- Menu toggle function

## Performance Tips

### For Best Results
1. Ensure good lighting conditions
2. Position hand 1-2 feet from camera
3. Use solid-colored background
4. Keep hand movements smooth

### M1 Optimization
- The app uses `model_complexity=1` for balanced performance
- Targets 60 FPS on M1 MacBooks
- Multi-threaded MediaPipe processing
- Optimized particle system updates

## Customization

### Color Schemes
Edit the color constants at the top of `main.py`:
\`\`\`python
CYAN = (255, 255, 0)
NEON_BLUE = (255, 180, 0)
NEON_GREEN = (0, 255, 150)
# ... add your own colors
\`\`\`

### Gesture Sensitivity
Adjust detection thresholds in `detect_advanced_gestures()`:
\`\`\`python
"is_open": avg_dist > 70,  # Increase for stricter detection
"is_fist": avg_dist < 40,  # Decrease for easier detection
"is_pinch": pinch_dist < 35,
\`\`\`

### Visual Effects
Modify glow intensity and particle counts:
\`\`\`python
draw_glow_circle(img, center, radius, color, thickness, glow=15, intensity=1.0)
create_particle_burst(center, color, count=20)
\`\`\`

## Architecture

### Main Components
- **Gesture Detection**: MediaPipe Hands with custom gesture recognition
- **Particle System**: Physics-based particle effects
- **AR Rendering**: Layered drawing system with alpha blending
- **Animation Engine**: Frame-based rotation and pulsing effects

### Performance Optimizations
- Efficient particle lifecycle management
- Optimized OpenCV drawing operations
- Smart alpha blending for glow effects
- Deque-based gesture history

## Troubleshooting

### Low FPS
- Reduce particle count in `create_particle_burst()`
- Decrease glow radius in `draw_glow_circle()`
- Lower camera resolution in capture settings

### Gesture Not Detected
- Improve lighting conditions
- Adjust detection confidence thresholds
- Ensure hand is fully visible in frame

### Camera Issues
- Grant camera permissions in System Preferences
- Check if another app is using the camera
- Try different camera index: `cv2.VideoCapture(1)`

## Future Enhancements
- Multi-hand interaction modes
- Voice command integration
- Gesture-based menu system
- Recording and replay functionality
- Custom gesture training
- VR headset integration

## Credits
Built with:
- OpenCV for image processing
- MediaPipe for hand tracking
- NumPy for mathematical operations
- Optimized for Apple Silicon (M1/M2/M3)

## License
MIT License - Feel free to modify and extend!
\`\`\`

```plaintext file="requirements.txt"
opencv-python>=4.8.0
mediapipe>=0.10.0
numpy>=1.24.0
