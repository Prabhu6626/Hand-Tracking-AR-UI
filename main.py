import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Support two hands
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    model_complexity=1  # Optimized for M1
)

CYAN = (255, 255, 0)
NEON_BLUE = (255, 180, 0)
NEON_GREEN = (0, 255, 150)
ORANGE = (0, 165, 255)
PURPLE = (255, 100, 200)
WHITE = (255, 255, 255)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
PINK = (180, 0, 255)

frame_count = 0
gesture_history = deque(maxlen=10)
particle_systems = []
active_menu = None
energy_level = 100
interaction_mode = "scan"  # scan, control, menu


class Particle:
    """Particle system for AR effects"""
    def __init__(self, pos, velocity, color, lifetime=30):
        self.pos = np.array(pos, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.color = color
        self.lifetime = lifetime
        self.age = 0
        
    def update(self):
        self.pos += self.velocity
        self.velocity *= 0.95  # Damping
        self.age += 1
        return self.age < self.lifetime
    
    def draw(self, img):
        alpha = 1 - (self.age / self.lifetime)
        if alpha > 0:
            pos = tuple(self.pos.astype(int))
            size = int(8 * alpha)
            cv2.circle(img, pos, size, self.color, -1)


def create_particle_burst(center, color, count=20):
    """Create a burst of particles"""
    particles = []
    for _ in range(count):
        angle = np.random.rand() * 2 * np.pi
        speed = np.random.rand() * 5 + 2
        velocity = [np.cos(angle) * speed, np.sin(angle) * speed]
        particles.append(Particle(center, velocity, color, lifetime=30))
    return particles


def draw_glow_circle(img, center, radius, color, thickness=2, glow=15, intensity=1.0):
    """Enhanced glowing circle with variable intensity"""
    for g in range(glow, 0, -2):
        alpha = (0.05 + 0.15 * (g / glow)) * intensity
        overlay = img.copy()
        cv2.circle(overlay, center, radius + g, color, thickness)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.circle(img, center, radius, color, thickness)


def draw_rotating_radial_ticks(img, center, radius, color, num_ticks=32, length=25, thickness=2, rotation=0):
    """Rotating radial ticks with animation"""
    for i in range(num_ticks):
        angle = np.deg2rad(i * (360 / num_ticks) + rotation)
        tick_length = length if i % 4 == 0 else length * 0.6
        tick_thickness = thickness + 1 if i % 4 == 0 else thickness
        x1 = int(center[0] + (radius - tick_length) * np.cos(angle))
        y1 = int(center[1] + (radius - tick_length) * np.sin(angle))
        x2 = int(center[0] + radius * np.cos(angle))
        y2 = int(center[1] + radius * np.sin(angle))
        cv2.line(img, (x1, y1), (x2, y2), color, tick_thickness)


def draw_hexagon(img, center, size, color, thickness=2, filled=False):
    """Draw hexagonal UI elements"""
    points = []
    for i in range(6):
        angle = np.deg2rad(60 * i - 30)
        x = int(center[0] + size * np.cos(angle))
        y = int(center[1] + size * np.sin(angle))
        points.append([x, y])
    points = np.array(points, np.int32)
    if filled:
        cv2.fillPoly(img, [points], color)
    else:
        cv2.polylines(img, [points], True, color, thickness)


def draw_energy_core(img, center, radius, energy_level, rotation):
    """Advanced energy core with pulsing effect"""
    pulse = 1.0 + 0.15 * np.sin(rotation * 0.1)
    
    # Outer energy ring
    for i in range(3):
        r = int(radius * (1.2 - i * 0.1) * pulse)
        alpha_val = 0.3 - i * 0.1
        overlay = img.copy()
        cv2.circle(overlay, center, r, NEON_BLUE, 2)
        cv2.addWeighted(overlay, alpha_val, img, 1 - alpha_val, 0, img)
    
    # Animated core pattern
    num_points = 60
    for i in range(num_points):
        t = i / num_points * 2 * np.pi
        wave = np.sin(6 * t + rotation * 0.05)
        r = radius * (0.6 + 0.2 * wave) * pulse
        x = int(center[0] + r * np.cos(t))
        y = int(center[1] + r * np.sin(t))
        brightness = int(255 * (0.5 + 0.5 * wave))
        color = (brightness, 255, 255 - brightness)
        cv2.circle(img, (x, y), 2, color, -1)
    
    # Central hexagon
    draw_hexagon(img, center, int(radius * 0.4), CYAN, 2)
    draw_hexagon(img, center, int(radius * 0.3), ORANGE, 1)
    
    # Energy level indicator
    energy_angle = int(360 * (energy_level / 100))
    cv2.ellipse(img, center, (int(radius * 0.5), int(radius * 0.5)), 
                -90, 0, energy_angle, NEON_GREEN, 3)


def draw_holographic_grid(img, center, size, rotation):
    """Draw holographic grid overlay"""
    for i in range(-2, 3):
        for j in range(-2, 3):
            angle = rotation * 0.02
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            x = i * size
            y = j * size
            rx = int(center[0] + x * cos_a - y * sin_a)
            ry = int(center[1] + x * sin_a + y * cos_a)
            
            alpha = 0.15
            overlay = img.copy()
            cv2.circle(overlay, (rx, ry), 3, CYAN, -1)
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_data_stream(img, center, rotation):
    """Draw streaming data visualization"""
    for i in range(5):
        angle = np.deg2rad(rotation + i * 72)
        length = 100 + 20 * np.sin(rotation * 0.05 + i)
        x = int(center[0] + length * np.cos(angle))
        y = int(center[1] + length * np.sin(angle))
        
        # Data line
        cv2.line(img, center, (x, y), CYAN, 1)
        
        # Data node
        draw_hexagon(img, (x, y), 8, NEON_GREEN, 1, filled=True)
        cv2.circle(img, (x, y), 12, NEON_GREEN, 1)


def draw_hud_panel(img, center, rotation):
    """Draw advanced HUD panels"""
    # Corner brackets
    bracket_size = 40
    for i in range(4):
        angle = np.deg2rad(i * 90 + 45)
        x = int(center[0] + 150 * np.cos(angle))
        y = int(center[1] + 150 * np.sin(angle))
        
        # Draw L-shaped bracket
        cv2.line(img, (x - bracket_size, y), (x, y), CYAN, 2)
        cv2.line(img, (x, y - bracket_size), (x, y), CYAN, 2)
    
    # Info bars
    for i in range(3):
        y_offset = center[1] + 80 + i * 25
        bar_length = int(120 * (0.7 + 0.3 * np.sin(rotation * 0.05 + i)))
        cv2.rectangle(img, (center[0] - 60, y_offset - 5),
                     (center[0] - 60 + bar_length, y_offset + 5),
                     [NEON_BLUE, NEON_GREEN, ORANGE][i], -1)
        cv2.rectangle(img, (center[0] - 60, y_offset - 5),
                     (center[0] + 60, y_offset + 5),
                     [NEON_BLUE, NEON_GREEN, ORANGE][i], 1)


def draw_scan_lines(img, center, rotation, height=300):
    """Draw scanning effect"""
    scan_y = int(center[1] - height/2 + (rotation * 3) % height)
    
    for i in range(-2, 3):
        y = scan_y + i * 2
        alpha = 0.3 - abs(i) * 0.1
        if 0 < y < img.shape[0]:
            overlay = img.copy()
            cv2.line(overlay, (center[0] - 100, y), (center[0] + 100, y), NEON_GREEN, 1)
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_gesture_indicator(img, pos, gesture_name, confidence):
    """Draw gesture recognition indicator"""
    color = NEON_GREEN if confidence > 0.8 else ORANGE
    
    # Background
    overlay = img.copy()
    cv2.rectangle(overlay, (pos[0] - 80, pos[1] - 30),
                 (pos[0] + 80, pos[1] + 10), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    
    # Border
    cv2.rectangle(img, (pos[0] - 80, pos[1] - 30),
                 (pos[0] + 80, pos[1] + 10), color, 2)
    
    # Text
    cv2.putText(img, gesture_name, (pos[0] - 70, pos[1] - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(img, f"{int(confidence * 100)}%", (pos[0] - 70, pos[1] + 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1)


def detect_advanced_gestures(landmarks):
    """Detect multiple gesture types"""
    lm = landmarks
    h, w = 480, 640
    points = [(int(l.x * w), int(l.y * h)) for l in lm]
    
    palm = points[9]
    tips = [points[i] for i in [4, 8, 12, 16, 20]]
    
    # Calculate distances
    finger_dists = [np.linalg.norm(np.array(tip) - np.array(palm)) for tip in tips]
    avg_dist = np.mean(finger_dists)
    
    # Pinch detection
    pinch_dist = np.linalg.norm(np.array(points[4]) - np.array(points[8]))
    
    # Peace sign detection
    peace_fingers = [finger_dists[1] > 60, finger_dists[2] > 60,
                    finger_dists[3] < 50, finger_dists[4] < 50]
    
    # OK sign detection
    ok_gesture = pinch_dist < 30 and all([d > 50 for d in finger_dists[2:]])
    
    # Swipe detection (simplified)
    swipe_dir = None
    if avg_dist > 80:
        index_tip = points[8]
        if index_tip[0] < palm[0] - 100:
            swipe_dir = "LEFT"
        elif index_tip[0] > palm[0] + 100:
            swipe_dir = "RIGHT"
    
    return {
        "palm": palm,
        "avg_dist": avg_dist,
        "pinch_dist": pinch_dist,
        "is_open": avg_dist > 70,
        "is_fist": avg_dist < 40,
        "is_pinch": pinch_dist < 35,
        "is_peace": all(peace_fingers),
        "is_ok": ok_gesture,
        "swipe": swipe_dir,
        "points": points,
        "finger_dists": finger_dists
    }


def draw_advanced_open_hand(img, gesture_data, rotation):
    """Advanced open hand visualization"""
    palm = gesture_data["palm"]
    points = gesture_data["points"]
    
    # Multi-layer circles
    draw_glow_circle(img, palm, 140, CYAN, 2, glow=25, intensity=0.8)
    draw_rotating_radial_ticks(img, palm, 140, CYAN, 32, 28, 2, rotation)
    draw_glow_circle(img, palm, 110, NEON_BLUE, 2, glow=20, intensity=0.6)
    draw_rotating_radial_ticks(img, palm, 110, NEON_BLUE, 24, 20, 1, -rotation * 0.7)
    draw_glow_circle(img, palm, 80, ORANGE, 2, glow=15, intensity=0.5)
    
    # Energy core
    draw_energy_core(img, palm, 40, energy_level, rotation)
    
    # Holographic grid
    draw_holographic_grid(img, palm, 25, rotation)
    
    # Data streams to fingertips
    for i, idx in enumerate([4, 8, 12, 16, 20]):
        tip = points[idx]
        
        # Animated line
        segments = 10
        for j in range(segments):
            t = j / segments
            x = int(palm[0] + t * (tip[0] - palm[0]))
            y = int(palm[1] + t * (tip[1] - palm[1]))
            next_x = int(palm[0] + (t + 1/segments) * (tip[0] - palm[0]))
            next_y = int(palm[1] + (t + 1/segments) * (tip[1] - palm[1]))
            
            alpha = 0.3 + 0.7 * np.sin(rotation * 0.1 + j * 0.5)
            brightness = int(255 * alpha)
            color = (brightness, 255, 255 - brightness)
            cv2.line(img, (x, y), (next_x, next_y), color, 2)
        
        # Fingertip nodes
        draw_hexagon(img, tip, 12, [CYAN, NEON_GREEN, ORANGE, PURPLE, PINK][i], 2)
        cv2.circle(img, tip, 8, [CYAN, NEON_GREEN, ORANGE, PURPLE, PINK][i], -1)
        cv2.circle(img, tip, 16, [CYAN, NEON_GREEN, ORANGE, PURPLE, PINK][i], 1)
    
    # HUD panels
    draw_hud_panel(img, palm, rotation)
    
    # Angle display
    v1 = np.array(points[4]) - np.array(palm)
    v2 = np.array(points[8]) - np.array(palm)
    try:
        angle = int(np.degrees(np.arccos(
            np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        )))
    except:
        angle = 0
    
    # Digital readout
    cv2.putText(img, f'ANGLE: {angle}', (palm[0] + 50, palm[1] - 50),
               cv2.FONT_HERSHEY_DUPLEX, 0.7, WHITE, 2)
    cv2.putText(img, f'ENERGY: {energy_level}%', (palm[0] + 50, palm[1] - 25),
               cv2.FONT_HERSHEY_DUPLEX, 0.5, NEON_GREEN, 1)


def draw_pinch_interface(img, gesture_data, rotation):
    """Enhanced pinch gesture interface"""
    palm = gesture_data["palm"]
    pinch_dist = gesture_data["pinch_dist"]
    pinch_val = int(100 - min(pinch_dist, 100))
    
    # Central glow
    draw_glow_circle(img, palm, 80, ORANGE, 3, glow=30, intensity=1.2)
    draw_glow_circle(img, palm, 60, PURPLE, 2, glow=20)
    
    # Pinch value visualization
    draw_hexagon(img, palm, 50, ORANGE, 2)
    
    # Radial pinch indicator
    num_segments = 12
    for i in range(num_segments):
        if i < (pinch_val / 100) * num_segments:
            angle = np.deg2rad(i * (360 / num_segments) - 90 + rotation * 0.5)
            length = 70
            x1 = int(palm[0] + 50 * np.cos(angle))
            y1 = int(palm[1] + 50 * np.sin(angle))
            x2 = int(palm[0] + length * np.cos(angle))
            y2 = int(palm[1] + length * np.sin(angle))
            cv2.line(img, (x1, y1), (x2, y2), ORANGE, 3)
    
    # Circular progress
    angle_span = int(360 * (pinch_val / 100))
    cv2.ellipse(img, palm, (90, 90), -90, 0, angle_span, NEON_GREEN, 4)
    
    # Value display
    cv2.putText(img, f'PINCH', (palm[0] - 50, palm[1] - 100),
               cv2.FONT_HERSHEY_DUPLEX, 0.8, ORANGE, 2)
    cv2.putText(img, f'{pinch_val}%', (palm[0] - 35, palm[1] + 10),
               cv2.FONT_HERSHEY_DUPLEX, 1.2, WHITE, 3)
    
    # Energy arcs
    for i in range(6):
        arc_angle = rotation + i * 60
        cv2.ellipse(img, (palm[0] + 100, palm[1]), (40, 40),
                   arc_angle, 0, pinch_val + 20, PURPLE, 2)


def draw_fist_mode(img, gesture_data, rotation):
    """Fist gesture - power mode"""
    palm = gesture_data["palm"]
    
    # Pulsing effect
    pulse = 1.0 + 0.3 * np.sin(rotation * 0.15)
    
    # Power rings
    for i in range(4):
        radius = int((60 + i * 20) * pulse)
        intensity = 0.8 - i * 0.15
        draw_glow_circle(img, palm, radius, RED, 3, glow=20, intensity=intensity)
    
    # Hexagonal power core
    draw_hexagon(img, palm, int(50 * pulse), RED, 3)
    draw_hexagon(img, palm, int(35 * pulse), YELLOW, 2, filled=True)
    
    # Power text
    cv2.putText(img, 'POWER MODE', (palm[0] - 80, palm[1] - 100),
               cv2.FONT_HERSHEY_DUPLEX, 0.8, RED, 2)
    
    # Energy indicator
    for i in range(8):
        angle = np.deg2rad(i * 45 + rotation)
        x = int(palm[0] + 70 * np.cos(angle))
        y = int(palm[1] + 70 * np.sin(angle))
        cv2.circle(img, (x, y), 5, YELLOW, -1)


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60)

# FPS calculation
fps_start_time = time.time()
fps_frame_count = 0
current_fps = 0

print("=" * 60)
print("HAND TRACKING AR UI - ADVANCED MODE")
print("=" * 60)
print("Gestures:")
print("  • OPEN HAND: Full AR interface with data streams")
print("  • PINCH: Pinch control interface")
print("  • FIST: Power mode")
print("  • PEACE SIGN: Special effects")
print("  • OK SIGN: Menu toggle")
print("\nPress ESC to exit")
print("=" * 60)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    frame = cv2.addWeighted(frame, 0.7, np.zeros_like(frame), 0.3, 0)
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    frame_count += 1
    rotation = frame_count % 360
    
    # Update particles
    particle_systems = [p for p in particle_systems if p.update()]
    for particle in particle_systems:
        particle.draw(frame)
    
    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Draw hand skeleton with custom style
            for connection in mp_hands.HAND_CONNECTIONS:
                start = hand_landmarks.landmark[connection[0]]
                end = hand_landmarks.landmark[connection[1]]
                start_point = (int(start.x * w), int(start.y * h))
                end_point = (int(end.x * w), int(end.y * h))
                cv2.line(frame, start_point, end_point, CYAN, 2)
            
            # Detect gestures
            gesture_data = detect_advanced_gestures(hand_landmarks.landmark)
            palm = gesture_data["palm"]
            
            # Draw scan lines
            if interaction_mode == "scan":
                draw_scan_lines(frame, palm, rotation)
            
            # Main gesture rendering
            if gesture_data["is_peace"]:
                # Peace sign - special particle effect
                if frame_count % 5 == 0:
                    particle_systems.extend(create_particle_burst(palm, NEON_GREEN, 15))
                draw_glow_circle(frame, palm, 100, NEON_GREEN, 3, glow=30)
                draw_hexagon(img, palm, 60, NEON_GREEN, 3)
                draw_gesture_indicator(frame, (palm[0], palm[1] - 120), "PEACE", 0.95)
                
            elif gesture_data["is_ok"]:
                # OK sign - toggle menu
                draw_glow_circle(frame, palm, 80, PURPLE, 3, glow=25)
                draw_hexagon(frame, palm, 50, PURPLE, 2)
                draw_gesture_indicator(frame, (palm[0], palm[1] - 120), "OK SIGN", 0.92)
                
            elif gesture_data["is_open"]:
                # Open hand - full AR interface
                draw_advanced_open_hand(frame, gesture_data, rotation)
                draw_gesture_indicator(frame, (palm[0], palm[1] + 170), "SCAN MODE", 0.98)
                
            elif gesture_data["is_pinch"]:
                # Pinch gesture
                draw_pinch_interface(frame, gesture_data, rotation)
                draw_gesture_indicator(frame, (palm[0], palm[1] + 130), "PINCH CONTROL", 0.94)
                
            elif gesture_data["is_fist"]:
                # Fist gesture
                draw_fist_mode(frame, gesture_data, rotation)
                draw_gesture_indicator(frame, (palm[0], palm[1] + 110), "POWER MODE", 0.90)
            
            # Swipe detection feedback
            if gesture_data["swipe"]:
                cv2.putText(frame, f'SWIPE {gesture_data["swipe"]}',
                           (palm[0] - 70, palm[1] + 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, YELLOW, 2)
    
    fps_frame_count += 1
    if time.time() - fps_start_time >= 1.0:
        current_fps = fps_frame_count
        fps_frame_count = 0
        fps_start_time = time.time()
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (300, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.rectangle(frame, (10, 10), (300, 100), CYAN, 2)
    
    cv2.putText(frame, f'FPS: {current_fps}', (20, 35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, NEON_GREEN, 2)
    cv2.putText(frame, f'MODE: {interaction_mode.upper()}', (20, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 1)
    cv2.putText(frame, f'FRAME: {frame_count}', (20, 85),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
    
    # Energy level animation
    energy_level = max(0, min(100, energy_level + np.random.randint(-2, 3)))
    
    cv2.imshow('Hand Tracking AR UI - Advanced', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('s'):
        interaction_mode = "scan"
    elif key == ord('c'):
        interaction_mode = "control"

cap.release()
cv2.destroyAllWindows()
print("\nAR UI Closed. Thank you!")
