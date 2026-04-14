import cv2
import mediapipe as mp
from collections import deque
import time
import subprocess
import platform

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Gesture tracking
hand_history = deque(maxlen=10)  # Store last 10 hand positions for stability
point_history = deque(maxlen=10)  # Store recent pointing directions
pinch_history = deque(maxlen=5)  # Store recent pinch detections
last_gesture_time = 0
COOLDOWN = 1.0  # Seconds between gestures
SWIPE_THRESHOLD = 0.08  # Minimum distance for swipe detection
POINTING_THRESHOLD = 0.1  # Minimum index-finger vector length for pointing
POINTING_RATIO = 1.2  # Directional ratio for pointing vs. the other axis
RESET_THRESHOLD = 0.02  # Hand must settle before the next gesture
DIRECTION_RATIO = 0.6  # Fraction of movement that must be in one direction

gesture_ready = True

def is_consistent_movement(history, axis='x'):
    deltas = []
    for i in range(1, len(history)):
        if axis == 'x':
            delta = history[i][0] - history[i-1][0]
        else:
            delta = history[i][1] - history[i-1][1]
        deltas.append(delta)
    if not deltas:
        return False
    total = sum(deltas)
    if abs(total) < 0.001:
        return False
    direction = 1 if total > 0 else -1
    same_direction = sum(1 for d in deltas if d * direction > 0.001)
    return same_direction / len(deltas) >= DIRECTION_RATIO

def is_index_pointing(hand_landmarks):
    index_tip = hand_landmarks.landmark[8]
    index_pip = hand_landmarks.landmark[6]
    index_dip = hand_landmarks.landmark[7]
    middle_tip = hand_landmarks.landmark[12]
    ring_tip = hand_landmarks.landmark[16]
    pinky_tip = hand_landmarks.landmark[20]
    thumb_tip = hand_landmarks.landmark[4]

    # Require index finger extended and others reasonably folded
    index_extended = index_tip.y < index_pip.y < index_dip.y
    other_fingers_folded = (
        middle_tip.y > hand_landmarks.landmark[10].y and
        ring_tip.y > hand_landmarks.landmark[14].y and
        pinky_tip.y > hand_landmarks.landmark[18].y
    )
    thumb_out = abs(thumb_tip.x - index_tip.x) > 0.05
    return index_extended and other_fingers_folded and thumb_out


def is_fist(hand_landmarks):
    # Check if all fingers are curled (fist) - anytime all fingers are bent
    fingers = [
        (8, 6),   # index tip, pip
        (12, 10), # middle tip, pip
        (16, 14), # ring tip, pip
        (20, 18), # pinky tip, pip
    ]
    fingers_curled = all(hand_landmarks.landmark[tip].y > hand_landmarks.landmark[pip].y for tip, pip in fingers)
    return fingers_curled


def detect_pinch(hand_landmarks):
    # Detect if thumb and index finger are pinching (close together)
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    
    # Calculate distance between thumb and index finger tips
    distance = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)**0.5
    
    # Pinch detected if distance is small (< 0.05)
    if distance < 0.05:
        # Determine if pinch is on left or right side (based on hand x position)
        hand_x = hand_landmarks.landmark[9].x
        return 'left' if hand_x < 0.5 else 'right'
    return None


def detect_pointing_direction(hand_landmarks):
    if not is_index_pointing(hand_landmarks):
        return None

    wrist = hand_landmarks.landmark[0]
    index_tip = hand_landmarks.landmark[8]

    dx = index_tip.x - wrist.x
    dy = index_tip.y - wrist.y

    if abs(dx) > abs(dy) * POINTING_RATIO and abs(dx) > POINTING_THRESHOLD:
        return 'right' if dx > 0 else 'left'
    if abs(dy) > abs(dx) * POINTING_RATIO and abs(dy) > POINTING_THRESHOLD:
        return 'up' if dy < 0 else None
    return None


def play_media_key(key_name):
    """Cross-platform media key control"""
    system = platform.system()
    
    if system == "Darwin":  # macOS
        if key_name == 'next':
            cmd = 'tell application "Spotify" to next track'
        elif key_name == 'prev':
            cmd = 'tell application "Spotify" to previous track'
        elif key_name == 'play_pause':
            cmd = 'tell application "Spotify" to playpause'
        elif key_name == 'rewind':
            # Rewind by sending keyboard shortcut
            subprocess.run(['osascript', '-e', 'tell application "System Events" to key code 123 using {command down, option down}'])
            return
        elif key_name == 'fastforward':
            # Fast forward by sending keyboard shortcut
            subprocess.run(['osascript', '-e', 'tell application "System Events" to key code 124 using {command down, option down}'])
            return
        else:
            cmd = ''
        if cmd:
            subprocess.run(['osascript', '-e', cmd])
    elif system == "Windows":
        key_map = {
            'next': 'nexttrack',
            'prev': 'prevtrack',
            'play_pause': 'playpause',
            'rewind': 'medialast',
            'fastforward': 'medianext'
        }
        import pyautogui
        pyautogui.press(key_map.get(key_name, 'playpause'))
    else:  # Linux
        key_map = {
            'next': 'Next',
            'prev': 'Previous',
            'play_pause': 'PlayPause',
            'rewind': 'Previous',
            'fastforward': 'Next'
        }
        subprocess.run(['playerctl', key_map.get(key_name, 'play-pause')])

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Flip for selfie view
    image = cv2.flip(image, 1)
    h, w, c = image.shape
    
    # Convert to RGB for MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks and results.multi_handedness:
        hand_landmarks = results.multi_hand_landmarks[0]
        handedness = results.multi_handedness[0].classification[0].label
        
        # Get hand center position (average of all landmarks)
        palm_x = hand_landmarks.landmark[9].x
        palm_y = hand_landmarks.landmark[9].y
        confidence = hand_landmarks.landmark[9].z
        
        hand_history.append((palm_x, palm_y, time.time()))
        pointing = detect_pointing_direction(hand_landmarks)
        point_history.append(pointing)
        pinch = detect_pinch(hand_landmarks)
        pinch_history.append(pinch)
        
        # Draw hand landmarks
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        current_time = time.time()

        if not is_fist(hand_landmarks):
            # Reset gesture readiness when the hand is stable again
            if not gesture_ready and len(hand_history) > 3:
                xs = [p[0] for p in hand_history]
                ys = [p[1] for p in hand_history]
                if max(xs) - min(xs) < RESET_THRESHOLD and max(ys) - min(ys) < RESET_THRESHOLD:
                    gesture_ready = True

            # Detect swipe gestures first
            if gesture_ready and len(hand_history) > 3 and current_time - last_gesture_time > COOLDOWN:
                oldest_x, oldest_y, _ = hand_history[0]
                newest_x, newest_y, _ = hand_history[-1]

                dx = newest_x - oldest_x
                dy = newest_y - oldest_y

                if dx > SWIPE_THRESHOLD and abs(dy) < 0.1 and is_consistent_movement(hand_history, 'x'):
                    print("RIGHT SWIPE - Next Track")
                    play_media_key('next')
                    last_gesture_time = current_time
                    gesture_ready = False
                    hand_history.clear()
                    point_history.clear()
                    pinch_history.clear()
                elif dx < -SWIPE_THRESHOLD and abs(dy) < 0.1 and is_consistent_movement(hand_history, 'x'):
                    print("LEFT SWIPE - Previous Track")
                    play_media_key('prev')
                    last_gesture_time = current_time
                    gesture_ready = False
                    hand_history.clear()
                    point_history.clear()
                    pinch_history.clear()
                elif dy < -SWIPE_THRESHOLD and abs(dx) < 0.1 and is_consistent_movement(hand_history, 'y'):
                    print("UP SWIPE - Play/Pause")
                    play_media_key('play_pause')
                    last_gesture_time = current_time
                    gesture_ready = False
                    hand_history.clear()
                    point_history.clear()
                    pinch_history.clear()

            # Detect pointing gestures if the hand is not actively swiping
            if gesture_ready and len(point_history) == point_history.maxlen and current_time - last_gesture_time > COOLDOWN:
                if all(p == 'right' for p in point_history):
                    print("RIGHT POINT - Next Track")
                    play_media_key('next')
                    last_gesture_time = current_time
                    gesture_ready = False
                    hand_history.clear()
                    point_history.clear()
                    pinch_history.clear()
                elif all(p == 'left' for p in point_history):
                    print("LEFT POINT - Previous Track")
                    play_media_key('prev')
                    last_gesture_time = current_time
                    gesture_ready = False
                    hand_history.clear()
                    point_history.clear()
                    pinch_history.clear()
                elif all(p == 'up' for p in point_history):
                    print("UP POINT - Play/Pause")
                    play_media_key('play_pause')
                    last_gesture_time = current_time
                    gesture_ready = False
                    hand_history.clear()
                    point_history.clear()
                    pinch_history.clear()

            # Detect pinch gestures
            if gesture_ready and len(pinch_history) == pinch_history.maxlen and current_time - last_gesture_time > COOLDOWN:
                if all(p == 'left' for p in pinch_history):
                    print("LEFT PINCH - Rewind")
                    play_media_key('rewind')
                    last_gesture_time = current_time
                    gesture_ready = False
                    hand_history.clear()
                    point_history.clear()
                    pinch_history.clear()
                elif all(p == 'right' for p in pinch_history):
                    print("RIGHT PINCH - Fast Forward")
                    play_media_key('fastforward')
                    last_gesture_time = current_time
                    gesture_ready = False
                    hand_history.clear()
                    point_history.clear()
                    pinch_history.clear()

    # Display info
    cv2.putText(image, "Right=Next | Left=Prev | Up=Play/Pause",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(image, "Pinch Left=Rewind | Pinch Right=FastForward",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(image, "Press ESC to quit", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.imshow('Hand Gesture Music Control', image)
    if cv2.waitKey(5) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
