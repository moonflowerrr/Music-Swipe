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
hand_history = deque(maxlen=15)  # Store last 15 hand positions
last_gesture_time = 0
COOLDOWN = 1.0  # Seconds between gestures
SWIPE_THRESHOLD = 0.1  # Minimum distance for swipe detection
RESET_THRESHOLD = 0.02  # Hand must settle before next gesture
DIRECTION_RATIO = 0.75  # Fraction of movement that must be in one direction

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
        else:
            cmd = ''
        if cmd:
            subprocess.run(['osascript', '-e', cmd])
    elif system == "Windows":
        key_map = {
            'next': 'nexttrack',
            'prev': 'prevtrack',
            'play_pause': 'playpause'
        }
        import pyautogui
        pyautogui.press(key_map.get(key_name, 'playpause'))
    else:  # Linux
        key_map = {
            'next': 'Next',
            'prev': 'Previous',
            'play_pause': 'PlayPause'
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
        
        # Draw hand landmarks
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        current_time = time.time()

        # Reset gesture readiness when the hand is stable again
        if not gesture_ready and len(hand_history) > 5:
            xs = [p[0] for p in hand_history]
            ys = [p[1] for p in hand_history]
            if max(xs) - min(xs) < RESET_THRESHOLD and max(ys) - min(ys) < RESET_THRESHOLD:
                gesture_ready = True

        # Detect swipe gestures
        if gesture_ready and len(hand_history) > 5 and current_time - last_gesture_time > COOLDOWN:
            oldest_x, oldest_y, _ = hand_history[0]
            newest_x, newest_y, _ = hand_history[-1]

            dx = newest_x - oldest_x  # Positive = rightward, Negative = leftward
            dy = newest_y - oldest_y

            if dx > SWIPE_THRESHOLD and abs(dy) < 0.05 and is_consistent_movement(hand_history, 'x'):
                print("RIGHT SWIPE - Next Track")
                play_media_key('next')
                last_gesture_time = current_time
                gesture_ready = False
                hand_history.clear()

            elif dx < -SWIPE_THRESHOLD and abs(dy) < 0.05 and is_consistent_movement(hand_history, 'x'):
                print("LEFT SWIPE - Previous Track")
                play_media_key('prev')
                last_gesture_time = current_time
                gesture_ready = False
                hand_history.clear()

            elif dy < -SWIPE_THRESHOLD and abs(dx) < 0.05 and is_consistent_movement(hand_history, 'y'):
                print("UP SWIPE - Play/Pause")
                play_media_key('play_pause')
                last_gesture_time = current_time
                gesture_ready = False
                hand_history.clear()

    # Display info
    cv2.putText(image, "Right Swipe: Next | Left Swipe: Prev | Up Swipe: Play/Pause", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(image, "Press ESC to quit", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.imshow('Hand Gesture Music Control', image)
    if cv2.waitKey(5) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
