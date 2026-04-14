# Music-Swipe

Music-Swipe is a hand gesture-based music control tool that uses your webcam, OpenCV, and MediaPipe Hands to control playback, track skipping, seeking, and special speed effects.

## Features

- Swipe right/left to skip tracks
- Swipe up to play/pause
- Point right/left/up as an alternative to swipes
- Pinch left/right to rewind/fast forward continuously while holding
- Peace sign to activate "slow/reverb" mode (0.75x)
- Hand heart shape to activate "nightcore" mode (1.5x)

## Requirements

- Python 3.8+
- Webcam access
- `opencv-python`
- `mediapipe`

Optional:

- macOS: Homebrew + `sox` for more advanced audio speed manipulation
- Linux: `playerctl` if controlling compatible media players
- Windows: `pyautogui` for keyboard-based control

## Installation

1. Clone the repository

```bash
git clone https://github.com/moonflowerrr/Music-Swipe.git
cd Music-Swipe
```

2. Install Python dependencies

```bash
pip install opencv-python mediapipe
```

3. Optional macOS setup for advanced audio speed effects

```bash
brew install sox
```

> If you do not have administrator access, you can install Homebrew into your home directory or skip the optional step.

## Running the app

Start the gesture controller with:

```bash
python gesture_control.py
```

If the script cannot open your camera, make sure the webcam is connected and that Python has permission to use it.

## Gesture controls

### Track and playback control

- `Swipe right` → Next track
- `Swipe left` → Previous track
- `Swipe up` → Play/Pause

### Pointing controls

- `Point right` → Next track
- `Point left` → Previous track
- `Point up` → Play/Pause

### Pinch seeking

- `Pinch left` and hold → Rewind continuously
- `Pinch right` and hold → Fast forward continuously

### Speed effect gestures

- `Peace sign` → Slow/reverb mode (0.75x)
- `Hand heart` → Nightcore mode (1.5x)

## How the commands work

The script uses macOS AppleScript control for Spotify by default. On other platforms, it falls back to:

- Windows: `pyautogui` for media control hotkeys
- Linux: `playerctl` for supported media players

The speed effect commands are implemented in `gesture_control.py` via the `play_media_key` and `set_playback_speed` functions.

## Notes and limitations

- The slow/nightcore gestures are currently best-effort and may require additional audio tools to actually change playback speed.
- The script is designed for one hand at a time.
- Gesture detection depends on good lighting and a clear view of your hand.

## Troubleshooting

- If the peace sign or hand heart do not trigger, try making the gesture more clearly with straight index/middle fingers and folded ring/pinky fingers.
- If the camera does not start, verify that no other app is using the webcam.
- If Spotify does not respond on macOS, make sure Spotify is running and allowed to be controlled by AppleScript.

## Customization

If you want to change how gestures behave, edit `gesture_control.py`:

- `SWIPE_THRESHOLD` controls how much hand movement is needed for a swipe
- `POINTING_THRESHOLD` controls how far the finger must be extended for pointing
- `COOLDOWN` controls how long the script waits between gestures

## Exit

Press `ESC` while the window is active to quit the app.
