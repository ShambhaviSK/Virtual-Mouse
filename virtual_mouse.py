import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Start webcam
cap = cv2.VideoCapture(0)  # 0 = your default camera

# Set up Mediapipe hand detector
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(
    max_num_hands=1,        # detect only 1 hand (faster)
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Get your screen size (PyAutoGUI needs this)
screen_w, screen_h = pyautogui.size()

def fingers_up(landmarks):
    """
    Returns a list of 5 values: [thumb, index, middle, ring, pinky]
    1 = finger is up, 0 = finger is down
    """
    tips = [4, 8, 12, 16, 20]   # landmark IDs for fingertips
    fingers = []

    # Thumb (checks left/right instead of up/down)
    if landmarks[tips[0]].x < landmarks[tips[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other 4 fingers (checks if tip is above the knuckle)
    for i in range(1, 5):
        if landmarks[tips[i]].y < landmarks[tips[i] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

prev_x, prev_y = 0, 0  # for smoothing mouse movement

while True:
    success, frame = cap.read()
    if not success:
        break

    # Flip frame (mirror effect — feels natural)
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert to RGB (Mediapipe needs RGB, OpenCV gives BGR)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            # Draw hand skeleton on frame
            mp_draw.draw_landmarks(frame, hand_landmarks,
            mp_hands.HAND_CONNECTIONS)

            lm = hand_landmarks.landmark  # shortcut to landmarks list

            # Get which fingers are up
            up = fingers_up(lm)

            # Index fingertip position (landmark 8)
            ix = int(lm[8].x * w)
            iy = int(lm[8].y * h)

            # ----- GESTURE 1: Move cursor (only index finger up) -----
            if up == [0, 1, 0, 0, 0]:
                # Map webcam coords to screen coords
                mouse_x = np.interp(ix, [0, w], [0, screen_w])
                mouse_y = np.interp(iy, [0, h], [0, screen_h])

                # Smoothing (avoids jittery movement)
                curr_x = prev_x + (mouse_x - prev_x) / 5
                curr_y = prev_y + (mouse_y - prev_y) / 5

                pyautogui.moveTo(curr_x, curr_y)
                prev_x, prev_y = curr_x, curr_y

            # ----- GESTURE 2: Left click (index + middle up) -----
            elif up == [0, 1, 1, 0, 0]:
                pyautogui.click()
                cv2.putText(frame, "CLICK", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the camera window
    cv2.imshow("Virtual Mouse", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()