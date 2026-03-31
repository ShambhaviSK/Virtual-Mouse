import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Start webcam
cap = cv2.VideoCapture(0)

# Set up Mediapipe hand detector
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Get screen size
screen_w, screen_h = pyautogui.size()

# Variables
prev_x, prev_y = 0, 0
last_click_time = 0
scroll_cooldown  = 0

def fingers_up(landmarks):
    tips = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb
    if landmarks[tips[0]].x < landmarks[tips[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other 4 fingers
    for i in range(1, 5):
        if landmarks[tips[i]].y < landmarks[tips[i] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(frame, hand_landmarks,
            mp_hands.HAND_CONNECTIONS)

            lm  = hand_landmarks.landmark
            up  = fingers_up(lm)

            # Index fingertip position
            ix = int(lm[8].x * w)
            iy = int(lm[8].y * h)

            current_time = time.time()

            # ---- GESTURE 1: Move cursor (only index up) ----
            if up == [0, 1, 0, 0, 0]:
                mouse_x = np.interp(ix, [0, w], [0, screen_w])
                mouse_y = np.interp(iy, [0, h], [0, screen_h])

                curr_x = prev_x + (mouse_x - prev_x) / 5
                curr_y = prev_y + (mouse_y - prev_y) / 5

                pyautogui.moveTo(curr_x, curr_y)
                prev_x, prev_y = curr_x, curr_y
                cv2.putText(frame, "Move", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            # ---- GESTURE 2: Single / Double click (index + middle up) ----
            elif up == [0, 1, 1, 0, 0]:
                if current_time - last_click_time < 0.3:
                    pyautogui.doubleClick()
                    cv2.putText(frame, "DOUBLE CLICK", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                else:
                    pyautogui.click()
                    cv2.putText(frame, "CLICK", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                last_click_time = current_time

            # ---- GESTURE 3: Scroll up (index + pinky up) ----
            elif up == [0, 1, 0, 0, 1]:
                if current_time - scroll_cooldown > 0.1:   # ✅ changed 0.3 → 0.1
                    pyautogui.scroll(10)                    # ✅ changed 3 → 10
                    scroll_cooldown = current_time
                cv2.putText(frame, "SCROLL UP", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 255), 2)

            # ---- GESTURE 4: Scroll down (only pinky up) ----
            elif up == [0, 0, 0, 0, 1]:
                if current_time - scroll_cooldown > 0.1:   # ✅ changed 0.3 → 0.1
                    pyautogui.scroll(-10)                   # ✅ changed -3 → -10
                    scroll_cooldown = current_time
                cv2.putText(frame, "SCROLL DOWN", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Virtual Mouse", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()