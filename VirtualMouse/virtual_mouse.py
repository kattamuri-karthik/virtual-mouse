import cv2
import numpy as np
import pyautogui
import math

screen_width, screen_height = pyautogui.size()

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

prev_click_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    roi = frame[100:400, 100:400]
    cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Skin color range
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (5, 5), 100)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        hand = max(contours, key=cv2.contourArea)

        if cv2.contourArea(hand) > 3000:
            hull = cv2.convexHull(hand)
            hull_indices = cv2.convexHull(hand, returnPoints=False)

            if hull_indices is not None and len(hull_indices) > 3:
                defects = cv2.convexityDefects(hand, hull_indices)
                finger_count = 0

                if defects is not None:
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start = tuple(hand[s][0])
                        end = tuple(hand[e][0])
                        far = tuple(hand[f][0])

                        a = math.dist(start, end)
                        b = math.dist(start, far)
                        c = math.dist(end, far)

                        angle = math.acos((b**2 + c**2 - a**2) / (2 * b * c)) * 57

                        if angle <= 90:
                            finger_count += 1

                finger_count += 1  # include thumb

                # Hand center
                moments = cv2.moments(hand)
                if moments["m00"] != 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])

                    screen_x = np.interp(cx, [0, 300], [0, screen_width])
                    screen_y = np.interp(cy, [0, 300], [0, screen_height])

                # Gesture actions
                if finger_count >= 4:
                    pyautogui.moveTo(screen_x, screen_y)

                elif finger_count == 1:
                    pyautogui.click()
                    pyautogui.sleep(0.4)

                elif finger_count == 2:
                    pyautogui.rightClick()
                    pyautogui.sleep(0.4)

                cv2.putText(
                    frame,
                    f"Fingers: {finger_count}",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

    cv2.imshow("Virtual Mouse - Hand Gestures", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
