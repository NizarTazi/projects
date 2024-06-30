import cv2
import mediapipe as mp
import numpy as np
import random

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils




window_name = 'Catch the Dot game'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1366, 768)  # Set window size to 1920x1080

#create a new random dot
def create_dot(frame):
    height, width, _ = frame.shape
    x = random.randint(50, width - 50)
    y = random.randint(50, height - 50)
    return (x, y)

#detect collision between dot and index finger
def detect_collision(dot, index_finger_tip):
    dot_x, dot_y = dot
    finger_x, finger_y = index_finger_tip
    distance = np.sqrt((dot_x - finger_x)**2 + (dot_y - finger_y)**2)
    return distance < 30


#move the dot
def move_dot(dot, direction, frame):
    height, width, _ = frame.shape
    x, y = dot
    dx, dy = direction

    x += dx
    y += dy

    #bounce off the edges
    if x <= 0 or x >= width:
        dx = -dx
    if y <= 0 or y >= height:
        dy = -dy

    return (x, y), (dx, dy)


dot = None
score_p1 = 0
score_p2 = 0

dot_direction=(100,100)


hand_to_player = {} #keep track of which hand is associated with which player




# main loop
cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        if dot is None:
            dot = create_dot(frame)

        for idx, (hand_landmarks, hand_handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
            hand_label = hand_handedness.classification[0].label
            player_id = hand_to_player.get(hand_label)

            if player_id is None:
                # Assign players based on the first detection
                if len(hand_to_player) == 0:
                    hand_to_player[hand_label] = "Player 1"
                    player_id = "Player 1"
                elif len(hand_to_player) == 1 and "Player 1" not in hand_to_player.values():
                    hand_to_player[hand_label] = "Player 1"
                    player_id = "Player 1"
                else:
                    hand_to_player[hand_label] = "Player 2"
                    player_id = "Player 2"

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            h, w, c = frame.shape
            index_finger_tip = (int(index_finger_tip.x * w), int(index_finger_tip.y * h))

            if detect_collision(dot, index_finger_tip):
                if player_id == "Player 1":
                    score_p1 += 1
                elif player_id == "Player 2":
                    score_p2 += 1
                dot = create_dot(frame)

    if dot is not None:
        dot, dot_direction = move_dot(dot, dot_direction, frame)
        cv2.circle(frame, dot, 15, (0, 0, 255), -1)

    cv2.putText(frame, f"Player 1 Score: {score_p1}", (10, 40), cv2.FONT_HERSHEY_TRIPLEX , 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Player 2 Score: {score_p2}", (10, 70), cv2.FONT_HERSHEY_TRIPLEX , 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imshow(window_name, frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
