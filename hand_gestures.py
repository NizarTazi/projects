import cv2
import mediapipe as mp
import numpy as np
import math
import collections

mp_hands=mp.solutions.hands
hands=mp_hands.Hands()
mp_drawing=mp.solutions.drawing_utils

#calculate the Euclidean distance between two points
def calculate_distance(point1, point2):
    return np.linalg.norm(np.array([point1.x, point1.y]) - np.array([point2.x, point2.y]))

#calculate the angle between three points
def calculate_angle(a, b, c):
    ba = np.array([a.x - b.x, a.y - b.y])
    bc = np.array([c.x - b.x, c.y - b.y])
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

#classify the hand sign
def classify_hand_sign(hand_landmarks):
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

    #Calculate distances from wrist to finger tips
    distances = {
        "thumb": calculate_distance(wrist, thumb_tip),
        "index": calculate_distance(wrist, index_tip),
        "middle": calculate_distance(wrist, middle_tip),
        "ring": calculate_distance(wrist, ring_tip),
        "pinky": calculate_distance(wrist, pinky_tip)
    }

    #Calculate angles for thumb and index finger
    thumb_angle = calculate_angle(thumb_mcp, thumb_tip, index_mcp)
    index_angle = calculate_angle(index_mcp, index_tip, middle_mcp)

    #Define thresholds for recognizing gestures
    open_hand_threshold = 0.3 

    if all(d > open_hand_threshold for d in distances.values()):
        return "Open Hand"
    if distances["thumb"] > open_hand_threshold and all(distances[finger] < open_hand_threshold for finger in ["index", "middle", "ring", "pinky"]):
        return "Thumbs up"
    if distances["index"] > open_hand_threshold and distances["middle"] > open_hand_threshold and all(distances[finger] < open_hand_threshold for finger in ["ring", "pinky","thumb"]):
        return "Peace Hand"
    if distances["thumb"] > open_hand_threshold and all(distances[finger] < open_hand_threshold for finger in ["index", "middle", "ring"]):
        return "Surf Hand"
    if all(distances[finger] < open_hand_threshold for finger in ["thumb","index", "middle", "ring","pinky"]):
        return "Closed Hand"
    if distances["middle"] > open_hand_threshold and all(distances[finger] < open_hand_threshold for finger in ["thumb","index", "pinky", "ring"]):
        return "MIDDLE FINGER"
    if all(distances[finger] < open_hand_threshold for finger in ["index","thumb"]):
        return "SIKE lmao"
    if distances["thumb"] > open_hand_threshold and distances["middle"] > open_hand_threshold and distances["index"] > open_hand_threshold and all(distances[finger] < open_hand_threshold for finger in ["pinky", "ring"]):
        return "3 times MR OLYMPIA"
    return "Unknown"

# ------------- WAVING -----------------

motion_history_x = collections.deque(maxlen=15)  
def detect_waving(wrist_x):
    motion_history_x.append(wrist_x)
    
    if len(motion_history_x) == motion_history_x.maxlen:
        #Calculate the range of motion
        motion_range = max(motion_history_x) - min(motion_history_x)
        
        #detect back and forth motion
        if motion_range > 0.1:  # Adjust this threshold as needed
            movements = np.diff(motion_history_x)
            positive_movements = movements > 0
            negative_movements = movements < 0
            
            #Count the changes in direction
            changes = np.diff(positive_movements.astype(int))
            num_changes = np.sum(np.abs(changes))
            
            if num_changes > 4:  #Heuristic threshold for waving motion
                return "Waving"
    return "Not Waving"

# ----------- Runnning the app ---------------

# Open the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = classify_hand_sign(hand_landmarks)

            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            waving = detect_waving(wrist.x)  # Using wrist.x for motion detection
            cv2.putText(image, f"{gesture}, {waving}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            

            #Display distances and angles for debugging
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

            cv2.putText(image, f"T: {calculate_distance(wrist, thumb_tip):.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, f"I: {calculate_distance(wrist, index_tip):.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, f"M: {calculate_distance(wrist, middle_tip):.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, f"R: {calculate_distance(wrist, ring_tip):.2f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, f"P: {calculate_distance(wrist, pinky_tip):.2f}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    cv2.imshow('Hand Detection', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()