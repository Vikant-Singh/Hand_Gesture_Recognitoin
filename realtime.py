import os
import pickle
import numpy as np
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands

DATA_DIR = './data'

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

data = []
labels = []

# Define maximum number of hand landmarks
max_landmarks = 21  # Assuming you're detecting 21 landmarks per hand

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y])

                # Pad or truncate landmarks to a fixed length
                landmarks_len = len(landmarks)
                if landmarks_len < 2 * max_landmarks:
                    landmarks.extend([0.0] * (2 * max_landmarks - landmarks_len))
                elif landmarks_len > 2 * max_landmarks:
                    landmarks = landmarks[:2 * max_landmarks]

                data.append(landmarks)
                labels.append(dir_)

# Save data to a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
