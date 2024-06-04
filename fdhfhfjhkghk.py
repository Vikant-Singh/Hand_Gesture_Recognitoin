# for extracting the data from images saving in the pickle file
# import os
# import pickle
# import numpy as np
# import mediapipe as mp
# import cv2
#
# mp_hands = mp.solutions.hands
#
# DATA_DIR = './data'
#
# hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
#
# data = []
# labels = []
#
# # Define maximum number of hand landmarks
# max_landmarks = 21  # Assuming you're detecting 21 landmarks per hand
#
# for dir_ in os.listdir(DATA_DIR):
#     for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
#         data_aux = []
#
#         img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#         results = hands.process(img_rgb)
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 landmarks = []
#                 for landmark in hand_landmarks.landmark:
#                     landmarks.extend([landmark.x, landmark.y])
#
#                 # Pad or truncate landmarks to a fixed length
#                 landmarks_len = len(landmarks)
#                 if landmarks_len < 2 * max_landmarks:
#                     landmarks.extend([0.0] * (2 * max_landmarks - landmarks_len))
#                 elif landmarks_len > 2 * max_landmarks:
#                     landmarks = landmarks[:2 * max_landmarks]
#
#                 data.append(landmarks)
#                 labels.append(dir_)
#
# # Save data to a pickle file
# with open('data.pickle', 'wb') as f:
#     pickle.dump({'data': data, 'labels': labels}, f)



# for training the model
# import pickle
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import numpy as np
#
# # Load the data from the pickle file
# data_dict = pickle.load(open('./data.pickle', 'rb'))
# data = np.asarray(data_dict['data'])
# labels = np.asarray(data_dict['labels'])
#
# # Split the data into training and testing sets
# x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
#
# # Initialize the RandomForestClassifier model
# model = RandomForestClassifier()
#
# # Train the model
# model.fit(x_train, y_train)
#
# # Predict labels for the test set
# y_predict = model.predict(x_test)
#
# # Calculate the accuracy score
# score = accuracy_score(y_predict, y_test)
#
# # Print the accuracy
# print('{}% of samples were classified correctly !'.format(score * 100))
#
# # Save the trained model to a file
# with open('model.p', 'wb') as f:
#     pickle.dump({'model': model}, f)


#testing the model
#
#
# import pickle
# import cv2
# import mediapipe as mp
# import numpy as np
#
# # Load the trained model
# model_dict = pickle.load(open('./model.p', 'rb'))
# model = model_dict['model']
#
# # Initialize video capture from webcam
# cap = cv2.VideoCapture(0)
#
# # Initialize MediaPipe Hands
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
#
# hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, max_num_hands=1)
#
# # Map gesture labels to characters
# labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # Convert the frame to RGB
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#     # Detect hand landmarks
#     results = hands.process(frame_rgb)
#
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(
#                 frame,  # image to draw
#                 hand_landmarks,  # model output
#                 mp_hands.HAND_CONNECTIONS,  # hand connections
#                 mp_drawing_styles.get_default_hand_landmarks_style(),
#                 mp_drawing_styles.get_default_hand_connections_style())
#
#         # Extract hand landmarks and normalize coordinates
#         landmarks = []
#         for landmark in hand_landmarks.landmark:
#             landmarks.extend([landmark.x, landmark.y])
#
#         # Make a prediction using the trained model
#         prediction = model.predict([landmarks])
#
#         # Overlay the predicted label on the frame
#         predicted_character = labels_dict[int(prediction[0])]
#         cv2.putText(frame, f'Prediction: {predicted_character}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
#                     cv2.LINE_AA)
#
#     # Display the processed video stream
#     cv2.imshow('Hand Gesture Recognition', frame)
#
#     # Exit the loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Release the video capture and close all windows
# cap.release()
# cv2.destroyAllWindows()
