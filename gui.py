import tkinter as tk
import threading
import pickle
import cv2
import mediapipe as mp
import PIL
import numpy as np


class HandGestureRecognitionApp:
    def __init__(self, master):
        self.master = master
        master.title("Hand Gesture Project")
        master.geometry("420x420")  # Decrease window size by 30%
        master.configure(bg="yellow")

        # Create heading label
        self.heading_label = tk.Label(master, text="Hand Gesture Recognition", font=("Arial", 16), bg="yellow")
        self.heading_label.pack(pady=10)

        # Calculate padding for the buttons (15% of window width)
        button_padding = int(0.15 * master.winfo_width())

        # Create start button with increased size and padding
        self.start_button = tk.Button(master, text="Start", font=("Arial", 14), bg="green",
                                      command=self.start_recognition)
        self.start_button.pack(side="right", padx=(button_padding, 40), pady=5, ipadx=10,
                               ipady=3)  # Increase button size by 20%

        # Create stop button with increased size and padding
        self.stop_button = tk.Button(master, text="Stop", font=("Arial", 14), bg="red", command=self.stop_recognition)
        self.stop_button.pack(side="left", padx=(40, button_padding), pady=5, ipadx=10,
                              ipady=3)  # Increase button size by 20%

        # Initialize video capture from webcam
        self.cap = cv2.VideoCapture(0)

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, max_num_hands=1)

        # Map gesture labels to characters
        self.labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}

        self.running = False

    def start_recognition(self):
        self.running = True
        threading.Thread(target=self.hand_gesture_recognition).start()

    def stop_recognition(self):
        self.running = False

    def hand_gesture_recognition(self):
        # Load the trained model
        model_dict = pickle.load(open('./model.p', 'rb'))
        model = model_dict['model']

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Convert the frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect hand landmarks
            results = self.hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame,  # image to draw
                        hand_landmarks,  # model output
                        self.mp_hands.HAND_CONNECTIONS,  # hand connections
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style())

                # Extract hand landmarks and normalize coordinates
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y])

                # Make a prediction using the trained model
                prediction = model.predict([landmarks])

                # Overlay the predicted label on the frame
                predicted_character = self.labels_dict[int(prediction[0])]
                cv2.putText(frame, f'Prediction: {predicted_character}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2,
                            cv2.LINE_AA)

            # Display the processed video stream
            cv2.imshow('Hand Gesture Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the capture and close windows
        self.cap.release()
        cv2.destroyAllWindows()


def main():
    root = tk.Tk()
    app = HandGestureRecognitionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
