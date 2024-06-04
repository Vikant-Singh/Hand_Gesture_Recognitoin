import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 5
dataset_size = 150

cap = cv2.VideoCapture(0)  # Use index 0 for the default webcam

for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    print('Ready? Press "Q" to start collecting.')
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" to start collecting.', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    print('Capturing images...')
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

        print('Saved image {} for class {}'.format(counter, j))

        counter += 1

print('Data collection completed.')
cap.release()
cv2.destroyAllWindows()

