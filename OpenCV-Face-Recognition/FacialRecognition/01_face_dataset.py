import cv2
import os
import time

# Initialize webcam
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # width
cam.set(4, 480)  # height

# Load Haar cascade
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Get user ID input
face_id = input('\nEnter user ID (e.g., 1, 2, 3): ')
print("\n[INFO] Initializing face capture. Look at the camera...")

# Create dataset folder if it doesn't exist
dataset_dir = 'dataset'
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

count = 0
last_capture_time = time.time()

while True:
    ret, img = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        if time.time() - last_capture_time > 0.2:
            count += 1
            last_capture_time = time.time()

            face_img = gray[y:y + h, x:x + w]
            face_img = cv2.resize(face_img, (200, 200))
            # Optional: Enhance contrast
            face_img = cv2.equalizeHist(face_img)

            filename = f"User.{face_id}.{count}.jpg"
            cv2.imwrite(os.path.join(dataset_dir, filename), face_img)

            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, f"Image Count: {count}/30", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('image', img)

    k = cv2.waitKey(1) & 0xff
    if k == 27 or count >= 30:
        break

print("\n[INFO] Done capturing. Cleaning up...")
cam.release()
cv2.destroyAllWindows()
