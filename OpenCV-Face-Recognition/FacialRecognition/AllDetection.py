import cv2
import numpy as np
import os

# === Define paths ===

# Go up two levels to reach 'new3/'
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
cascade_dir = os.path.join(base_dir, 'OpenCV-Face-Recognition', 'FacialRecognition', 'Cascades')
trainer_path = os.path.join(base_dir, 'trainer', 'trainer.yml')

# Debug print to confirm path
print(f"[INFO] Looking for trainer.yml at: {trainer_path}")
if not os.path.exists(trainer_path):
    print(f"[ERROR] Trainer file not found at: {trainer_path}")
    exit()

# === Load Haar cascades ===
face_cascade = cv2.CascadeClassifier(os.path.join(cascade_dir, 'haarcascade_frontalface_default.xml'))
eye_cascade = cv2.CascadeClassifier(os.path.join(cascade_dir, 'haarcascade_eye.xml'))
smile_cascade = cv2.CascadeClassifier(os.path.join(cascade_dir, 'haarcascade_smile.xml'))

# === Load face recognizer ===
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(trainer_path)

# === Define label names (match training IDs) ===
names = ['Unknown', 'Joan']  # ID 1 = Joan

# === Start webcam ===
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(100, 100)
    )

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Face recognition
        id_, confidence = recognizer.predict(roi_gray)
        if confidence < 55:
            label = names[id_] if id_ < len(names) else 'Unknown'
            label_text = f"{label} ({round(100 - confidence)}%)"
        else:
            label_text = "Unknown"

        # Draw face box and name
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Eyes
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=12)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # Smile
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=22)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)

    # Display
    cv2.imshow("Face | Eye | Smile Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
