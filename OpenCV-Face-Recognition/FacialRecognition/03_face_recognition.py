import cv2
import numpy as np
import os 

# Load the trained recognizer model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

# Load Haar cascade for face detection
cascadePath = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# Font settings for text overlay
font = cv2.FONT_HERSHEY_SIMPLEX

# List of names - index must match the ID used during training
# Index 0 is usually for 'Unknown'
names = ['Unknown', 'Joan']  # Add more names as needed: e.g., names[2] = "Alice"

# Initialize webcam
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # width
cam.set(4, 480)  # height

# Define minimum face size for detection
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

while True:
    ret, img = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        face = gray[y:y + h, x:x + w]
        face_resized = cv2.resize(face, (200, 200))  # Match training size

        id, confidence = recognizer.predict(face_resized)

        # Lower confidence means better match
        if confidence < 70:
            name = names[id] if id < len(names) else "Unknown"
        else:
            name = "Unknown"

        confidence_text = f"  {round(100 - confidence)}%"

        cv2.putText(img, str(name), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence_text), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    cv2.imshow('camera', img)

    k = cv2.waitKey(10) & 0xff
    if k == 27:  # Press 'ESC' to exit
        break

# Cleanup
print("\n[INFO] Exiting Program and cleaning up...")
cam.release()
cv2.destroyAllWindows()
