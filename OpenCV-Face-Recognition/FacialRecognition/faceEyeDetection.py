import cv2
import os

# Paths for cascades
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
face_cascade_path = os.path.join(BASE_DIR, 'Cascades', 'haarcascade_frontalface_default.xml')
eye_cascade_path = os.path.join(BASE_DIR, 'Cascades', 'haarcascade_eye.xml')

faceCascade = cv2.CascadeClassifier(face_cascade_path)
eyeCascade = cv2.CascadeClassifier(eye_cascade_path)

if faceCascade.empty() or eyeCascade.empty():
    print("[ERROR] Could not load cascade files.")
    exit()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow backend for faster capture on Windows

# Resize factor (speed up detection)
resize_width = 320

while True:
    ret, frame = cap.read()
    if not ret:
        break

    

    # Resize frame to speed up detection
    height, width = frame.shape[:2]
    scale = resize_width / float(width)
    small_frame = cv2.resize(frame, (resize_width, int(height * scale)))

    gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

    # Detect faces on small frame for speed
    faces = faceCascade.detectMultiScale(
        gray_small,
        scaleFactor=1.1,      # smaller step → better accuracy, but slower
        minNeighbors=6,       # increase to reduce false positives
        minSize=(50, 50)      # ignore smaller faces for stability
    )

    # Scale face coords back to original frame size
    faces = [(int(x / scale), int(y / scale), int(w / scale), int(h / scale)) for (x, y, w, h) in faces]

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        roi_gray = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
        roi_color = frame[y:y + h, x:x + w]

        eyes = eyeCascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.3,    # eyes can use larger scaleFactor for speed
            minNeighbors=8,
            minSize=(15, 15)
        )

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    cv2.imshow('video', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
