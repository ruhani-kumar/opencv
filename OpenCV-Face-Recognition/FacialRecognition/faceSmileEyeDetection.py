import cv2
import os

# Get script directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths to cascade XML files
face_cascade_path = os.path.join(BASE_DIR, 'Cascades', 'haarcascade_frontalface_default.xml')
eye_cascade_path = os.path.join(BASE_DIR, 'Cascades', 'haarcascade_eye.xml')
smile_cascade_path = os.path.join(BASE_DIR, 'Cascades', 'haarcascade_smile.xml')

# Load cascades
faceCascade = cv2.CascadeClassifier(face_cascade_path)
eyeCascade = cv2.CascadeClassifier(eye_cascade_path)
smileCascade = cv2.CascadeClassifier(smile_cascade_path)

# Check if cascades loaded
if faceCascade.empty() or eyeCascade.empty() or smileCascade.empty():
    print("[ERROR] Could not load one or more cascade files.")
    exit()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

resize_width = 320  # for speed-up

while True:
    ret, frame = cap.read()
    if not ret:
        break

    

    # Resize frame for faster detection
    height, width = frame.shape[:2]
    scale = resize_width / float(width)
    small_frame = cv2.resize(frame, (resize_width, int(height * scale)))
    gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

    # Detect faces on resized frame
    faces = faceCascade.detectMultiScale(
        gray_small,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(50, 50)
    )

    # Scale detected faces to original frame size
    faces = [(int(x / scale), int(y / scale), int(w / scale), int(h / scale)) for (x, y, w, h) in faces]

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        roi_gray = cv2.equalizeHist(cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY))
        roi_gray = cv2.medianBlur(roi_gray, 3)
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes
        eyes = eyeCascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(15, 15)
        )
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # Detect smiles
        smiles = smileCascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.3,
            minNeighbors=20,
            minSize=(30, 30)
        )
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 255), 2)

    cv2.imshow('Face, Eye and Smile Detection', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to quit
        break

cap.release()
cv2.destroyAllWindows()
