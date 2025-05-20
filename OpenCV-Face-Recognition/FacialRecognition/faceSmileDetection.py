import cv2
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
face_cascade_path = os.path.join(BASE_DIR, 'Cascades', 'haarcascade_frontalface_default.xml')
smile_cascade_path = os.path.join(BASE_DIR, 'Cascades', 'haarcascade_smile.xml')

faceCascade = cv2.CascadeClassifier(face_cascade_path)
smileCascade = cv2.CascadeClassifier(smile_cascade_path)

if faceCascade.empty() or smileCascade.empty():
    print("[ERROR] Could not load cascade files.")
    exit()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 640)
cap.set(4, 480)

resize_width = 320

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]
    scale = resize_width / float(width)
    small_frame = cv2.resize(frame, (resize_width, int(height * scale)))
    gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray_small,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(50, 50)
    )

    faces = [(int(x / scale), int(y / scale), int(w / scale), int(h / scale)) for (x, y, w, h) in faces]

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        roi_gray = cv2.equalizeHist(cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY))
        roi_gray = cv2.medianBlur(roi_gray, 3)
        roi_color = frame[y:y+h, x:x+w]

        smiles = smileCascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.3,
            minNeighbors=20,
            minSize=(40, 40)
        )

        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)

    cv2.imshow('video', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
