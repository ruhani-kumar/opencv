import cv2
import numpy as np
import os
import time
import pyttsx3

# ---------- SETUP ----------

# Initialize TTS
engine = pyttsx3.init()
def speak(text):
    print(f"[TTS] {text}")
    engine.say(text)
    engine.runAndWait()

# Create snapshots folder
if not os.path.exists("snapshots"):
    os.makedirs("snapshots")

# Load trained recognizer and Haar cascades
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
smileCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

font = cv2.FONT_HERSHEY_SIMPLEX
names = ['Unknown', 'Joan']  # Update with more trained names if needed

# ---------- MENU SYSTEM ----------

# Default feature flags
enable_smile_detection = True
enable_announcements = True

def audio_menu():
    global enable_smile_detection, enable_announcements
    speak("Welcome to the face detection system.")
    speak("Press 1 to enable smile detection.")
    speak("Press 2 to disable name announcements.")
    speak("Press 3 to hear system time.")
    speak("Press 4 to continue to face detection.")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "1":
        enable_smile_detection = True
        speak("Smile detection enabled.")
    elif choice == "2":
        enable_announcements = False
        speak("Name announcements disabled.")
    elif choice == "3":
        speak("The current time is " + time.ctime())
    elif choice == "4":
        speak("Starting face detection.")
    else:
        speak("Invalid choice. Please try again.")
        audio_menu()

# Run the menu
audio_menu()

# ---------- CAMERA SETUP ----------
cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

last_spoken_name = ""
last_snapshot_time = 0

# ---------- MAIN LOOP ----------
while True:
    ret, img = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        face_resized = cv2.resize(face, (200, 200))

        id, confidence = recognizer.predict(face_resized)
        name = names[id] if confidence < 70 and id < len(names) else "Unknown"
        confidence_text = f"  {round(100 - confidence)}%"

        # Speak name if allowed
        if enable_announcements and name != last_spoken_name:
            speak(f"Hello {name}")
            last_spoken_name = name

        # Draw face rectangle and info
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, name, (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, confidence_text, (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        # Detect eyes
        eyes = eyeCascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(15, 15))
        if len(eyes) > 0:
            cv2.putText(img, "Eyes Detected", (x + 5, y + h + 20), font, 0.6, (255, 0, 0), 2)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

        # Detect smiles
        if enable_smile_detection:
            smiles = smileCascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=20, minSize=(25, 25))
            if len(smiles) > 0:
                cv2.putText(img, "Smiling", (x + w - 100, y + h + 20), font, 0.6, (0, 0, 255), 2)

                # Trigger snapshot + speech once every 3 sec
                if time.time() - last_snapshot_time > 3:
                    speak(f"{name} is smiling")
                    timestamp = int(time.time())
                    filename = f"snapshots/{name}_{timestamp}.jpg"
                    cv2.imwrite(filename, img)

                    # Log event
                    with open("log.txt", "a") as log:
                        log.write(f"{time.ctime(timestamp)}: {name} smiling - saved {filename}\n")

                    last_snapshot_time = time.time()

                for (sx, sy, sw, sh) in smiles:
                    cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)

    cv2.imshow('camera', img)

    k = cv2.waitKey(10) & 0xff
    if k == 27:  # ESC to exit
        break

# ---------- CLEANUP ----------
speak("Exiting program.")
cam.release()
cv2.destroyAllWindows()
