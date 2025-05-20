# Explanation:
# Uses ultrasonic sensor to detect person within 1 meter.

# After detection, waits for touch sensor press to start face/eye/smile detection.

# Uses cv2.VideoCapture(0, cv2.CAP_V4L2) for libcamera on RPi 4.

# TTS announces recognized faces and smiles.

# Stops recognition on ESC key or after 30 seconds inactivity.

# Cleans up GPIO and camera on exit.

# Notes for setup:
# Make sure you have pyttsx3 installed on RPi (pip install pyttsx3).

# Make sure OpenCV is installed with face modules (opencv-contrib-python).

# The ultrasonic and touch sensor pins must be wired correctly as per your GPIO numbers.

# libcamera must be enabled and working on your Raspberry Pi (e.g., with sudo raspi-config).

# The rest of your Haar cascade XML and recognizer files stay the same.

import cv2
import numpy as np
import os
import time
import pyttsx3
import RPi.GPIO as GPIO
from libcamera import app

# --- Setup GPIO ---
GPIO.setmode(GPIO.BCM)

# Define pins (change as per your wiring)
ULTRASONIC_TRIG = 23
ULTRASONIC_ECHO = 24
TOUCH_SENSOR_PIN = 18

GPIO.setup(ULTRASONIC_TRIG, GPIO.OUT)
GPIO.setup(ULTRASONIC_ECHO, GPIO.IN)
GPIO.setup(TOUCH_SENSOR_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Assuming pull-up resistor

# --- TTS Setup ---
engine = pyttsx3.init()
def speak(text):
    print(f"[TTS] {text}")
    engine.say(text)
    engine.runAndWait()

# --- Snapshot folder ---
if not os.path.exists("snapshots"):
    os.makedirs("snapshots")

# --- Load recognizers and cascades ---
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
smileCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

font = cv2.FONT_HERSHEY_SIMPLEX
names = ['Unknown', 'Joan']

# --- Helper function for ultrasonic distance ---
def get_distance():
    # Send trigger pulse
    GPIO.output(ULTRASONIC_TRIG, True)
    time.sleep(0.00001)
    GPIO.output(ULTRASONIC_TRIG, False)

    start = time.time()
    stop = time.time()

    # Wait for echo start
    while GPIO.input(ULTRASONIC_ECHO) == 0:
        start = time.time()
    # Wait for echo end
    while GPIO.input(ULTRASONIC_ECHO) == 1:
        stop = time.time()

    elapsed = stop - start
    distance = (elapsed * 34300) / 2  # cm
    return distance

# --- Wait for touch press ---
def wait_for_touch():
    speak("Person detected. Please press the touch sensor to start recognition.")
    while GPIO.input(TOUCH_SENSOR_PIN) == GPIO.HIGH:
        time.sleep(0.1)  # Wait until pressed (assuming active low)
    speak("Starting face recognition.")

# --- Setup camera ---
cam = cv2.VideoCapture(0, cv2.CAP_V4L2)  # Use libcamera backend with Video4Linux2 interface
cam.set(3, 640)
cam.set(4, 480)

minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

last_spoken_name = ""
last_snapshot_time = 0
enable_smile_detection = True
enable_announcements = True

try:
    while True:
        distance = get_distance()
        if distance < 100:  # Person detected within 1 meter
            print(f"Person detected at {distance:.1f} cm")
            wait_for_touch()

            # Start detection loop after touch press
            detection_start = time.time()
            while True:
                ret, img = cam.read()
                if not ret:
                    break

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = faceCascade.detectMultiScale(
                    gray, scaleFactor=1.2, minNeighbors=5,
                    minSize=(int(minW), int(minH))
                )

                for (x, y, w, h) in faces:
                    face = gray[y:y + h, x:x + w]
                    face_resized = cv2.resize(face, (200, 200))
                    id, confidence = recognizer.predict(face_resized)
                    name = names[id] if confidence < 70 and id < len(names) else "Unknown"
                    confidence_text = f"  {round(100 - confidence)}%"

                    # Speak name once
                    if enable_announcements and name != last_spoken_name:
                        speak(f"Hello {name}")
                        last_spoken_name = name

                    # Draw rectangle & info
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(img, name, (x + 5, y - 5), font, 1, (255, 255, 255), 2)
                    cv2.putText(img, confidence_text, (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

                    roi_gray = gray[y:y + h, x:x + w]
                    roi_color = img[y:y + h, x:x + w]

                    # Eyes detection
                    eyes = eyeCascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(15, 15))
                    if len(eyes) > 0:
                        cv2.putText(img, "Eyes Detected", (x + 5, y + h + 20), font, 0.6, (255, 0, 0), 2)
                        for (ex, ey, ew, eh) in eyes:
                            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

                    # Smile detection
                    if enable_smile_detection:
                        smiles = smileCascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=20, minSize=(25, 25))
                        if len(smiles) > 0:
                            cv2.putText(img, "Smiling", (x + w - 100, y + h + 20), font, 0.6, (0, 0, 255), 2)

                            # Take snapshot & announce every 3 seconds max
                            if time.time() - last_snapshot_time > 3:
                                speak(f"{name} is smiling")
                                timestamp = int(time.time())
                                filename = f"snapshots/{name}_{timestamp}.jpg"
                                cv2.imwrite(filename, img)

                                with open("log.txt", "a") as log:
                                    log.write(f"{time.ctime(timestamp)}: {name} smiling - saved {filename}\n")

                                last_snapshot_time = time.time()

                            for (sx, sy, sw, sh) in smiles:
                                cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)

                cv2.imshow('camera', img)

                # Press ESC to exit detection early
                k = cv2.waitKey(10) & 0xff
                if k == 27:
                    speak("Stopping face recognition.")
                    break

                # Timeout detection after 30 seconds of inactivity (optional)
                if time.time() - detection_start > 30:
                    speak("No activity detected. Stopping face recognition.")
                    break

        else:
            print(f"No person detected. Distance: {distance:.1f} cm")
            time.sleep(0.5)

except KeyboardInterrupt:
    speak("Exiting program.")

finally:
    cam.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()
