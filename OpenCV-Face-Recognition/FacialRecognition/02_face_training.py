import cv2
import numpy as np
from PIL import Image
import os

# Path for face image database
dataset_path = 'dataset'
trainer_path = 'trainer'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:
        try:
            PIL_img = Image.open(imagePath).convert('L')
            img_numpy = np.array(PIL_img, 'uint8')
            id = int(os.path.split(imagePath)[-1].split(".")[1])

            faces = detector.detectMultiScale(img_numpy)
            for (x, y, w, h) in faces:
                face = img_numpy[y:y+h, x:x+w]
                face = cv2.resize(face, (200, 200))  # Resize for consistency
                faceSamples.append(face)
                ids.append(id)
        except Exception as e:
            print(f"[WARNING] Skipping {imagePath}: {e}")

    return faceSamples, ids

print("\n[INFO] Training faces. Please wait...")
faces, ids = getImagesAndLabels(dataset_path)
recognizer.train(faces, np.array(ids))

# Save model
if not os.path.exists(trainer_path):
    os.makedirs(trainer_path)

recognizer.write(os.path.join(trainer_path, 'trainer.yml'))
print(f"\n[INFO] {len(np.unique(ids))} unique face(s) trained. Model saved to {trainer_path}/trainer.yml.")
