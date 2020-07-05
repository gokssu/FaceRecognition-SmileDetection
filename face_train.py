import cv2
import numpy as np
from PIL import Image
import os

path = 'data'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml")



def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceExmp = []
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')  # gri
        img_numpy = np.array(PIL_img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[0])

        faces = detector.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            faceExmp.append(img_numpy[y:y + h, x:x + w])
            ids.append(id)
    return faceExmp, ids



faces, ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))
recognizer.write('train/train.yml')


