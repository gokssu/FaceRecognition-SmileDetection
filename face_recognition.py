import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def print_utf8_text(image, xy, text, color):
    fontName = 'FreeSerif.ttf'
    font = ImageFont.truetype(fontName, 24)
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    draw.text((xy[0], xy[1]), text, font=font,
              fill=(color[0], color[1], color[2], 0))
    image = np.array(img_pil)
    return image


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('train/train.yml')
cascadePath = "Cascades/haarcascade_frontalface_default.xml"
smilecascadePath = "Cascades/haarcascade_smile.xml"
smile_cascade = cv2.CascadeClassifier(smilecascadePath);
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX

id = 0
names = ['None', 'NAME1 ', 'NAME2','NAME3']


cam = cv2.VideoCapture(0)
cam.set(3, 1000)
cam.set(4, 800)

minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)
while True:
    ret, img = cam.read()
    gri = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gri,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gri[y:y + h, x:x + w])

        if (confidence < 100):
            id= names[id]

        else:
            id = "unknown"

        color = (255, 255, 255)
        cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 0, 255), 2)

        smile = smile_cascade.detectMultiScale(gri,
                                               scaleFactor=1.7,
                                               minNeighbors=22,
                                               minSize=(25, 25),
                                               )

        for (ex, ey, ew, eh) in smile:
            cv2.rectangle(gri, (ex, ey), (ex + ew, ey + eh), (0, 255, 255), 2)
            smile = "keep smiling "
            cv2.putText(img, str(smile), (x + 5, y + h + 25), font, 1, (255, 255, 0), 2)





    cv2.imshow('Video', img)

    k = cv2.waitKey(10) & 0xff
    if k == 27 or k == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
