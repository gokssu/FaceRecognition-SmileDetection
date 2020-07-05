import cv2

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)
faceDetection = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')

max_fotonumber = 30
face_id = 1

count = 0

while (True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetection.detectMultiScale(gray, 1.2, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1
        cv2.imwrite("data/" + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])
        cv2.imshow('Video', img)


    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
    elif count >= max_fotonumber:
        break

cam.release()
cv2.destroyAllWindows()