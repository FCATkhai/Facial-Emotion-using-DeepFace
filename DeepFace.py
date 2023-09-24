import cv2
from deepface import DeepFace
import numpy as np

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
video = cv2.VideoCapture(1)
font = cv2.FONT_HERSHEY_PLAIN

print(cv2.WINDOW_NORMAL)
while video.isOpened():
    _,frame = video.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for x, y, w, h in face:
        img = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
        try:
            analyze = DeepFace.analyze(frame, actions=["emotion"])
            print(analyze[0]["dominant_emotion"])
            status = analyze[0]["dominant_emotion"]
        except:
            print("no face")
            status = "no face"
        x1, y1, w1, h1 = 0, 0, 175, 75
        # Draw black background rectangle
        cv2.rectangle(frame, (x1, x1), (x1+ w1, y1 + h1), (0,0,0), -1)
        # Add text
        cv2.putText(frame, status, (x1 + int(w1/10),y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.putText(frame, status, (100, 150), font, 3, (0, 0, 255),2,cv2.LINE_4)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255))

    cv2.namedWindow("video", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("video", 900, 900)
    cv2.imshow("video", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
video.release()

# imgpath = "test.jpg"
# image = cv2.imread(imgpath)

# analyze = DeepFace.analyze(image, actions=["emotion"])
# print(type(analyze[0]["dominant_emotion"]))