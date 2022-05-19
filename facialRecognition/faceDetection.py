import numpy as np
import cv2


faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

while True:
    ret, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
            gray, scaleFactor=1.01, minNeighbors=10, minSize=(30,30)) 

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]  
    cv2.imshow('Video',img)
    B = cv2.waitKey(2) & 0xff
    if B == 27: 
        break
    
cap.release()
cv2.destroyAllWindows()
    
        
    




















