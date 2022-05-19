import os
import cv2
import numpy as np #library renames to fr
import faceRecognition as fr #class renames to fr


#This module captures images via webcam and performs face recognition
face_recognizer = cv2.face.LBPHFaceRecognizer_create() # LBPH Recognizer
face_recognizer.read('trainingData.yml')# Load saved training data

name = {0:"Talha",1:"Priyanka", 2: "Kangana"} # id's from 0 to 2 with names


cap=cv2.VideoCapture(0) # captures video
cap.set(3,640)# Width
cap.set(4,480)# Height

while True:
    ret,test_img=cap.read()# captures frame and returns boolean value and captured image
    faces_detected,gray_img=fr.faceDetection(test_img)# using faceDetection function from fr which uses all the paremeters 


    for (x,y,w,h) in faces_detected: 
      cv2.rectangle(test_img,(x,y),(x+w,y+h),(0,0,255),thickness=2) 

    resized_img = cv2.resize(test_img, (640, 480))
    cv2.imshow('Face Detection',resized_img) #shoes 'Face Detected' resized image
    cv2.waitKey(10)


    for face in faces_detected:
        (x,y,w,h) = face
        roi_gray = gray_img[y:y+w, x:x+h]
        ID,confidence = face_recognizer.predict(roi_gray)#predicting the id of given image
        print("Confidence: ",confidence)
        print("ID: ",ID)
        fr.draw_rect(test_img, face)
        predicted_name = name[ID]
        
       
        if confidence < 40:#If confidence less than 37 then don't print predicted face text on screen
           fr.put_text(test_img,predicted_name,x,y)
          


    resized_img = cv2.resize(test_img, (640, 480))
    cv2.imshow('Face Recognition',resized_img)
    if cv2.waitKey(10) == ord('b'):#wait until 'b' key is pressed
        break


cap.release()
cv2.destroyAllWindows