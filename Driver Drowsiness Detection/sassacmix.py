import cv2
import time
import random
import datetime
import dlib
import numpy as np
from scipy.spatial import distance
from playsound import playsound
from pygame import mixer 
  # Starting the mixer 
mixer.init() 
  
# Loading the song 
mixer.music.load("beep_warning.mp3") 
  
# Setting the volume 
mixer.music.set_volume(0.7) 
  
# Start playing the song  
def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear_aspect_ratio = (A+B)/(2.0*C)
    return ear_aspect_ratio

hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")
cap = cv2.VideoCapture(0)

# Initialize variables to keep track of eye closure time
eye_closed_time = 0
last_eye_open_time = time.time()
total_eye_closed_time=0
threshold=5
facepresent=0
counter=0
drowssy=0
a=0
b=0
wrongmsg=""
while cap.isOpened():
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # cv2.putText(img,"No Person Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv2.LINE_AA)

    faces = hog_face_detector(gray)
    if(len(faces)>0):
        counter=0
    if(len(faces)>0 and facepresent==0):
        last_eye_open_time = time.time()
        facepresent=1
    else :
        counter+=1
        if(counter>=10):
            facepresent=0
            cv2.putText(img,"No Face Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv2.LINE_AA)

    for face in faces:
        face_landmarks = dlib_facelandmark(gray, face)
        leftEye = []
        rightEye = []

        for n in range(36,42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x,y))
            next_point = n+1
            if n == 41:
                next_point = 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(img,(x,y),(x2,y2),(0,255,0),1)

        for n in range(42,48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x,y))
            next_point = n+1
            if n == 47:
                next_point = 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(img,(x,y),(x2,y2),(0,255,0),1)

        left_ear = calculate_EAR(leftEye)
        right_ear = calculate_EAR(rightEye)

        EAR = (left_ear+right_ear)/2
        EAR = round(EAR,2)
        # if EAR<0.26:

        eyes_closed = True
        any_eye_open = False
        if(EAR>0.24):
            any_eye_open=True

        if not any_eye_open:
            eye_closed_time=time.time() - last_eye_open_time

        else:
            last_eye_open_time = time.time()
            eye_closed_time=0
        print(round(eye_closed_time,1))
        if(eye_closed_time>threshold and not drowssy):
            drowssy=1
            a=random.randint(10,30)
            b=random.randint(10,30)
            wrongmsg=""
        if(drowssy):
            cv2.putText(img,"Drowsiness Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
            if(not mixer.music.get_busy()):
               mixer.music.play()
            if(eye_closed_time<2):
                cv2.putText(img,"{}+{}".format(a,b), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
                cv2.putText(img,wrongmsg, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
                cv2.imshow('img', img)
                if cv2.waitKey(0) & 0xFF == ord(str((a+b)//10)):
                    if cv2.waitKey(0) & 0xFF == ord(str((a+b)%10)):
                        drowssy=0
                        last_eye_open_time = time.time()
                    else:
                        wrongmsg="Try again"
                else :
                    wrongmsg="Try again"
        else :
            cv2.putText(img,"Active",(20,50),
            cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),4)
           
            cv2.putText(img,"EAR:",(480,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),4)
            cv2.putText(img,str(EAR),(550,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),4)
            
       
    datet=str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    img=cv2.putText(img,datet,(30,470),cv2.FONT_HERSHEY_DUPLEX,1,(255,255,255),2,2)
    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
