import numpy as np
import cv2
import face_recognition
import os
from datetime import datetime

path="ImageAttendance"
images=[]
classnames=[]
mylist=os.listdir(path)

for cls in mylist:
    currentimg=cv2.imread(f'{path}/{cls}')
    images.append(currentimg)
    classnames.append(os.path.splitext(cls)[0])

print(classnames)

def findEncodings(images):
    encodelist=[]
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

def markattendance(name):
    with open('Attendance.csv','r+') as f:
        myDatalist= f.readlines()
        namelist=[]
        for line in myDatalist:
            entry=line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now= datetime.now()
            dtString = now.strftime('%h:%M:%S')
            f.writelines(f'\n{name},{dtString}')


encodelistlistknown=findEncodings(images)
print('Encoding Complete')

cap= cv2.VideoCapture(0)
while True:
    success, img=cap.read()
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS,faceCurFrame)

    for encodeface,faceloc in zip(encodeCurFrame,faceCurFrame):
        matches=face_recognition.compare_faces(encodelistlistknown,encodeface)
        faceDis=face_recognition.face_distance(encodelistlistknown,encodeface)
        matchIndex=np.argmin(faceDis)

        if matches[matchIndex]:
            name=classnames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1=faceloc
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markattendance(name)


    cv2.imshow('Webcam',img)
    cv2.waitKey(1)