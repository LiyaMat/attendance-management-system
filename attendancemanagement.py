import cv2
import numpy as np
import face_recognition
import os
from datetime import  datetime

#impoerting images from folder
path='TrainingImage'
images=[]
names=[]
list=os.listdir(path)
for item in list:
    currentItem=cv2.imread(f'{path}/{item}')
    images.append(currentItem)
    names.append(os.path.splitext(item)[0])


#find encodings

def findencodings(images):
    encodeList=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode= face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
encodeListKnown=findencodings(images)
print("Encoding sucess" )
def markAttendence(name):
    with open('attendancesheet.csv','r+') as f:
        dataList=f.readlines()
        nameList=[]
        for line in dataList:
            entry=line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now=datetime.now()
            dtstring=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtstring}')



#matching images from webcam
capture=cv2.VideoCapture(0)
while True:
    sucess , img=capture.read()
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)#reducing size increases accuracy
    imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
    #find encoding of webcam
    CurentFrameFaces=face_recognition.face_locations(imgS)
    encodeCurentFrame=face_recognition.face_encodings(imgS,CurentFrameFaces)

    for encodeFace,faceloc in zip(encodeCurentFrame,CurentFrameFaces):
        matches=face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDistance=face_recognition.face_distance(encodeListKnown,encodeFace)
        matchIndex=np.argmin(faceDistance)

        #bounding box
        if matches[matchIndex]:
            name=names[matchIndex].upper()
            #print(name)
            y1,x2,y2,x1=faceloc
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,0.5,
                        (255,255,255),2)
            markAttendence(name)



    cv2.imshow('Webcam',img)
    cv2.waitKey(1)
