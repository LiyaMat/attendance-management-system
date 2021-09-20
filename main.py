import cv2
import numpy as np
import face_recognition


imgElon=face_recognition.load_image_file('TrainingImage/elon org.jpg')
imgElon=cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
imgTest=face_recognition.load_image_file('TrainingImage/elon musk.jpg')
imgTest=cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

faceLoc=face_recognition.face_locations(imgElon)[0]
encodeElon=face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)
faceLoc=face_recognition.face_locations(imgTest)[0]
encodeTest=face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)


results=face_recognition.compare_faces([encodeElon],encodeTest)
facedistance=face_recognition.face_distance([encodeElon],encodeTest)
cv2.putText(imgTest,f'{results}{round(facedistance[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

down_width = 900
down_height = 900
down_points = (down_width, down_height)
imgElon= cv2.resize(imgElon, down_points, interpolation= cv2.INTER_LINEAR)
cv2.imshow('Elon Musk', imgElon)
imgTest = cv2.resize(imgTest, down_points, interpolation= cv2.INTER_LINEAR)
cv2.imshow('Elon TEst', imgTest)
cv2.waitKey()





