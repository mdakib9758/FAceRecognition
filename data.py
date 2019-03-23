import cv2
import numpy as np

id=input('enter id')
name=input('enter name')
num=0
face=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
cam=cv2.VideoCapture(0);
while(True):
    ret,img=cam.read();
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face.detectMultiScale(gray,1.3,5);
    for(x,y,w,h) in faces:
        num=num+1;
        cv2.imwrite("Data/user."+str(id)+"."+str(name)+"."+str(num)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.waitKey(100);
    cv2.imshow("face",img);
    cv2.waitKey(1);
    if(num>30):
        break;
cam.release()
cv2.destroyAllWindows()
