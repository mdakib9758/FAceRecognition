import cv2
import numpy as np
import os


face=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
cam=cv2.VideoCapture(0);
rec=cv2.face.LBPHFaceRecognizer_create();
rec.read("recon\\train.yml")
id=0
font=cv2.FONT_HERSHEY_SIMPLEX
path='Data'
while(True):
    ret,img=cam.read();
    imagePath=[os.path.join(path,f) for f in os.listdir(path)]
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face.detectMultiScale(gray,1.2,5);
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        for i in imagePath:
                rd=int(os.path.split(i)[-1].split('.')[1])
                print(rd)
                if  rd==id:
                    d=os.path.split(i)[-1].split('.')[2]
                    break;
                    
                
        print(conf)
        if(conf<60):
            id=d
        else:
            id="unknown"
        cv2.putText(img,str(id),(x,x+h),font,2,(0,0,255),3);
    cv2.imshow("face",img);
    if(cv2.waitKey(1)==ord('q')):
        break;
cam.release()
cv2.destroyAllWindows()
