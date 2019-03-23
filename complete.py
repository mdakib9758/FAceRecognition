import cv2
import numpy as np
import os
from PIL import Image

face=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
cam=cv2.VideoCapture(0);
rec=cv2.face.LBPHFaceRecognizer_create();
def ran():
    d=0
    f=0
    rec.read("recon\\train.yml")
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
            print(conf)
            if(conf<55):
                for i in imagePath:
                  rd=int(os.path.split(i)[-1].split('.')[1])
                  print(rd)
                  if  rd==id:
                    d=os.path.split(i)[-1].split('.')[2]
                    break;
                id=d
            else:
                id="unknown pres w to add"
                if(cv2.waitKey(2)==ord('w')):
                    main()
                    f=1
                    break;
            cv2.putText(img,str(id),(x,x+h),font,1,(0,0,255),2);
        cv2.imshow("face",img);
        if(cv2.waitKey(1)==ord('q')or f==1):
            break;
def getImages(path):
        imagePath=[os.path.join(path,f) for f in os.listdir(path)]
        faces=[]
        ids=[]
        for i in imagePath:
            faceImg=Image.open(i).convert('L');
            faceNP=np.array(faceImg,'uint8')
            id=int(os.path.split(i)[-1].split('.')[1])
            print(id)
            faces.append(faceNP)
            ids.append(id)
            cv2.waitKey(10)
        return np.array(ids),faces
def main():
     path='Data'
     imagePath=[os.path.join(path,f) for f in os.listdir(path)] 
     while(True):
         f=0
         id=int(input('enter id'))
         for i in imagePath:
            rd=int(os.path.split(i)[-1].split('.')[1])
            if id==rd:
               f=1
         if f==1:
               print("id already exist pls enter diff id")
         else:
               break;
            
         
     name=input('enter name')
     num=0
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
            recognizer=cv2.face.LBPHFaceRecognizer_create();
            ids,faces=getImages(path)
            recognizer.train(faces,ids)
            recognizer.write('recon/train.yml')
            break;
     



     

ran()

   
cam.release()
cv2.destroyAllWindows()
