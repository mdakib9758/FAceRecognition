import os
import cv2
import numpy as np
from PIL import Image

recognizer=cv2.face.LBPHFaceRecognizer_create();
path='Data'

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
        cv2.imshow("training",faceNP)
        cv2.waitKey(10)
    return np.array(ids),faces


ids,faces=getImages(path)
recognizer.train(faces,ids)
recognizer.write('recon/train.yml')
cv2.destroyAllWindows()
