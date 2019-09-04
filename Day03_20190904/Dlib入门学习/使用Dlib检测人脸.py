__author__ = "Luke Liu"
#encoding="utf-8"

# import reference modules

import dlib
import matplotlib.pyplot as plt
import glob
from  PIL import Image
import  os
import numpy as np
import  cv2

face_images_dir = "../../../dataset/lfw_new_imgs"
faces_images_path=[os.path.join(face_images_dir,filename) for filename in os.listdir(face_images_dir)]#
# 利用dlib建立一个face_detector

detector = dlib.get_frontal_face_detector()

for image in faces_images_path:
    img=Image.open(image)
    imgs=img.convert("RGB")
    imgs= np.array(imgs)
    print(type(imgs))
    # img=np.asarray(img)
    #后面这个1 代表放大了一倍，便于检测
    dets=detector(imgs,1)

    print("detect %d people's face"%(len(dets)))
    for i ,d in enumerate(dets):
        print("NO {} person, Left {}\tRight {}\tTOP {}\tBottom {}\t ".format(i,d.left(),d.right(),d.top(),d.bottom()))
        print(type(d.left()))

    imgs=cv2.cvtColor(imgs,cv2.COLOR_RGB2BGR)
    cv2.rectangle(imgs,(dets[0].left(),dets[0].top()),(dets[0].right(),dets[0].bottom()),(0,255,0),2)
    cv2.imshow("",imgs)
    cv2.waitKey()
    cv2.destroyAllWindows()
