__author__ = "Luke Liu"
#encoding="utf-8"
import dlib
import matplotlib.pyplot as plt
import glob
from  PIL import Image
import  os
import numpy as np
import  cv2

face_images_dir = "../../../dataset/dlib"

detector = dlib.get_frontal_face_detector()
predictor_path =os.path.join(face_images_dir,'shape_predictor_68_face_landmarks.dat')
predictor = dlib.shape_predictor(predictor_path)

cap = cv2.VideoCapture(0)

while True:
    ret,image = cap.read()
    if ret:
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        dets=detector(gray,1)
        for i,d in enumerate(dets):

            landmarks = np.matrix([[p.x, p.y] for p in predictor(gray, dets[i]).parts()])
            for idx, point in enumerate(landmarks):
                # 68点的坐标
                pos = (point[0, 0], point[0, 1])
                # 利用cv2.circle给每个特征点画一个圈，共68个
                cv2.circle(image, pos, 5, color=(0, 255, 0))
                # 利用cv2.putText输出1-68
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(image, str(idx), pos, font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(image,(d.left(),d.top()),(d.bottom(),d.right()),(255,0,0),2)
        cv2.imshow("",image)

        if cv2.waitKey(25)& 0xFF == 27:
            cap.release()
            cv2.destroyAllWindows()
            break

