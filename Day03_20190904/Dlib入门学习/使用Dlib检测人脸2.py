__author__ = "Luke Liu"
#encoding="utf-8"
import dlib
import cv2
import os

detector = dlib.get_frontal_face_detector()
font=cv2.FONT_HERSHEY_SIMPLEX
cap = cv2.VideoCapture(0)

while True:
    ret,image=cap.read()
    if ret:
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        dets,scores,idx=detector.run(gray,1,-1)

        # 便利标记所有的人脸的信心
        faces=[]
        for i,d in enumerate(dets):
            plts1 = (d.left(),d.top())
            plts2 =(d.right(),d.bottom())
            x1=int((d.left()+d.right())*0.5)
            y=d.top()-5
            cv2.rectangle(image,plts1,plts2,(0,255,0),2)
            imgzi = cv2.putText(image, '{}'.format(scores[i]), (x1,y), font, 1.2, (255, 255, 255), 2)
            faces_info = (plts1,plts2)
            faces.append(faces_info)

        cv2.imshow("",image)

        if cv2.waitKey(25) & 0xFF == 27:
            cap.release()
            cv2.destroyAllWindows()
            break



