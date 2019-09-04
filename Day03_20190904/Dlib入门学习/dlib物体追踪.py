__author__ = "Luke Liu"
#encoding="utf-8"
# -*- coding: utf-8 -*-
# with the help of labelimg

import dlib
from imageio import imread
import glob
import cv2
import numpy as np

cap = cv2.VideoCapture('../../../dataset/sequences_image/bottle.mp4')

fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧速率
# 创建一个tracker
tracker = dlib.correlation_tracker()
path = "../../../dataset/sequences_image/0.png"
img = imread(path)

# 捕捉最开始的矩形区域最为目标，适合跟踪形态几乎不变的物体
tracker.start_track(img, dlib.rectangle(133, 153, 330, 734))

while cap.isOpened():
	ret, image_np = cap.read()
	if len((np.array(image_np)).shape) == 0:
		break
	image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
	tracker.update(image_np)

	plt1=(int(tracker.get_position().left()),int(tracker.get_position().top()))
	plt2=(int(tracker.get_position().right()),int(tracker.get_position().bottom()))

	cv2.rectangle(image_np,plt1,plt2,(0,255,0),2)
	cv2.imshow("",image_np)
	cv2.waitKey(int(1000/fps))

cap.release()
cv2.destroyAllWindows()



# #定位跟踪某个矩形框的位置，之后能够自动更新与定位这个矩形框的位置
#
#
# paths = sorted(glob.glob("C:/Users/asus/Desktop/sequences_image/*.png"))

# # 指定追踪的物体的
# for i, path in enumerate(paths):
# 	img = imread(path)
# 	# 第一帧，指定一个区域
# 	if i == 0:
# 		tracker.start_track(img, dlib.rectangle(133, 153, 330, 734))
# 	# 后续帧，自动追踪
# 	else:
# 		tracker.update(img)
