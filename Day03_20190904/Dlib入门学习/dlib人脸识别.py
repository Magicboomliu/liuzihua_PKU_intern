__author__ = "Luke Liu"
#encoding="utf-8"

import dlib
from imageio import imread
import glob
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor_path = 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)
face_rec_model_path = 'dlib_face_recognition_resnet_model_v1.dat'
facerec = dlib.face_recognition_model_v1(face_rec_model_path)
labeled = glob.glob('labeled/*.jpg')

labeled_data = {}
unlabeled = glob.glob('unlabeled/*.jpg')

#定义一下距离函数
def distance(a, b):
	# d = 0
	# for i in range(len(a)):
	# 	d += (a[i] - b[i]) * (a[i] - b[i])
	# return np.sqrt(d)
	return np.linalg.norm(np.array(a) - np.array(b), ord=2)

# 读取标注图片并保存对应的128向量
for path in labeled:
	img = imread(path)
	name = path.split('/')[1].rstrip('.jpg')
	dets = detector(img, 1)
	# 这里假设每张图只有一个人脸
	shape = predictor(img, dets[0])
	face_vector = facerec.compute_face_descriptor(img, shape)
	labeled_data[name] = face_vector

# 读取未标注图片，并和标注图片进行对比
for path in unlabeled:
	img = imread(path)
	name = path.split('/')[1].rstrip('.jpg')
	dets = detector(img, 1)
	# 这里假设每张图只有一个人脸
	shape = predictor(img, dets[0])
	face_vector = facerec.compute_face_descriptor(img, shape)
	matches = []
	for key, value in labeled_data.items():
		d = distance(face_vector, value)
		if d < 0.6:
			matches.append(key + ' %.2f' % d)

	print('{}：{}'.format(name, ';'.join(matches)))