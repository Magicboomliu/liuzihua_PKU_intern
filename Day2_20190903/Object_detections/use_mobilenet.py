__author__ = "Luke Liu"
#encoding="utf-8"
__author__ = "Luke Liu"
#encoding="utf-8"
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from utils import label_map_util
from utils import visualization_utils as vis_util
import os
# mobilenet的模型位置
path="D:/BaiduYunDownload/python_exe/dataset/object_detections/ssd_mobilenet_v1_coco_2018_01_28"
PATH_TO_CKPT = os.path.join(path,'frozen_inference_graph.pb')

PATH_TO_LABELS ='mscoco_label_map.pbtxt'
# 一共有90个class
NUM_CLASSES = 90

# 加载模型
# 首先建立一个detect_graph
detection_graph = tf.Graph()
#调用这个默认的图
with detection_graph.as_default():

	od_graph_def = tf.GraphDef()
	with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
#解析这个图
		od_graph_def.ParseFromString(fid.read())
		tf.import_graph_def(od_graph_def, name='')

# 加载数据的标签，load labelmap
#将label_map转化为categories,最后转化为idx
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

#将数组
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


TEST_IMAGE_PATHS = ['test_images/nba_allstars.jpg']

with detection_graph.as_default():
	with tf.Session(graph=detection_graph) as sess:
		image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
		detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
		detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
		detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
		num_detections = detection_graph.get_tensor_by_name('num_detections:0')
		for image_path in TEST_IMAGE_PATHS:
			image = Image.open(image_path)
			image_np = load_image_into_numpy_array(image)
			#增加一维度的image
			image_np_expanded = np.expand_dims(image_np, axis=0)
			# 输出image 的output结果
			(boxes, scores, classes, num) = sess.run(
				[detection_boxes, detection_scores, detection_classes, num_detections],
				feed_dict={image_tensor: image_np_expanded})
			#np.squeeze主要是去掉shape为1维度，这里主要输出第一个维度
			vis_util.visualize_boxes_and_labels_on_image_array(image_np, np.squeeze(boxes),
															   np.squeeze(classes).astype(np.int32), np.squeeze(scores),
															   category_index, use_normalized_coordinates=True,
															   line_thickness=8)
			plt.figure(figsize=[12, 8])
			plt.imshow(image_np)
			plt.show()