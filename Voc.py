import glob
import cv2
import os
import numpy as np

class VOC(object):
	def __init__(self):
		self.root_path = '/Users/xincoder/Documents/Dataset/VOCdevkit/VOC2007'
		self.segmentation_class_path_list = glob.glob(os.path.join(self.root_path, 'SegmentationClass/*.png'))
		self.image_path_list = [x.replace('SegmentationClass', 'JPEGImages').replace('png', 'jpg') for x in self.segmentation_class_path_list]
		self.instance_name_list = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle', 'Bus', 'Car', 
							'Cat', 'Chair', 'Cow', 'Diningtable', 'Dog', 'Horse', 'Motorbike', 
							'Person', 'Pottedplant', 'Sheep', 'Sofa', 'Train', 'Tvmonitor']
		self.color_list = {
			'Aeroplane' : np.array([0, 0, 128], dtype=np.uint8), # 0
			'Bicycle' : np.array([0, 128, 0], dtype=np.uint8), # 1
			'Bird' : np.array([0, 128, 128], dtype=np.uint8), # 2
			'Boat' : np.array([128, 0, 0], dtype=np.uint8), # 3
			'Bottle' : np.array([128, 0, 128], dtype=np.uint8), # 4 
			'Bus' : np.array([128, 128, 0], dtype=np.uint8), # 5
			'Car' : np.array([128, 128, 128], dtype=np.uint8), # 6 
			'Cat' : np.array([0, 0, 64], dtype=np.uint8), # 7
			'Chair' : np.array([0, 0, 192], dtype=np.uint8), # 8 
			'Cow' : np.array([0, 128, 64], dtype=np.uint8), # 9
			'Diningtable' : np.array([0, 128, 192], dtype=np.uint8), # 10 
			'Dog' : np.array([128, 0, 64], dtype=np.uint8), # 11
			'Horse' : np.array([128, 0, 192], dtype=np.uint8), # 12 
			'Motorbike' : np.array([128, 128, 64], dtype=np.uint8), # 13
			'Person' : np.array([127, 127, 191], dtype=np.uint8), # 14
			'Pottedplant' : np.array([0, 64, 0], dtype=np.uint8), # 15
			'Sheep' : np.array([0, 64, 128], dtype=np.uint8), # 16
			'Sofa' : np.array([0, 192, 0], dtype=np.uint8), # 17
			'Train' : np.array([0, 192, 128], dtype=np.uint8), # 18
			'Tvmonitor': np.array([128, 64, 0], dtype=np.uint8)} # 19


	def get_next_image(self, pra_instance_index):
		self.now_instance_index = pra_instance_index
		self.now_instance_name = self.instance_name_list[self.now_instance_index]
		print self.now_instance_name

		for image_path, segmentation_class_path in zip(self.image_path_list, self.segmentation_class_path_list):
			image = cv2.imread(image_path)
			image_segment = cv2.imread(segmentation_class_path)

			# cv2.imshow('img', image)
			# cv2.imshow('seg', image_segment)
			
			now_color = self.color_list[self.now_instance_name]
			image_mask = cv2.inRange(image_segment, now_color, now_color+1)

			# cv2.imshow(self.now_instance_name, cv2.resize(image_mask, (250, 250)))
			# for instance_name in self.instance_name_list:
			# 	print instance_name 
			# 	now_color = self.color_list[instance_name]
			# 	image_mask = cv2.inRange(image_segment, now_color, now_color+1)
			# 	cv2.imshow(instance_name, cv2.resize(image_mask, (50, 50)))
			# cv2.waitKey(1)
			if np.max(image_mask)>0:
				yield image, image_segment, image_mask


if __name__ == '__main__':
	voc = VOC()
	num_name_list = len(voc.instance_name_list)
	for ind in range(num_name_list):
		for image, image_segment, image_mask in voc.get_next_image(ind):
			cv2.imshow('img', image)
			cv2.imshow('seg', image_segment)
			cv2.imshow('mask', cv2.resize(image_mask, (50, 50)))
			cv2.waitKey(1)
			