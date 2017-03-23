import numpy as np 
import tensorflow as tf
import timeit
import cv2
from AlexNet import AlexNet
from tensorflow.python.platform import gfile
import glob
import os
from Voc import VOC

class Display_Layers(object):
	def __init__(self):
		batch_size = 1
		self.layer_name_list = ['conv1', 'norm1', 'pool1', 'conv2', 'norm2', 'pool2', 'conv3', 'conv4', 'conv5', 'pool5']#, 'fc6', 'fc7', 'fc8', 'prob']
		self.frame_resize = 227
		self.sess = tf.Session()
		print 'Loading Model...'
		self.images = tf.placeholder(tf.float32, [batch_size, self.frame_resize, self.frame_resize, 3])
		net = AlexNet({'data': self.images})
		self.feature_name_list = []
		for layer_name in self.layer_name_list:
			self.feature_name_list.append(net.layers[layer_name])
		self.sess.run(tf.global_variables_initializer())
		net.load('AlexNet.npy', self.sess)
		print 'Model has been successfully loaded.'

		self.bool_display = False
		self.col_splitter_width = 10
		self.row_splitter_width = 10
		self.new_map_size = 40
		self.num_cols = 24.

		self.my_voc = VOC()


	def Get_layers(self, use_video=False):
		# extract feature maps for each image
		# for image_path in pra_image_paths:
		# 	print image_path
		# 	# image = [cv2.resize(cv2.imread(image_path), (self.frame_resize, self.frame_resize))]
		# 	image = cv2.imread(image_path)
		if use_video:
			print 'video capture'
			video_capture = cv2.VideoCapture(0)
			while(True):
				ret, frame = video_capture.read()
				if not ret:
					break
				# frame = cv2.resize(frame, (frame.shape[1]/2, frame.shape[0]/2))
				# frame = cv2.flip(frame, 1) # a frame from camera is reversed left to right
				# cv2.imshow('frame', frame)
				# cv2.waitKey(1)

				self.ori_image_size = frame.shape[1::-1]
				frame = cv2.resize(frame, (self.frame_resize, self.frame_resize))
				frame = cv2.flip(frame, 1) # a frame from camera is reversed left to right
				image = frame.copy()
				cv2.imshow('frame', frame)
				cv2.waitKey(1)

				feed = {self.images:[frame.copy()]}
				feature_list = self.sess.run(self.feature_name_list, feed_dict=feed)
				# display all layers
				combined_image = self.Combine_layers(feature_list)
				cv2.imshow('combined', combined_image)
				cv2.waitKey(1)

		else:
			for image, image_segment, image_mask in self.my_voc.get_next_image('Person'):
				self.ori_image_size = image.shape[1::-1]
				image = [cv2.resize(image, (self.frame_resize, self.frame_resize))]
				cv2.imshow('ori', image[0])
				cv2.imshow('GT_mask', cv2.resize(image_mask, (self.frame_resize, self.frame_resize)))
				feed = {self.images:image}
				feature_list = self.sess.run(self.feature_name_list, feed_dict=feed)
				# display all layers
				combined_image = self.Combine_layers(feature_list)
				cv2.imshow('combined', combined_image)
				cv2.waitKey(1)




	def Combine_layers(self, pra_feature_map):
		'''
		Display all layers 
			Input: feature maps belong to all layers
		'''
		all_map_list = []
		for index, feature_map in enumerate(pra_feature_map):
			feature_map = feature_map[0]
			now_map_list = np.array([cv2.resize(feature_map[:,:,x], (self.new_map_size, self.new_map_size)) for x in range(feature_map.shape[-1])])
			all_map_list.extend(now_map_list)
			
			# Draw and display feature map
			if self.bool_display:
				now_layout = self.Layout_map(now_map_list)
				# print 'Layer: {}'.format(self.layer_name_list[index])
				cv2.imshow(self.layer_name_list[index], now_layout)
				cv2.waitKey(1)


		all_map_list = np.array(all_map_list)
		# print all_map_list.shape
		# return all_map_list

		weight_list = np.random.random(len(all_map_list))
		# print weight_list
		weighted_map = np.array([x*y for x,y in zip(all_map_list.astype(float), weight_list)])
		combined_map = np.sum(weighted_map, axis=0)
		combined_map *= 255.0/combined_map.max()
		combined_map = cv2.resize(combined_map.astype(np.uint8), (self.frame_resize, self.frame_resize))
		# # print combined_map.min(), combined_map.max(), combined_map.shape
		# cv2.imshow('combined', cv2.resize(combined_map, (self.frame_resize, self.frame_resize)))
		# # cv2.imshow('combined', cv2.resize(combined_map, self.ori_image_size))
		# cv2.waitKey(0)
		return combined_map


	def Layout_map(self, pra_map_list):
		'''
		Draw all maps belong to the same layer on one panel
			Input: pra_map_list is the list of feature maps 
			Output: return an image with all maps
		'''
		num_map = pra_map_list.shape[0]
		num_rows = int(np.ceil(num_map/self.num_cols))
		now_map_list = pra_map_list.astype(np.uint8)
		
		# generate empty drawable panel 
		layout_rows = int(num_rows*self.new_map_size + (num_rows-1)*self.row_splitter_width)
		layout_cols = int(self.num_cols*self.new_map_size + (self.num_cols-1)*self.col_splitter_width)
		now_layout = np.zeros((layout_rows, layout_cols), dtype=np.uint8) + 100

		# draw maps on the drawable panel
		for index, now_map in enumerate(now_map_list):
			row_ind = index//self.num_cols
			col_ind = index - row_ind*self.num_cols
			start_row = int(row_ind * (self.new_map_size+self.row_splitter_width))
			start_col = int(col_ind * (self.new_map_size+self.col_splitter_width))
			now_layout[start_row:start_row+self.new_map_size, start_col:start_col+self.new_map_size] = now_map

		return now_layout


if __name__ == '__main__':
	image_root = 'image'
	image_path_list = glob.glob(os.path.join(image_root, '*/*.jpg'))
	print 'In total: {} images'.format(len(image_path_list))
	
	my_display_layers = Display_Layers()
	# my_display_layers.Get_layers(use_video=True)
	my_display_layers.Get_layers()

	
