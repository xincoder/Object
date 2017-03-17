import numpy as np 
import tensorflow as tf
import timeit
import cv2
from AlexNet import AlexNet
from tensorflow.python.platform import gfile
import glob
import os

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


	def Get_layers(self, pra_image_paths=[]):
		if len(pra_image_paths)>0:
			# extract feature maps for each image
			for image_path in pra_image_paths:
				print image_path
				image = [cv2.resize(cv2.imread(image_path), (self.frame_resize, self.frame_resize))]
				cv2.imshow('ori', image[0])
				feed = {self.images:image}
				feature_list = self.sess.run(self.feature_name_list, feed_dict=feed)

				# display all layers
				self.Display_layers(feature_list)
		else:
			print 'video capture'
			video_capture = cv2.VideoCapture(0)
			while(True):
				ret, frame = video_capture.read()
				if not ret:
					break
				frame = cv2.resize(frame, (self.frame_resize, self.frame_resize))
				cv2.imshow('frame', frame)
				cv2.waitKey(1)
				feed = {self.images:[frame.copy()]}
				feature_list = self.sess.run(self.feature_name_list, feed_dict=feed)
				# display all layers
				self.Display_layers(feature_list)


	def Display_layers(self, pra_feature_map):
		'''
		Display all layers 
			Input: feature maps belong to all layers
		'''
		for index, feature_map in enumerate(pra_feature_map[:4]):
			now_layout = self.Layout_map(feature_map[0])
			# print 'Layer: {}'.format(self.layer_name_list[index])
			cv2.imshow(self.layer_name_list[index], now_layout)
			cv2.waitKey(1)

	def Layout_map(self, pra_map_list):
		'''
		Draw all maps belong to the same layer on one panel
			Input: pra_map_list is the list of feature maps 
			Output: return an image with all maps
		'''
		num_map = pra_map_list.shape[-1]
		num_cols = 24.
		num_rows = int(np.ceil(num_map/num_cols))
		col_splitter_width = 10
		row_splitter_width = 10
		new_map_size = 40
		now_map_list = pra_map_list.astype(np.uint8)
		now_map_list = np.array([cv2.resize(now_map_list[:,:,x], (new_map_size, new_map_size)) for x in range(now_map_list.shape[-1])])
		
		# generate empty drawable panel 
		layout_rows = int(num_rows*new_map_size + (num_rows-1)*row_splitter_width)
		layout_cols = int(num_cols*new_map_size + (num_cols-1)*col_splitter_width)
		now_layout = np.zeros((layout_rows, layout_cols), dtype=np.uint8) + 100

		# draw maps on the drawable panel
		for index, now_map in enumerate(now_map_list):
			row_ind = index//num_cols
			col_ind = index - row_ind*num_cols
			start_row = int(row_ind * (new_map_size+row_splitter_width))
			start_col = int(col_ind * (new_map_size+col_splitter_width))
			now_layout[start_row:start_row+new_map_size, start_col:start_col+new_map_size] = now_map

		return now_layout


if __name__ == '__main__':
	image_root = 'image'
	image_path_list = glob.glob(os.path.join(image_root, '*/*.jpg'))
	print 'In total: {} images'.format(len(image_path_list))
	my_display_layers = Display_Layers()
	# my_display_layers.Get_layers(image_path_list)
	my_display_layers.Get_layers()

	
