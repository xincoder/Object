import numpy as np 
import tensorflow as tf
import timeit
import cv2
from AlexNet import AlexNet
from tensorflow.python.platform import gfile
import glob
import os

def Layout_map(pra_map_list):
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
	new_map_size = 30
	now_map_list = pra_map_list.astype(np.uint8)
	now_map_list = np.array([cv2.resize(now_map_list[:,:,x], (new_map_size, new_map_size)) for x in range(now_map_list.shape[-1])])
	
	# generate empty drawable panel 
	layout_rows = num_rows*new_map_size + (num_rows-1)*row_splitter_width
	layout_cols = num_cols*new_map_size + (num_cols-1)*col_splitter_width
	now_layout = np.zeros((layout_rows, layout_cols), dtype=np.uint8) + 100

	# draw maps on the drawable panel
	for index, now_map in enumerate(now_map_list):
		row_ind = index//num_cols
		col_ind = index - row_ind*num_cols
		start_row = row_ind * (new_map_size+row_splitter_width)
		start_col = col_ind * (new_map_size+col_splitter_width)
		now_layout[start_row:start_row+new_map_size, start_col:start_col+new_map_size] = now_map

	return now_layout


def Display_layers(pra_feature_map):
	'''
	Display all layers 
		Input: feature maps belong to all layers
	'''
	for feature_map in pra_feature_map:
		now_layout = Layout_map(feature_map[0])
		cv2.imshow('layout', now_layout)
		cv2.waitKey(0)

	
def Get_layers(pra_image_paths):
	batch_size = 1
	layer_name_list = ['conv1', 'norm1', 'pool1', 'conv2', 'norm2', 'pool2', 'conv3', 'conv4', 'conv5', 'pool5']#, 'fc6', 'fc7', 'fc8', 'prob']

	with tf.Session() as sess:
		print 'Loading Model...'
		images = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
		net = AlexNet({'data': images})
		feature_name_list = []
		for layer_name in layer_name_list:
			feature_name_list.append(net.layers[layer_name])
		sess.run(tf.global_variables_initializer())
		net.load('AlexNet.npy', sess)

		print 'Model has been successfully loaded.'

		for image_path in pra_image_paths:
			print image_path
			image = [cv2.resize(cv2.imread(image_path), (227,227))]
			cv2.imshow('ori', image[0])
			feed = {images:image}
			
			feature_list = sess.run(feature_name_list, feed_dict=feed)

			# print layer_name_list[0], feature_list[0]
			Display_layers(feature_list)


if __name__ == '__main__':
	image_root = 'image'
	image_path_list = glob.glob(os.path.join(image_root, '*/*.jpg'))
	print 'In total: {} images'.format(len(image_path_list))
	Get_layers(image_path_list)

	
