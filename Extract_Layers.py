import numpy as np
import cv2
import tensorflow as tf
import numpy as np
from AlexNet import AlexNet
from Configuration import FLAGS

class Extract_Layers(object):
	def __init__(self, pra_config):
		self.config = pra_config

		batch_size = 1
		self.layer_name_list = pra_config.layer_name_list
		self.frame_resize = pra_config.frame_resize
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

		# self.bool_display = pra_config.bool_display
		# self.col_spliter_width = pra_config.col_spliter_width
		# self.row_spliter_width = pra_config.row_spliter_width
		# self.new_map_size = pra_config.new_map_size
		# self.num_cols = pra_config.num_cols

	def Get_layers(self, pra_image):
		# extract feature maps for the input image
		image = [cv2.resize(pra_image, (self.frame_resize, self.frame_resize))]
		feed = {self.images:image}
		feature_list = self.sess.run(self.feature_name_list, feed_dict=feed)

		all_map_list = []
		for feature_map in feature_list:
			feature_map = feature_map[0]
			now_map_list = np.array([cv2.resize(feature_map[:,:,x], (self.config.new_map_size, self.config.new_map_size)) for x in range(feature_map.shape[-1])])
			all_map_list.extend(now_map_list)
		all_map_list = np.array(all_map_list)
		return all_map_list 


if __name__ == '__main__':
	extract_layers = Extract_Layers(FLAGS)
	
