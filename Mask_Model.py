import numpy as np
import os
import tensorflow as tf
import random
import cv2
from Configuration import FLAGS
from VOC import VOC
from Extract_Layers import Extract_Layers

class Mask_Model(object):
	def __init__(self, pra_config):
		self.config = pra_config		
		self.new_map_size = new_map_size = pra_config.new_map_size
		number_layer = 1280

		with tf.variable_scope('Input_Layer'):
			self.global_step = tf.Variable(
				initial_value = 0,
				name = 'global_step',
				trainable = False#,
				# collections = [tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.VARIABLES]
				)

			self.feature_map_list = tf.placeholder(tf.float32, [number_layer, new_map_size, new_map_size], name='Feature_Map_list')
			self.ground_truth_mask = tf.placeholder(tf.float32, [new_map_size, new_map_size], name='Ground_Truth_mask')	

		
		with tf.variable_scope('Combine_Layer'):
			# self.weights = self.get_weight(pra_shape=[number_layer,])
			self.weights = self.get_weight(pra_shape=self.feature_map_list.get_shape())
			self.combined_image = tf.zeros(shape=self.feature_map_list.get_shape()[-2:])

			print self.feature_map_list.get_shape()
			print self.weights.get_shape()

			# expand_weights = tf.expand_dims(tf.expand_dims(self.weights, -1), -1)
			expand_weights = self.weights

			weighted_map = tf.mul(self.feature_map_list, expand_weights)
			# self.combined_map = tf.sigmoid(tf.reduce_mean(weighted_map, 0))*255

			combined_map = tf.reduce_mean(weighted_map, 0) 
			combined_map = combined_map - tf.reduce_min(combined_map)
			self.combined_map = (combined_map / tf.reduce_max(combined_map)) * 255

		with tf.variable_scope('Train_Layer'):
			self.cost = tf.reduce_mean(tf.square(tf.sub(self.combined_map, self.ground_truth_mask)))
			self.train_op = tf.train.AdamOptimizer(1e-1).minimize(self.cost, global_step=self.global_step)
			# for feature_map in self.feature_map_list:
				# w = tf.Variable(tf.random_normal([1]))
				# tf.add(self.combined_image, tf.mul(w, tf.cast(feature_map, tf.float32)))
			# tf.scalar_summary(tf.constant('loss_', tf.string) + self.now_process, self.cost)
			tf.scalar_summary('loss', self.cost)

			# tf.summary.scalar('loss_training', self.cost)


	def get_weight(self, pra_shape, pra_name='Weights'):
			init = tf.random_normal_initializer(mean=0, stddev=1.)
			w = tf.get_variable(name=pra_name, shape=pra_shape, initializer=init)
			return w


class Mask_Controller(object):
	def __init__(self, pra_config):
		self.config = pra_config
		if not os.path.exists(pra_config.mask_model_folder):
			os.makedirs(pra_config.mask_model_folder)
		if not os.path.exists(pra_config.mask_summary_folder):
			os.makedirs(pra_config.mask_summary_folder)

		self.extract_layers = Extract_Layers(pra_config)


		self.graph = tf.Graph()
		self.sess = tf.Session(graph=self.graph)

		with self.sess.graph.as_default():
			self.mask_model = Mask_Model(pra_config)

		self.voc = VOC(pra_config)

	def Train(self):
		with self.sess.graph.as_default(), tf.Session() as sess:
			merged = tf.summary.merge_all()
			writer = tf.train.SummaryWriter(self.config.mask_summary_folder, sess.graph)
			model_saver = tf.train.Saver()
			init = tf.global_variables_initializer()
			sess_manager = tf.train.SessionManager()
			sess = sess_manager.prepare_session("", init_op=init, saver=model_saver, checkpoint_dir=self.config.mask_model_folder)

			max_epoch = 1000
			for epoch_index in range(max_epoch):
				# for name in self.voc.instance_name_list:				
				for image, image_segment, image_mask in self.voc.get_next_image('Aeroplane'):
					input_feature_map_list = self.extract_layers.Get_layers(image).astype(float)
					input_ground_truth_mask = cv2.resize(image_mask, (self.config.new_map_size, self.config.new_map_size)).astype(float)
					
					feed = {
						self.mask_model.feature_map_list : input_feature_map_list,
						self.mask_model.ground_truth_mask : input_ground_truth_mask
					}
					# rs, gt = sess.run([merged, self.mask_model.ground_truth_mask], feed_dict=feed)
					rs, cost, combined_map, _, global_step = sess.run([merged, self.mask_model.cost, self.mask_model.combined_map, self.mask_model.train_op, self.mask_model.global_step], feed_dict=feed)
					print 'Epoch:{}, Global_step: {}, Cost: {}'.format(epoch_index, global_step, cost)
					# print combined_map.shape
					# print 
					
				
					if epoch_index > 500:
						cv2.imshow('now', cv2.resize(combined_map.astype(np.uint8), (100, 100)))
						cv2.imshow('gt', cv2.resize(input_ground_truth_mask, (100, 100)))
						cv2.waitKey(0)
					else:
						cv2.imshow('now', combined_map.astype(np.uint8))
						cv2.imshow('gt', input_ground_truth_mask)
						cv2.waitKey(1)
					
					writer.add_summary(rs, global_step=global_step)
					


def main(_):
	# mask_model = Mask_Model(FLAGS)
	# mask_model.Train()
	mask_controller = Mask_Controller(FLAGS)
	mask_controller.Train()

if __name__ == '__main__':
	tf.app.run()
	



