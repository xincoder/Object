import os
import tensorflow as tf 

FLAGS = tf.app.flags.FLAGS



tf.app.flags.DEFINE_string('dataset_root_path', '/Users/xincoder/Documents/Dataset/VOCdevkit/VOC2007',
								'The path of image dataset')

# #################################################################################
# ##################  Configuration of pre-trained Alexnet  #######################
# 
# #################################################################################
# tf.app.flags.DEFINE_string('layer_name_list', ['conv1', 'norm1', 'pool1', 'conv2', 'norm2', 'pool2', 'conv3', 'conv4', 'conv5', 'pool5'], #'fc6', 'fc7', 'fc8', 'prob']
# 								'The names of layers in the pre-trained model')
tf.app.flags.DEFINE_string('layer_name_list', ['conv2', 'conv3', 'conv4', 'conv5'],
								'The names of layers in the pre-trained model')

tf.app.flags.DEFINE_integer('frame_resize', 227,
								'The size of the input of Alexnet')
# #################################################################################


# Configuration for cache path
tf.app.flags.DEFINE_string('cache_root_path', '../data_cache',
								'The root path of cache data')

# #################################################################################
# ######################  Configuration of Mask_Model  ############################
# 
# #################################################################################

tf.app.flags.DEFINE_string('mask_model_folder', os.path.join(FLAGS.cache_root_path, 'mask_model'),
								'Mask Model will be saved in this path')
tf.app.flags.DEFINE_string('mask_model_save_path', os.path.join(FLAGS.cache_root_path, 'mask_model_folder/my_caption_model'),
								'Mask Model name')
tf.app.flags.DEFINE_string('mask_summary_folder', os.path.join(FLAGS.cache_root_path, 'mask_log'),
								'Mask Model log will be saved in this path')

# #################################################################################



# # #################################################################################
# # ################## Configuration to Display Feature Maps  #######################
# # 
# # #################################################################################
# tf.app.flags.DEFINE_boolean('bool_display', False,
# 								'Indicate whether to Diplay the feature maps')
tf.app.flags.DEFINE_integer('new_map_size', 40,
								'The new size of feature map')
# tf.app.flags.DEFINE_integer('num_cols', 24.,
# 								'The number of columns in each row')
# tf.app.flags.DEFINE_integer('col_spliter_width', 10,
# 								'The width of spliter used to split columns')
# tf.app.flags.DEFINE_integer('row_spliter_width', 10,
# 								'The width of spliter used to split rows')
# # #################################################################################




