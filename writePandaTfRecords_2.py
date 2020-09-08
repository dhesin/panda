from tensorflow_examples.models.pix2pix import pix2pix   
import openslide
import skimage.io
import pandas as pd
import PIL
import cv2
import io
import os

# General packages
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import Image, display
from IPython.display import clear_output


import tensorflow as tf

print(tf.__version__)
tf.config.list_physical_devices('GPU')

# Location of the training images
data_dir = '/tf/kaggle/input/panda/train_images'
mask_dir = '/tf/kaggle/input/panda/train_label_masks'
output_dir = '/tf/panda/working'
checkpoint_path = "/tf/panda/working/training_3/cp.ckpt"

# Location of training labels
train_labels = pd.read_csv('/tf/panda/train.csv',
                           names=["image_id", "data_provider", "isup_grade", "gleason_score"]).set_index('image_id')


tiff_level=2
block_width=224
block_height=224
nonzeropixels=(int)(((block_width*block_height*3)/10))


@tf.function(experimental_relax_shapes=True)
def map_func_1(region_all):
  return tf.map_fn(lambda x: tf.math.count_nonzero(tf.math.not_equal(tf.cast(x,tf.int32),255)),
                   region_all, dtype=tf.int64, parallel_iterations=4)

@tf.function(experimental_relax_shapes=True)
def map_func_2(mask):
  return tf.map_fn(lambda x:tf.less(nonzeropixels,x), mask, dtype=tf.bool, parallel_iterations=4)



overlap_width=(int)(block_width/10)
overlap_height=(int)(block_height/10)

@tf.function
def get_image_blocks_overlapped(file):

    region_all = openslide.OpenSlide(file)
    width, height = region_all.level_dimensions[tiff_level]
    region_all = region_all.read_region((0,0), tiff_level, (width, height))
    region_all = np.asarray(region_all)
    region_all = region_all[:,:,0:3]
    print(region_all.shape)

    #region_all = skimage.io.MultiImage(file)
    #region_all = region_all[tiff_level]
    ##region_all = region_all.astype(np.float32)
    #(height, width, depth) = region_all.shape

    #region_all = tf.image.per_image_standardization(region_all)
    #plt.imshow(region_all)
    #plt.show()
    #print(region_all.shape)

    step_x = (block_width-overlap_width)
    step_y = (block_width-overlap_height)
    num_blocks_x=(int)((width-block_width)/step_x)+1
    num_blocks_y=(int)((height-block_height)/step_y)+1

    blocks = []
    for i in range(num_blocks_x):
        for j in range(num_blocks_y):

            block = region_all[j*step_y:j*step_y+block_height, i*step_x:i*step_x+ block_width]
            blocks.append(block)

    region_all = tf.stack(blocks)
    print(region_all.shape)


    ##mask = tf.map_fn(tf.function(map_func_1), region_all, dtype=tf.int64, parallel_iterations=4)
    mask = map_func_1(region_all)
    ##print(mask)
    ##mask = tf.map_fn(tf.function(map_func_2), mask, dtype=tf.bool, parallel_iterations=4)
    mask = tf.cast(mask, dtype=tf.int32)
    mask = map_func_2(mask)

    #print(mask)

    region_all = tf.boolean_mask(region_all, mask)
    print(region_all.shape)
    #plt.imshow(region_all[10])
    #plt.show()
    #print(region_all[10].shape)

    return region_all, mask

@tf.function
def get_mask_blocks_overlapped(file, mask_mask):

    region_all = openslide.OpenSlide(file)
    width, height = region_all.level_dimensions[tiff_level]
    #print(height, width)
    region_all = region_all.read_region((0,0), tiff_level, (width, height))
    region_all = np.asarray(region_all)
    region_all = region_all[:,:,0]
    print(region_all.shape)

    #region_all = skimage.io.MultiImage(file)
    #region_all = region_all[tiff_level]
    ##region_all = region_all.astype(np.float32)
    #(height, width, depth) = region_all.shape

    #plt.imshow(region_all)
    #plt.show()
    #print(region_all.shape)

    step_x = (block_width-overlap_width)
    step_y = (block_width-overlap_height)
    num_blocks_x=(int)((width-block_width)/step_x)+1
    num_blocks_y=(int)((height-block_height)/step_y)+1

    blocks = []
    for i in range(num_blocks_x):
        for j in range(num_blocks_y):

            block = region_all[j*step_y:j*step_y+block_height, i*step_x:i*step_x+ block_width]
            blocks.append(block)

    region_all = tf.stack(blocks)
    print(region_all.shape)


    region_all = tf.boolean_mask(region_all, mask_mask)
    print(region_all.shape)
    #plt.imshow(region_all[10])
    #plt.show()
    #print(region_all[10].shape)

    return region_all
	

image_num=0
	
for index, row in train_labels.iterrows():
        mask_file = os.path.join(mask_dir, f'{index}_mask.tiff')
        file = os.path.join(data_dir, f'{index}.tiff')

        if os.path.isfile(mask_file) and os.path.isfile(file):
            print(mask_file, row["data_provider"])
            blocks, mask_mask = get_image_blocks_overlapped(file)
            masks = get_mask_blocks_overlapped(mask_file, mask_mask)

            blocks = blocks.numpy()
            masks = masks.numpy()

            assert(blocks.shape[0] == masks.shape[0])

            #plt.figure()
            #plt.subplot(1,2,1)
            #plt.imshow(blocks[0])
            #plt.subplot(1,2,2)
            #plt.imshow(masks[0])
            #plt.show()


            dp = row['data_provider']
            dp = [bytes(elem, 'utf8') for elem in dp]
            #print(dp)

            gs = row["gleason_score"]
            gs = [bytes(elem, 'utf8') for elem in gs]
            #print(gs)

            feature = {
                'image': tf.train.Feature(int64_list=tf.train.Int64List(value=blocks.reshape(-1))),
                'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[block_height])),
                'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[block_width])),
                'depth': tf.train.Feature(int64_list=tf.train.Int64List(value=[blocks.shape[0]])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(row["isup_grade"])])),
                'provider': tf.train.Feature(bytes_list=tf.train.BytesList(value=dp)),
                'mask': tf.train.Feature(int64_list=tf.train.Int64List(value=masks.reshape(-1))),
                'gleason': tf.train.Feature(bytes_list=tf.train.BytesList(value=gs)),
            }

            tf_example = tf.train.Example(features=tf.train.Features(feature=feature))

            record_file = output_dir + f'/tfrecords/img_blocks_{image_num}.tfrecords'
            image_num = image_num+1
            with tf.io.TFRecordWriter(record_file) as writer:

                writer.write(tf_example.SerializeToString())


