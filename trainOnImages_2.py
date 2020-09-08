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
import fnmatch

import tensorflow as tf

print(tf.__version__)
tf.config.list_physical_devices('GPU')

# Location of the training images
data_dir = '/tf/kaggle/input/panda/train_images'
output_dir = '/tf/kaggle/working'
checkpoint_path_block_model = "/tf/panda/working/cp_block.ckpt"

# Location of training labels
train_labels = pd.read_csv('/tf/panda/train.csv', 
                           names=["image_id", "data_provider", "isup_grade", "gleason_score"])#.set_index('image_id')

TIMESTEPS=80
BATCH_SIZE=4
EPOCHS=100

tiff_level=0
block_width=224
block_height=224
filter_size=7

overlap_width=(int)(block_width/10)
overlap_height=(int)(block_height/10)
nonzeropixels=(int)(block_width*block_height)

image_mask_feature_description = {
    'image': tf.io.VarLenFeature(tf.int64),
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'depth': tf.io.FixedLenFeature([], tf.int64),
    'label': tf.io.FixedLenFeature([], tf.int64),
    'provider': tf.io.VarLenFeature(tf.string),
    'mask': tf.io.VarLenFeature(tf.int64),
    'gleason': tf.io.VarLenFeature(tf.string),
}

block_width=224
block_height=224
max_num_blocks=100
features = tf.keras.applications.MobileNetV2(input_shape=[block_height, block_width, 3], include_top=False)
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    0.005,
    decay_steps=10,
    decay_rate=0.96,
    staircase=True)

optimizer = tf.keras.optimizers.Adam(lr_schedule)


def parse_tfrecord(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    #print(example_proto)
    image_features = tf.io.parse_example(example_proto, image_mask_feature_description)
        
    blocks = image_features['image']
    masks = image_features['mask']
    block_width = image_features['width'] 
    block_height = image_features['height']
    num_blocks = image_features['depth']
    label = image_features['label']
    gs = image_features['gleason']
    provider = image_features['provider']
    print("num_blocks: ", num_blocks)
    
    gs = tf.sparse.to_dense(gs)
    provider = tf.sparse.to_dense(provider)
    #print('gleason :', gs)
    #print('provider :', provider)
    
    #print(data.shape)
    blocks = tf.sparse.to_dense(blocks)#.numpy().astype(dtype=np.float32)
    blocks = tf.reshape(blocks, (num_blocks, block_height, block_width, 3))
    blocks = tf.cast(blocks, tf.float32)
    #blocks = np.reshape(blocks, (num_blocks, block_height, block_width, 3))
    if (num_blocks > max_num_blocks):
        blocks = blocks[0:max_num_blocks,:,:,:]
    blocks = tf.image.per_image_standardization(blocks)  
    blocks = features(blocks)#.numpy()
    
    #mask = tf.io.decode_image(mask)
    #print(mask_data.shape)
    masks = tf.sparse.to_dense(masks)#.numpy()
    #masks = tf.cast(blocks, tf.int32)
    #masks = np.reshape(masks, (num_blocks, block_height, block_width))
    masks = tf.reshape(masks, (num_blocks, block_height, block_width))
    masks = tf.cast(masks, tf.int32)
    if (num_blocks > max_num_blocks):
        masks = masks[0:max_num_blocks,:,:]
    
    #blocks = blocks.numpy()
    #masks = masks.numpy()

    #plt.figure()
    #plt.subplot(1,2,1)
    #plt.imshow(blocks[0])
    #plt.subplot(1,2,2)
    #plt.imshow(masks[0])
    #plt.show()

    return blocks, masks, label
    


listOfFiles = os.listdir("/tf/kaggle/tf_records")
pattern = "*.tfrecords"
listOfFiles = ["/tf/kaggle/tf_records/"+x for x in listOfFiles if fnmatch.fnmatch(x,pattern)]
print(listOfFiles)

dataset = tf.data.TFRecordDataset(listOfFiles)
print(dataset)

#train=dataset.map(parse_tfrecord)
train=dataset.map(lambda x1: tf.py_function(func = parse_tfrecord , inp=[x1], Tout=[tf.float32, tf.int32, tf.int64]))
train=train.batch(4)
print(train)   

layer_1 = tf.keras.layers.Conv2D(6, (7, 7), activation='relu', input_shape=(7, 7, 1280))

layer_12 = tf.keras.layers.Conv2D(6, (1, 1), activation='relu', input_shape=(1, 1, 6))
layer_2 = tf.keras.layers.Conv3D(6, (max_num_blocks, 1, 1), activation='relu', input_shape=(7, 7, 1280))

    


for blocks, masks, labels in train:
    
    print(blocks.shape)
    with tf.GradientTape() as tape:
        
        layer_1_out=[]
        for i in range(blocks.shape[0]):
            print(blocks[i].shape)
            x = layer_1(blocks[i])
            #print(x.shape)
            x = tf.keras.layers.maximum(tf.split(x,x.shape[0], axis=0))
            #print(x.shape)
            x = layer_12(x)
            #print(x.shape)
            x = tf.squeeze(x)
            layer_1_out.append(x)
                    
        print(layer_1_out[0].shape, len(layer_1_out))
        logits = tf.stack(layer_1_out, axis=0)
        print(logits.shape)

        loss_value = loss(labels, logits)
        print(loss_value.numpy())
        variables = layer_1.trainable_variables
        #print(variables)
        gradients = tape.gradient(loss_value, variables)
        optimizer.apply_gradients(zip(gradients, variables))
