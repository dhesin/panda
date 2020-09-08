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

train_labels = pd.read_csv('/tf/panda/train.csv',
                           names=["image_id", "data_provider", "isup_grade", "gleason_score"]).set_index('image_id')

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

block_width=96
block_height=96
depth=3

def filter_tfrecord_2(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    #print(example_proto)
    image_features = tf.io.parse_example(example_proto, image_mask_feature_description)

    num_blocks = image_features['depth']
    if (num_blocks>2000):
        return False
    return True

def filter_tfrecord(example_proto):
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

    num_nz_pixels=(int)((block_width*block_height*depth*9)/10)
    num_nz_pixels_mask=(int)((block_width*block_height))


    gs = tf.sparse.to_dense(gs)
    provider = tf.sparse.to_dense(provider)
    #print('gleason :', gs)
    #print('provider :', provider)

    #print(data.shape)
    blocks = tf.sparse.to_dense(blocks)#.numpy().astype(dtype=np.float32)
    blocks = tf.reshape(blocks, (num_blocks, block_height, block_width, 3))
    blocks = tf.cast(blocks, tf.int32)

    masks = tf.sparse.to_dense(masks)
    masks = tf.reshape(masks, (num_blocks, block_height, block_width))
    masks = tf.cast(masks, tf.int32)


    nz_counts = tf.math.count_nonzero(tf.math.not_equal(blocks,255), [1,2,3])
    nz_mask = tf.map_fn(lambda x:tf.math.less_equal(num_nz_pixels,x), nz_counts, dtype=(tf.bool))
    blocks = tf.boolean_mask(blocks, nz_mask)
    masks = tf.boolean_mask(masks, nz_mask)
    #print(blocks.shape)
    #print(masks.shape)

    nz_counts = tf.math.count_nonzero(tf.math.not_equal(masks,0), [1,2])
    nz_mask = tf.map_fn(lambda x:tf.math.less_equal(num_nz_pixels_mask,x), nz_counts, dtype=(tf.bool))
    masks = tf.boolean_mask(masks, nz_mask)  
    masks=None
    blocks = tf.boolean_mask(blocks, nz_mask)
    print(blocks.shape)
    #print(masks.shape)

    if (blocks.shape[0] > 1000):
        blocks = tf.slice(blocks,[0,0,0,0],[1000,block_height,block_width,3])

    blocks = tf.image.resize(blocks, (96, 96), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, 
                             preserve_aspect_ratio=True, antialias=True)   
    blocks = tf.cast(blocks, tf.float32)
    blocks = tf.image.per_image_standardization(blocks) 

    blocks = tf.cond(tf.greater(tf.random.uniform([]), 0.75), lambda: tf.image.flip_up_down(blocks), lambda: blocks)  
    blocks = tf.cond(tf.greater(tf.random.uniform([]), 0.75), lambda: tf.image.flip_left_right(blocks), lambda: blocks)  
    blocks = features(blocks)
    blocks = tf.reshape(blocks,(-1,9*1280))

    if blocks.shape[0] < 1000:
        paddings = tf.constant([[0,1000-blocks.shape[0]], [0,0]])
        blocks = tf.pad(blocks, paddings, constant_values=1.)


    #masks = tf.tile(tf.expand_dims(masks,axis=3),(1,1,1,3))
    #masks = tf.image.resize(masks, (96, 96), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, 
    #                         preserve_aspect_ratio=True, antialias=True)  
    #masks[:,:,:,0]
    #print(masks.shape)

    #plt.figure()
    #plt.subplot(1,2,1)
    #plt.imshow(blocks[0])
    #plt.subplot(1,2,2)
    #plt.imshow(masks[0,:,:,0])
    #plt.show()

    return blocks, label

    
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    0.0001,
    decay_steps=10,
    decay_rate=0.96,
    staircase=True)

optimizer = tf.keras.optimizers.Adam(lr_schedule)   
features = tf.keras.applications.MobileNetV2(input_shape=[96, 96, 3], include_top=False)
features.trainable=False

listOfFiles = os.listdir("/tf/kaggle/tf_records")
pattern = "*.tfrecords"
listOfFiles = ["/tf/kaggle/tf_records/"+x for x in listOfFiles if fnmatch.fnmatch(x,pattern)]
#print(listOfFiles)

dataset = tf.data.TFRecordDataset(listOfFiles).filter(filter_tfrecord_2).prefetch(tf.data.experimental.AUTOTUNE)
dataset=dataset.map(lambda x1: tf.py_function(func=filter_tfrecord , inp=[x1], Tout=[tf.float32, tf.int64]), num_parallel_calls=tf.data.experimental.AUTOTUNE)
#print(dataset)
dataset=dataset.batch(2).repeat()

model = tf.keras.models.Sequential([
    tf.keras.layers.Attention(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, input_shape=((9*1280),))])

Q = tf.Variable(tf.random.normal([10,9*1280]), trainable=True)

checkpoint_directory = "/tf/panda/chkpts/"
checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model, variable=Q)
status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))


writer = tf.summary.create_file_writer("/tf/panda/tb_logs/")


epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

for blocks, labels in dataset:

    with tf.GradientTape() as tape:
        print(blocks.shape)
        logits = model([Q,blocks])
        print(labels.numpy())
        loss_value = loss(labels, logits)
        print(loss_value.numpy())
        variables = model.trainable_variables
        variables.append(Q)
        gradients = tape.gradient(loss_value, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        lr = optimizer.learning_rate(optimizer.iterations)
        print(optimizer.iterations.numpy(), lr.numpy())
        epoch_accuracy.update_state(labels, logits)
        print(epoch_accuracy.result().numpy())
        if optimizer.iterations.numpy()%25==1:
            checkpoint.save(file_prefix=checkpoint_prefix)

        with writer.as_default():
            tf.summary.scalar("loss", loss_value, step=optimizer.iterations)
            tf.summary.scalar("accuracy", epoch_accuracy.result(), step=optimizer.iterations)
            writer.flush()

