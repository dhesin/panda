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
output_dir = '/tf/panda/working'

# Location of training labels
train_labels = pd.read_csv('/tf/panda/train.csv',
                           names=["image_id", "data_provider", "isup_grade", "gleason_score"]).set_index('image_id')

checkpoint_path_block_model = "/tf/panda/working/cp_block.ckpt"
checkpoint_path_sequence_model = "/tf/panda/working/cp_sequence.ckpt"

TIMESTEPS=80
BATCH_SIZE=8
EPOCHS=100

tiff_level=1
block_width=224
block_height=224
filter_size=7

overlap_width=(int)(block_width/10)
overlap_height=(int)(block_height/10)
nonzeropixels=(int)((block_width*block_height*3)/2)


def build_model_block_processing():

    model = tf.keras.models.Sequential([
        #tf.keras.layers.LayerNormalization(input_shape=[block_height, block_width, 3]),
        tf.keras.applications.MobileNetV2(input_shape=[block_height, block_width, 3], include_top=False),
    ])
    return model

#block_model = build_model_block_processing()
#block_model.load_weights(checkpoint_path_block_model)
#block_model.trainable = False

def build_model_sequence():

    model = tf.keras.models.Sequential([
        #tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(600, filter_size, activation='relu', input_shape=(filter_size,filter_size,1280)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(300, 1, activation='relu', input_shape=(1,1,600)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(150, 1, activation='relu', input_shape=(1,1,300)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(20, 1, activation='relu', input_shape=(1,1,150)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(6),
    ])
    return model

sequence_model = build_model_sequence()
sequence_model.load_weights(checkpoint_path_sequence_model)
sequence_model.trainable = True

tf.keras.utils.plot_model(sequence_model)
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    0.0002,
    decay_steps=50,
    decay_rate=0.96,
    staircase=True)

optimizer = tf.keras.optimizers.Adam(lr_schedule)


import skimage.io
tensorboard_log_dir ="/kaggle/working/logs"
file_writer = tf.summary.create_file_writer(tensorboard_log_dir)


image_batch=[]
image_label_batch=[]
batch_index=0
num_epochs=0
numzeroblockimages=0

epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

@tf.function
def map_func_1(x):
    return tf.math.count_nonzero(tf.less(x, 0))

@tf.function
def map_func_2(x):
    return (x > nonzeropixels)


block_model = tf.keras.applications.MobileNetV2(input_shape=[block_height, block_width, 3], include_top=False)
block_model.trainable = False

@tf.function
def map_func_3(x):
    return block_model(tf.expand_dims(x,0))

max_l = tf.keras.layers.Maximum()

for epoch in range(EPOCHS):

    for index, row in train_labels.iterrows():

        i_g = row["isup_grade"]
        if i_g==0 or i_g==1:
            if np.random.uniform() > 0.5:
                continue

        file = os.path.join(data_dir, f'{index}.tiff')

        if os.path.isfile(file):

            region_all = skimage.io.MultiImage(file)
            region_all = region_all[tiff_level]

            (height, width, depth) = region_all.shape
            if (height > 10000 or width > 10000):
                region_all = None
                continue

            print(epoch, file, row["data_provider"], width, height, depth)
            pad_w = block_width-(width%block_width)
            pad_h = block_height-(height%block_height)
            split_w = (int)(width/block_width)+1
            split_h = (int)(height/block_height)+1
            paddings = tf.constant([[0, pad_h], [0, pad_w], [0,0]])
            region_all = tf.pad(region_all, paddings, constant_values=255)
            region_all = tf.cast(region_all, tf.float32)

            region_all = tf.image.per_image_standardization(region_all)
            #plt.imshow(region_all)
            #plt.show()
            #print(region_all.shape)

            #print(region_all.shape)
            region_all = tf.split(region_all, split_h, axis=0)
            #print(region_all[0].shape, len(region_all))

            region_all = tf.stack(region_all)
            #print(region_all.shape)

            region_all = tf.split(region_all, split_w, axis=2)
            #print(region_all[0].shape, len(region_all))

            region_all = tf.stack(region_all)
            #print(region_all.shape)

            region_all = tf.reshape(region_all, (-1, block_height, block_width,3))
            #print(region_all.shape)

            mask = tf.map_fn(map_func_1, region_all, dtype=tf.int64, parallel_iterations=region_all.shape[0])
            #print(mask)
            mask = tf.map_fn(map_func_2, mask, dtype=tf.bool, parallel_iterations=mask.shape[0])
            #print(mask)

            region_all = tf.boolean_mask(region_all, mask)
            #print(region_all.shape)

            if (region_all.shape[0] <= 1):
                continue

            if region_all.shape[0] > 120:
                region_all = tf.slice(region_all, [0,0,0,0], [120,block_height,block_width,3])
                #print(region_all.shape)

            with tf.GradientTape() as tape:
                #region_all = tf.map_fn(tf.function(map_func_3), region_all, dtype=tf.float32, parallel_iterations=region_all.shape[0])
                #print(region_all.shape)
                #block_batch = tf.squeeze(region_all)
                #print(block_batch.shape)




                #print(block_batch.shape)
                block_batch = block_model(region_all)
                region_all = None
                #print(block_batch.shape)

                block_batch = tf.split(block_batch, block_batch.shape[0])
                #print(block_batch[0].shape, len(block_batch))

                block_batch = max_l(block_batch)
                block_batch = tf.squeeze(block_batch)
                #print(block_batch.shape)

                image_batch.append(block_batch)
                image_label_batch.append(row["isup_grade"])
                batch_index = batch_index+1

                if batch_index >= BATCH_SIZE:

                    image_batch = tf.stack(image_batch, axis=0)
                    #image_batch = tf.math.add_n(image_batch)
                    #print(image_batch.shape)
                    image_label_batch = np.stack(image_label_batch, axis=0)
                    image_label_batch = image_label_batch.astype(np.float)

                    #with tf.GradientTape() as tape:
                    tf_image_batch = sequence_model(image_batch, training=True)
                    #print(tf_image_batch.shape)
                    loss_value = loss(image_label_batch, tf_image_batch)
                    #print(loss_value.numpy())
                    tf.summary.scalar("loss", loss_value)

                    variables = block_model.trainable_variables + sequence_model.trainable_variables
                    #variables = sequence_model.trainable_variables
                    gradients = tape.gradient(loss_value, variables)
                    optimizer.apply_gradients(zip(gradients, variables))

                    epoch_accuracy.update_state(image_label_batch, tf_image_batch)
                    print(loss_value.numpy(), epoch_accuracy.result().numpy())

                    lr = optimizer.learning_rate(optimizer.iterations)
                    print(optimizer.iterations.numpy(), lr.numpy())

                    if optimizer.iterations.numpy()%100==0:
                        block_model.save_weights(checkpoint_path_block_model)
                        sequence_model.save_weights(checkpoint_path_sequence_model)


                    image_batch=[]
                    image_label_batch=[]
                    batch_index=0
