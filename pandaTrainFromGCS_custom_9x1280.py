import numpy as np
import tensorflow as tf
import os

print(tf.__version__)
tf.config.list_physical_devices('TPU')

from io import BytesIO
PROJECT_ID = 'rih-ai-dev'
from google.cloud import storage

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tf/panda/rih-rl-dev-cefa3a49bbf3.json"

storage_client = storage.Client(project=PROJECT_ID)
bucket = storage_client.get_bucket('awxyz-us-central1')   

# Detect hardware, return appropriate distribution strategy
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.

print("REPLICAS: ", strategy.num_replicas_in_sync)

EPOCHS = 64
per_replica_batch_size = 32
BATCH_SIZE = per_replica_batch_size * strategy.num_replicas_in_sync # this is 8 on TPU v3-8, it is 1 on CPU and GPU
LR_START = 0.000001
LR_MAX = 0.00001 * strategy.num_replicas_in_sync
LR_MIN = 0.00000001
LR_RAMPUP_EPOCHS = 5
LR_SUSTAIN_EPOCHS = 5
LR_EXP_DECAY = .98

def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
    return lr


validation_split = 0.19
filenames = tf.io.gfile.glob("gs://awxyz-us-central1/panda/tfrecords_2/record_*.tfrecords") # list files on GCS
print("number of record files {}".format(len(filenames)))
split = len(filenames) - int(len(filenames) * validation_split)
TRAINING_FILENAMES = filenames[:split]
VALIDATION_FILENAMES = filenames[split:]
TRAIN_STEPS = (len(TRAINING_FILENAMES)*8) // BATCH_SIZE
print("number of train steps {}".format(TRAIN_STEPS))
TEST_STEPS = (len(VALIDATION_FILENAMES)*8) // BATCH_SIZE
print("number of test steps {}".format(TEST_STEPS))


AUTO = tf.data.experimental.AUTOTUNE
ignore_order = tf.data.Options()
ignore_order.experimental_deterministic = False

image_mask_feature_description = {
    'image': tf.io.FixedLenFeature([17280000], tf.float32),
    'label': tf.io.FixedLenFeature([], tf.int64),
    'provider': tf.io.VarLenFeature(tf.string),
    'gleason': tf.io.VarLenFeature(tf.string),
    'filename':tf.io.VarLenFeature(tf.string)
}


def get_test_dataset(batch_size, is_training=True):

    dataset = tf.data.TFRecordDataset(VALIDATION_FILENAMES, num_parallel_reads=AUTO)

    def parse_tfrecord(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        #print(example_proto)
        image_features = tf.io.parse_example(example_proto, image_mask_feature_description)
        blocks = image_features['image']
        label = image_features['label']
        return blocks, label


    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(parse_tfrecord)
    dataset = dataset.batch(batch_size)

    return dataset


def get_train_dataset(batch_size, is_training=True):

    dataset = tf.data.TFRecordDataset(TRAINING_FILENAMES, num_parallel_reads=AUTO)

    def parse_tfrecord(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        #print(example_proto)
        image_features = tf.io.parse_example(example_proto, image_mask_feature_description)
        blocks = image_features['image']
        label = image_features['label']
        return blocks, label

    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Only shuffle and repeat the dataset in training. The advantage to have a
    # infinite dataset for training is to avoid the potential last partial batch
    # in each epoch, so users don't need to think about scaling the gradients
    # based on the actual batch size.
    if is_training:
        dataset = dataset.shuffle(40)
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size)

    return dataset



with strategy.scope():
    Q = tf.Variable(tf.random.normal([1]), trainable=True)

    inputs = tf.keras.layers.Input(shape=(17280000,))
    #reshape = tf.keras.layers.Reshape((1500,3*3*1280))(inputs)
    #conv3d = tf.keras.layers.Conv3D(5, kernel_size=(1000,3,3), activation='relu')(reshape[:,:1000,:,:,:])
    #attn = tf.keras.layers.Attention()([Q,conv3d[:,0,0,:,:]])
    #attn = tf.keras.layers.Attention()([Q,reshape])
    #flat = tf.keras.layers.Flatten()(attn)
    #dense = tf.keras.layers.Dense(6, activation='relu')(flat)
    dense = tf.keras.layers.Dense(6, activation='relu')(inputs)

    model = tf.keras.models.Model(inputs=inputs, outputs=dense)
    model.trainable=True

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        0.00001,
        decay_steps=TRAIN_STEPS//2,
        decay_rate=0.96,
        staircase=True)

    optimizer = tf.keras.optimizers.Adam(lr_schedule)


    training_loss = tf.keras.metrics.Mean('training_loss', dtype=tf.float32)
    training_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      'training_accuracy', dtype=tf.float32)

    test_loss = tf.keras.metrics.Mean('training_loss', dtype=tf.float32)
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      'training_accuracy', dtype=tf.float32)

    ckpt = tf.train.Checkpoint(step=tf.Variable(1), Q=Q, model=model, optimizer=optimizer)
    manager = tf.train.CheckpointManager(ckpt, 'gs://writeallchkpts/dense_only_custom/chkpt', max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    #lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #    0.000005,
    #    decay_steps=TRAIN_STEPS//2,
    #    decay_rate=0.96,
    #    staircase=True)

    #optimizer = tf.keras.optimizers.Adam(lr_schedule)

    writer = tf.summary.create_file_writer("gs://writeallchkpts/dense_only_custom/chkpt/logs")


train_dataset = strategy.experimental_distribute_datasets_from_function(
    lambda _: get_train_dataset(per_replica_batch_size, is_training=True))

test_dataset = strategy.experimental_distribute_datasets_from_function(
    lambda _: get_test_dataset(per_replica_batch_size, is_training=False))

train_iterator = iter(train_dataset)
test_iterator = iter(test_dataset)



@tf.function
def train_step(iterator):
    """The step function for one training step"""
    def step_fn(inputsx):
        """The computation to run on each TPU device."""
        blocks, labels = inputsx
        tf.dtypes.cast(blocks, tf.float32)
        tf.dtypes.cast(labels, tf.int64)


        with tf.GradientTape() as tape:
            logits = model(blocks)
            #blocks = inputs(blocks)
            #blocks = reshape(blocks)
            #blocks = conv3d(blocks)
            #blocks = attn(blocks)
            #blocks = flat(blocks)
            #logits = dense(blocks)
            tf.dtypes.cast(logits, tf.float32)
            tf.dtypes.cast(blocks, tf.float32)
            loss_value = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
            loss = tf.nn.compute_average_loss(loss_value, global_batch_size=BATCH_SIZE)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))

        training_loss.update_state(loss * strategy.num_replicas_in_sync)
        training_accuracy.update_state(labels, logits)

    strategy.run(step_fn, args=(next(iterator),))


@tf.function
def test_step(blocks,labels):
    """The step function for one training step"""
    def step_fn(inputsx):
        """The computation to run on each TPU device."""
        blocks, labels = inputsx

        with tf.GradientTape() as tape:
            logits = model(blocks)
            loss_value = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
            loss = tf.nn.compute_average_loss(loss_value, global_batch_size=BATCH_SIZE)

        test_loss.update_state(loss * strategy.num_replicas_in_sync)
        test_accuracy.update_state(labels, logits)

    strategy.run(step_fn, args=((blocks,labels),))





for epoch in range(EPOCHS):

    print('Epoch: {}/{}'.format(epoch, EPOCHS))
    progbar = tf.keras.utils.Progbar(TRAIN_STEPS)
    for i in range(TRAIN_STEPS):
        train_step(train_iterator)
        ckpt.step.assign_add(1)
        progbar.update(i+1)
        if (ckpt.step.numpy()) % (TRAIN_STEPS // 2) == 2:
            save_path = manager.save()
            print("\nSaved checkpoint for step {}: {}\n".format(int(ckpt.step.numpy()), save_path))

    with writer.as_default():
        tf.summary.scalar("Train/loss", training_loss.result(), step=optimizer.iterations)
        tf.summary.scalar("Train/accuracy", training_accuracy.result(), step=optimizer.iterations)
        writer.flush()

    save_path = manager.save()
    print("\nSaved checkpoint for step {}: {}\n".format(int(ckpt.step.numpy()), save_path))
    print('Current step: {}, lr: {}, training loss: {}, accuracy: {}%'.format(
        optimizer.iterations.numpy(),
        optimizer.learning_rate(optimizer.iterations).numpy(),
        round(float(training_loss.result()), 4),
        round(float(training_accuracy.result()) * 100, 2)))
    training_loss.reset_states()
    training_accuracy.reset_states()

    #for i in range(TEST_STEPS-1):
    #    test_step(test_iterator)

    for block, label in test_dataset:
        test_step(block, label)

    print('test loss: {}, accuracy: {}%'.format(
      round(float(test_loss.result()), 4),
      round(float(test_accuracy.result()) * 100, 2)))

    with writer.as_default():
        tf.summary.scalar("Test/loss", test_loss.result(), step=optimizer.iterations)
        tf.summary.scalar("Test/accuracy", test_accuracy.result(), step=optimizer.iterations)
        writer.flush()

    test_loss.reset_states()
    test_accuracy.reset_states()
