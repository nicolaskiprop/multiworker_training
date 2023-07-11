import json
import os
import sys

#disable all GPUs to prevent errors caused by all workers trying to use the same GPU.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#Reset the TF_CONFIG environment variable
os.environ.pop('TF_CONFIG', None)
#make sure the current directory is on Python's path
if '.' not in sys.path:
    sys.path.insert(0, '.')
#Install tf-nightly, as the frequency of checkpoint saving at a particular step with the save_freq argument in tf.keras.callbacks.BackupAndRestore is introduced from TensorFlow 2.10:
#pip install tf-nightly
import tensorflow as tf
from tensorflow import keras
import pathlib
#dataset and model definition
 #create a dataset
img_height = 180
img_width = 180
num_classes = None
def flower_dataset(batch_size):
    dataset_url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'
    data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
    data_dir = pathlib.Path(data_dir)
    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(image_count)
   

    #It's good practice to use a validation split when developing your model. Use 80% of the images for training and 20% for validation.
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset='training',
        seed=123,
        image_size=(img_height,img_width),
        batch_size=batch_size
    )
    print(train_ds)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset='validation',
        seed=123,
        image_size=(img_height,img_width),
        batch_size=batch_size
    )
    print(val_ds)

    global num_classes
    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(class_names)

    for image_batch, labels_batch in train_ds:
        print(image_batch.shape)
        print(labels_batch.shape)
        break

    #configure the dataset for performance using Dataset.cache() and Dataset.prefetch() methods.
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    train_ds = train_ds.repeat()
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    #standardize the data
    import numpy as np
    from keras import layers
    normalization_layer =layers.Rescaling(1./255)
    normalized_ds = train_ds.map(lambda x,y:(normalization_layer(x),y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    print(np.min(first_image),np.max(first_image))
    return normalized_ds

def build_and_compile_cnn_model():
    #create a model
   
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255,input_shape=(img_height,img_width,3)),
        tf.keras.layers.Conv2D(16,3,padding='same',activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32,3,padding='same',activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64,3,padding='same',activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ])

    #compile the model
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    return model