import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random
import tensorflow as tf
import cv2
from tqdm import tqdm
import datetime
from tensorflow import keras
from tensorflow.keras.layers import Conv2D , MaxPooling2D , UpSampling2D , Concatenate , Input , Add , Conv2DTranspose
from tensorflow.keras.models import Sequential , Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import SGD ,Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy , MeanSquaredError , BinaryCrossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.keras import callbacks

from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from IPython.display import clear_output
# %matplotlib inline

from IPython.display import HTML
from base64 import b64encode

train_data_dir = "training\image_2/"
train_gt_dir = "training\gt_image_2/"
test_data_dir = "testing/"

TRAINSET_SIZE = int(len(os.listdir(train_data_dir))*0.8)
print(f"Number Of Training Examples: {TRAINSET_SIZE}")

VALIDSET_SIZE = int(len(os.listdir(train_data_dir))*0.1)
print(f"Number of Validation Examples: {VALIDSET_SIZE}")

TESTSET_SIZE = int(len(os.listdir(train_data_dir))-TRAINSET_SIZE-VALIDSET_SIZE)
print(f"Number of testing Examples: {TESTSET_SIZE}")

IMG_SIZE = 128
N_CHANNELS = 3
N_CLASSES = 1
SEED = 123

def parse_image(img_path : str) -> dict:
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image , channels=3)
    image = tf.image.convert_image_dtype(image , tf.uint8)

    mask_path = tf.strings.regex_replace(img_path,"image_2","gt_image_2")
    mask_path = tf.strings.regex_replace(mask_path,"um_","um_road_")
    mask_path = tf.strings.regex_replace(mask_path,"umm_","umm_road_")
    mask_path = tf.strings.regex_replace(mask_path,"uu_","uu_road_")

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask,channels=3)

    non_road_label = np.array([255,0,0])
    road_label = np.array([255,0,255])
    other_road_label = np.array([0,0,0])

    mask = tf.experimental.numpy.all(mask == road_label,axis=2)
    mask = tf.cast(mask,tf.uint8)
    mask = tf.expand_dims(mask,axis = -1)

    return {'image':image , 'segmentation_mask':mask}

all_dataset = tf.data.Dataset.list_files(train_data_dir +"*.png",seed=SEED)
all_dataset = all_dataset.map(parse_image)

train_dataset = all_dataset.take(TRAINSET_SIZE+VALIDSET_SIZE)
val_dataset = train_dataset.skip(TRAINSET_SIZE)
train_dataset = train_dataset.take(TRAINSET_SIZE)
test_dataset = all_dataset.skip(TRAINSET_SIZE + VALIDSET_SIZE)
@tf.function
def normalize(input_image : tf.Tensor , input_mask: tf.Tensor)-> tuple:
    input_image = tf.cast(input_image , tf.float32)/255.0
    return input_image,input_mask

@tf.function
def load_image_train(datapoint:dict)->tuple:
    input_image = tf.image.resize(datapoint['image'],(IMG_SIZE,IMG_SIZE))
    input_mask = tf.image.resize(datapoint['segmentation_mask'],(IMG_SIZE,IMG_SIZE))

    if tf.random.uniform(())>0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)
    input_image,input_mask = normalize(input_image,input_mask)

    return input_image,input_mask

@tf.function
def load_image_test(datapoint:dict)->tuple:
    input_image = tf.image.resize(datapoint['image'],(IMG_SIZE,IMG_SIZE))
    input_mask = tf.image.resize(datapoint['segmentation_mask'],(IMG_SIZE,IMG_SIZE))
    input_image,input_mask = normalize(input_image,input_mask)
    
    return input_image,input_mask

BATCH_SIZE = 32
BUFFER_SIZE=1000

dataset = {"train":train_dataset,"val":val_dataset,"test":test_dataset}
#train Dataset
dataset['train'] = dataset['train'].map(load_image_train , num_parallel_calls = tf.data.AUTOTUNE)
dataset['train'] = dataset['train'].shuffle(buffer_size=BUFFER_SIZE,seed=SEED)
dataset['train'] = dataset['train'].repeat()
dataset['train'] = dataset['train'].batch(BATCH_SIZE)
dataset['train'] = dataset['train'].prefetch(buffer_size=tf.data.AUTOTUNE)

#validation dataset

dataset['val'] = dataset['val'].map(load_image_test)
dataset['val'] = dataset['val'].repeat()
dataset['val'] = dataset['val'].batch(BATCH_SIZE)
dataset['val'] = dataset['val'].prefetch(buffer_size=tf.data.AUTOTUNE)

#testing dataset

dataset['test'] = dataset['test'].map(load_image_test)
dataset['test'] = dataset['test'].batch(BATCH_SIZE)
dataset['test'] = dataset['test'].prefetch(buffer_size=tf.data.AUTOTUNE)

print(dataset['train'])
print(dataset['val'])
print(dataset['test'])

def display_sample(display_list):
    plt.figure(figsize=(18,18))
    title = ['Input Image','True Mask','Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1,len(display_list),i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')

    plt.show()
for image,mask in dataset["train"].take(1):
    sample_image , sample_mask = image , mask

display_sample([sample_image[0],sample_mask[0]])
vgg16_model = VGG16()
vgg16_model.summary()


input_shape = (IMG_SIZE,IMG_SIZE,N_CHANNELS)
#input
inputs = Input(input_shape)
#VGG network
vgg16_model = VGG16(include_top = False , weights = 'imagenet', input_tensor = inputs)
#encoder layer
c1 = vgg16_model.get_layer("block3_pool").output
c2 = vgg16_model.get_layer("block4_pool").output
c3 = vgg16_model.get_layer("block5_pool").output

#decoder
u1 = UpSampling2D((2,2),interpolation = 'bilinear')(c3)
d1 = Concatenate()([u1 , c2])

u2 = UpSampling2D((2,2), interpolation = 'bilinear')(d1)
d2 = Concatenate()([u2 , c1])

#output
u3 = UpSampling2D((8,8), interpolation = 'bilinear')(d2)
outputs = Conv2D(N_CLASSES,1,activation = 'sigmoid')(u3)

model = Model(inputs , outputs, name = "VGG_FCN8")

m_iou = tf.keras.metrics.MeanIoU(2)
model.compile(optimizer=Adam(),loss=BinaryCrossentropy(),metrics=[m_iou])

def create_mask(pred_mask : tf.Tensor)->tf.Tensor:
    pred_mask = tf.math.round(pred_mask)
    pred_mask = tf.expand_dims(pred_mask , axis =-1)
    return pred_mask

def show_predictions(dataset=None,num=1):
    if dataset:
        for image , mask in dataset.take(num):
            pred_mask = model.predict(image)
            display_sample([image[0], true_mask ,create_mask(pred_mask)])
    else:
        # Predict and show the sample image
        inference = model.predict(sample_image)
        display_sample([sample_image[0], sample_mask[0],
                        inference[0]])
        
for image, mask in dataset['train'].take(1):
    sample_image, sample_mask = image, mask

show_predictions()

# Callbacks and Logs
class DisplayCallback(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions()
        print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

callbacks = [
    DisplayCallback(),
    callbacks.TensorBoard(logdir, histogram_freq = -1),
    callbacks.EarlyStopping(patience = 10, verbose = 1),
    callbacks.ModelCheckpoint('best_model.keras', verbose = 1, save_best_only = True)
]
        
# Set Variables
EPOCHS = 200
STEPS_PER_EPOCH = TRAINSET_SIZE // BATCH_SIZE
VALIDATION_STEPS = VALIDSET_SIZE // BATCH_SIZE

model_history = model.fit(dataset['train'], epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_data = dataset["val"],
                          validation_steps=VALIDATION_STEPS,
                          callbacks = callbacks)