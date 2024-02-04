# /**
#  * @file train_unet.py 
#  * @author Samay Pashine
#  * @updated Samay Pashine
#  * @brief Code to train a image segementation model to track the ball in the frames using Encoder-Decoder UNet. 
#  * @version 1.5
#  * @date 2023-11-8
#  * @copyright Copyright (c) 2023
#  */

# Importing necessary libraries
import os
import logging
import random
import argparse
import numpy as np
from glob import glob
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
from keras.layers import Input, BatchNormalization, Activation, Dropout
from keras.layers import Conv2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.layers import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


# Enabling the memory growth on GPU.
try:
    physical_devices = tf.config.list_physical_devices('GPU')
    print("Devices : ", physical_devices)
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*4)])
except Exception as e:
    print('No GPU device found.')

# Function to read the images and masks.
def read_image(image_path, mask=False):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image.set_shape([None, None, 3])

    if mask: image = tf.image.rgb_to_grayscale(image)
    image = tf.image.resize(images=image, size=IMAGE_SIZE)
    image = image / 255.0
    return image

# Function to load the images and masks.
def load_data(image_list, mask_list):
    img = read_image(image_list)
    mask = read_image(mask_list, mask=True)
    return img, mask

# Function to map and batch the list of images to the function above. 
def data_generator(image_list, mask_list):
    dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    return dataset

def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    """Function to add 2 convolutional layers with the parameters passed to it"""

    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm: x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm: x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x

def get_unet(input_img, n_filters = 16, dropout = 0.1, batchnorm = True):
    """Function to define the UNET Model"""
    # Contracting Path
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    
    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    
    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    
    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    
    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
    outputs = Conv2D(NUM_CLASSES, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

if __name__ == '__main__':
    
    # Configuring the log handler.
    log_path = r'logs/train-unet' + os.sep + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.log'
    os.makedirs(r'logs/train-unet', exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
                        handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
    
    # Initializing the argument parser.
    parser = argparse.ArgumentParser(description='Script to train and save the ball tracking model.')
    parser.add_argument('-d', '--dataset', default='./dataset', help='Path to dataset.')
    parser.add_argument('-b', '--batch-size', default=2, type=int, help='Batch size of the model.')
    parser.add_argument('-s', '--size', default=(512, 512), help='Image resizing shape for the model.')
    parser.add_argument('-lr', '--learning-rate', default=1e-5, type=float, help='Learning rate')
    parser.add_argument('-wd', '--weight-decay', default=1e-5, type=float, help='Weight Decay')
    parser.add_argument('-e', '--epochs', default=500, type=int, help='Epochs to train the model.')
    parser.add_argument('-c', '--classes', default=2, type=int, help='Number of classes to segment.')
    parser.add_argument('-ckp', '--checkpoint-period', default=5, type=int, help='Number of epochs between checkpoints.')
    parser.add_argument('-o', '--output', default='./output', help='Output directory path.')
    args = parser.parse_args()

    # Intitializing the variables with arguement parser.
    NUM_CLASSES = args.classes
    IMAGE_SIZE = args.size
    BATCH_SIZE = args.batch_size
    DATA_DIR = args.dataset
    OUTPUT_DIR = args.output
    LEARNING_RATE = args.learning_rate
    WEIGHT_DECAY = args.weight_decay
    EPOCHS = args.epochs
    CKP = args.checkpoint_period

    # Creating new folder for output and models for new instances.
    os.makedirs(rf'{os.path.join(OUTPUT_DIR)}', exist_ok=True)
    os.makedirs(rf'{os.path.join(OUTPUT_DIR, "models")}', exist_ok=True)
    os.makedirs(rf'{os.path.join(OUTPUT_DIR, "graphs")}', exist_ok=True)
    folder_name = len(os.listdir(os.path.join(OUTPUT_DIR, 'models'))) + 1

    MODEL_DIR = os.path.join(OUTPUT_DIR, 'models', f'v{folder_name}-unet')
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'models', f'v{folder_name}-unet', 'checkpoints')
    GRAPHS_DIR = os.path.join(OUTPUT_DIR, 'graphs', f'v{folder_name}-unet')
    os.makedirs(rf'{MODEL_DIR}', exist_ok=True)
    os.makedirs(rf'{CHECKPOINT_DIR}', exist_ok=True)
    os.makedirs(rf'{GRAPHS_DIR}', exist_ok=True)

    logging.info('##########################################################################################################')
    logging.info(f'TRAINING VERSION                        :   V-{folder_name}-unet')
    logging.info(f'DATASET DIRECTORY                       :   {DATA_DIR}')
    logging.info(f'OUTPUT DIRECTORY                        :   {OUTPUT_DIR}')
    logging.info(f'BATCH SIZE                              :   {BATCH_SIZE}')
    logging.info(f'IMAGE SIZE                              :   {IMAGE_SIZE}')
    logging.info(f'LEARNING RATE                           :   {LEARNING_RATE}')
    logging.info(f'WEIGHT DECAY                            :   {WEIGHT_DECAY}')
    logging.info(f'EPOCHS                                  :   {EPOCHS}')
    logging.info('##########################################################################################################')

    # Checking if the dataset directory heirarchy is valid or not.
    assert os.path.isdir(os.path.join(DATA_DIR, 'images'))
    assert os.path.isdir(os.path.join(DATA_DIR, 'masks'))

    # Setting the number of training and validation images.
    NUM_TRAIN_IMAGES = int(len(os.listdir(os.path.join(DATA_DIR, 'masks'))) * 0.9)
    NUM_VAL_IMAGES = int(len(os.listdir(os.path.join(DATA_DIR, 'masks'))) * 0.1)

    # Listing the paths of all the images in the dataset, sorting them by names and spliting into training and vaidation set.
    # data = dict(zip(sorted(glob(os.path.join(DATA_DIR, "images/*"))), sorted(glob(os.path.join(DATA_DIR, "masks/*")))))
    data = dict(zip(sorted(glob(os.path.join(DATA_DIR, "images/*")))[: NUM_TRAIN_IMAGES + NUM_VAL_IMAGES], sorted(glob(os.path.join(DATA_DIR, "masks/*")))))
    data_list = list(data.items())

    # Shuffling the list of images
    random.seed(3)
    random.shuffle(data_list)
    train_data = dict(data_list)

    # Spliting the list of images into 3 partition i.e. train / val / test.
    train_images = list(train_data.keys())[:NUM_TRAIN_IMAGES]
    train_masks = list(train_data.values())[:NUM_TRAIN_IMAGES]

    val_images = list(train_data.keys())[NUM_TRAIN_IMAGES:NUM_TRAIN_IMAGES+NUM_VAL_IMAGES]
    val_masks = list(train_data.values())[NUM_TRAIN_IMAGES:NUM_TRAIN_IMAGES+NUM_VAL_IMAGES]

    train_dataset = data_generator(train_images, train_masks)
    val_dataset = data_generator(val_images, val_masks)

    logging.info(f"Train images                            : {train_images[:5]}")
    logging.info(f"Train masks                             : {train_masks[:5]}")
    logging.info(f"val images                              : {val_images[:5]}")
    logging.info(f"val masks                               : {val_masks[:5]}")
    logging.info(f"Train Dataset                           : {train_dataset}")
    logging.info(f"Validation Dataset                      : {val_dataset}")

    callbacks = [
        EarlyStopping(patience=5, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=5, min_lr=LEARNING_RATE, verbose=1),
        ModelCheckpoint(os.path.join(CHECKPOINT_DIR, "checkpoint_{epoch:08d}.h5"),
                        verbose=0,
                        monitor="val_loss",
                        save_best_only=False,
                        save_weights_only=False,
                        period=CKP),
        ModelCheckpoint(os.path.join(CHECKPOINT_DIR, "checkpoint_best.h5"),
                        verbose=0,
                        monitor="val_loss",
                        save_best_only=True,
                        save_weights_only=False)
    ]

    input_img = Input((IMAGE_SIZE[0], IMAGE_SIZE[0], 3), name='img')

    model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)
    model.compile(optimizer=Adam(weight_decay=WEIGHT_DECAY), loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    results = model.fit(train_dataset, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=callbacks,\
                        validation_data=val_dataset)
    model.save(os.path.join(MODEL_DIR, 'final_model.h5'))

    # Converting the model to TFlite model and saving it.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(os.path.join(MODEL_DIR, 'final_model.tflite'), 'wb') as f: f.write(tflite_model)

    # Creating and saving the traning and validation graphs in the respective directory
    plt.style.use("ggplot")
    plt.plot(results.history["loss"])
    plt.plot( np.argmin(results.history["loss"]), np.min(results.history["loss"]), marker="x", color="r", label="best model")
    plt.title("Training Loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend()
    plt.savefig(os.path.join(GRAPHS_DIR, 'training_loss.jpg'))
    plt.clf()

    plt.plot(results.history["accuracy"])
    plt.plot( np.argmax(results.history["accuracy"]), np.max(results.history["accuracy"]), marker="x", color="r", label="best model")
    plt.title("Training Accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend()
    plt.savefig(os.path.join(GRAPHS_DIR, 'training_accuracy.jpg'))
    plt.clf()

    plt.plot(results.history["val_loss"])
    plt.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
    plt.title("Validation Loss")
    plt.ylabel("val_loss")
    plt.xlabel("epoch")
    plt.legend()
    plt.savefig(os.path.join(GRAPHS_DIR, 'validation_loss.jpg'))
    plt.clf()

    plt.plot(results.history["val_accuracy"])
    plt.plot( np.argmax(results.history["val_accuracy"]), np.max(results.history["val_accuracy"]), marker="x", color="r", label="best model")
    plt.title("Validation Accuracy")
    plt.ylabel("val_accuracy")
    plt.xlabel("epoch")
    plt.legend()
    plt.savefig(os.path.join(GRAPHS_DIR, 'validation_accuracy.jpg'))
    plt.clf()