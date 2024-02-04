# /**
#  * @file train_deeplabv3.py 
#  * @author Samay Pashine
#  * @updated Samay Pashine
#  * @brief Code to train a Deeplabv3 segementation model to track the ball in the frames. 
#  * @version 2.4
#  * @date 2023-10-26
#  * @copyright Copyright (c) 2023
#  */

# Importing necessary libraries.
import os
import cv2
import random
import logging
import argparse
import numpy as np
from glob import glob
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


# Function to read the images and masks.
def read_image(image_path, mask=False):
    image = tf.io.read_file(image_path)
    if mask:
        image = tf.image.decode_png(image, channels=1)
        image.set_shape([None, None, 1])
        image = tf.image.resize(images=image, size=IMAGE_SIZE)
    else:
        image = tf.image.decode_png(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.image.resize(images=image, size=IMAGE_SIZE)
        image = tf.keras.applications.resnet50.preprocess_input(image)
    return image

# Function to load the images and masks.
def load_data(image_list, mask_list):
    image = read_image(image_list)
    mask = read_image(mask_list, mask=True)

    # uniq_val = np.unique(mask)
    mask = tf.where(mask > 0, 1, 0)
    # mask = tf.where(mask == uniq_val[2], 2, 0)
    return image, mask

# Function to map and batch the list of images to the function above. 
def data_generator(image_list, mask_list):
    dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    return dataset

# Function to add Convolutional block in the network graph.
def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False,
):
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
    )(block_input)
    x = layers.BatchNormalization()(x)
    return tf.nn.relu(x)

# Function to Concatenate the output from different Convultional layers.
def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output

# Function to build the complete Deeplab model. 
def DeeplabV3Plus(image_size, num_classes):
    model_input = keras.Input(shape=(image_size, image_size, 3))
    resnet50 = keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_tensor=model_input
    )
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)

    input_a = layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",)(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
    return keras.Model(inputs=model_input, outputs=model_output)


if __name__ == '__main__':

    # Configuring the log handler.
    log_path = r'logs/train-deeplab' + os.sep + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.log'
    os.makedirs(r'logs/train-deeplab', exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
                        handlers=[logging.FileHandler(log_path), logging.StreamHandler()])


    # Enabling the memory growth in the gpu if available and manually configuring the gpu memory limits.
    try:
        tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
        gpu = tf.config.experimental.list_physical_devices('GPU')[0]
        tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*4)])
    except Exception as e:
        logging.error(f'Exception: {e}')

    # Initializing the argument parser.
    parser = argparse.ArgumentParser(description='Script to train and save the ball tracking model.')
    parser.add_argument('-d', '--dataset', default='./dataset', help='Path to dataset.')
    parser.add_argument('-b', '--batch-size', default=2, type=int, help='Batch size of the model.')
    parser.add_argument('-s', '--size', default=(512, 512), help='Image resizing shape for the model.')
    parser.add_argument('-lr', '--learning-rate', default=1e-5, type=float, help='Learning rate')
    parser.add_argument('-wd', '--weight-decay', default=1e-5, type=float, help='Weight Decay')
    parser.add_argument('-p', '--pretrained', default='None', type=str, help='Path to pretrained weights (if any)')
    parser.add_argument('-e', '--epochs', default=100, type=int, help='Epochs to train the model.')
    parser.add_argument('-c', '--classes', default=3, type=int, help='Number of classes to segment.')
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
    PRETRAINED = args.pretrained

    # Creating new folder for output and models for new instances.
    os.makedirs(rf'{os.path.join(OUTPUT_DIR, "models")}', exist_ok=True)
    os.makedirs(rf'{os.path.join(OUTPUT_DIR, "graphs")}', exist_ok=True)
    folder_name = len(os.listdir(os.path.join(OUTPUT_DIR, 'models'))) + 1

    MODEL_DIR = os.path.join(OUTPUT_DIR, 'models', f'v{folder_name}-deeplab')
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'models', f'v{folder_name}-deeplab', 'checkpoints')
    GRAPHS_DIR = os.path.join(OUTPUT_DIR, 'graphs', f'v{folder_name}-deeplab')
    os.makedirs(rf'{MODEL_DIR}', exist_ok=True)
    os.makedirs(rf'{CHECKPOINT_DIR}', exist_ok=True)
    os.makedirs(rf'{GRAPHS_DIR}', exist_ok=True)

    logging.info('##########################################################################################################')
    logging.info(f'TRAINING VERSION                        :   V-{folder_name}-deeplab')
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
    logging.info(f'Number of Total Images                  :   {NUM_TRAIN_IMAGES+NUM_VAL_IMAGES}')
    logging.info(f'Number of Training Images               :   {NUM_TRAIN_IMAGES}')
    logging.info(f'Number of Validation Images             :   {NUM_VAL_IMAGES}')

    # Listing the paths of all the images in the dataset, sorting them by names and spliting into training and vaidation set.
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

    try:
        # Creating the train / val / test dataset.
        train_dataset = data_generator(train_images, train_masks)
        val_dataset = data_generator(val_images, val_masks)

        logging.info(f"Train images                            : {train_images[:5]}")
        logging.info(f"Train masks                             : {train_masks[:5]}")
        logging.info(f"val images                              : {val_images[:5]}")
        logging.info(f"val masks                               : {val_masks[:5]}")
        logging.info(f"Train Dataset                           : {train_dataset}")
        logging.info(f"Validation Dataset                      : {val_dataset}")

        # Creating the model.
        model = DeeplabV3Plus(image_size=IMAGE_SIZE[0], num_classes=NUM_CLASSES)
        if PRETRAINED != 'None': model = tf.keras.models.load_model(PRETRAINED)

        # Compiling the model.
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY),
            loss=loss,
            metrics=["accuracy"],
        )

        # Model callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=20),
            ReduceLROnPlateau(factor=0.1, patience=3, min_lr=LEARNING_RATE, verbose=1),
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

        # Training and saving the model.
        history = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS, callbacks=callbacks)
        model.save(os.path.join(MODEL_DIR, 'final_model.h5'))
        
        # Converting the model to TFlite model and saving it.
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        with open(os.path.join(MODEL_DIR, 'final_model.tflite'), 'wb') as f: f.write(tflite_model)

        # Creating and saving the traning and validation graphs in the respective directory
        plt.plot(history.history["loss"])
        plt.title("Training Loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend()
        plt.savefig(os.path.join(GRAPHS_DIR, 'training_loss.jpg'))
        plt.clf()

        plt.plot(history.history["accuracy"])
        plt.title("Training Accuracy")
        plt.ylabel("accuracy")
        plt.xlabel("epoch")
        plt.legend()
        plt.savefig(os.path.join(GRAPHS_DIR, 'training_accuracy.jpg'))
        plt.clf()

        plt.plot(history.history["val_loss"])
        plt.title("Validation Loss")
        plt.ylabel("val_loss")
        plt.xlabel("epoch")
        plt.legend()
        plt.savefig(os.path.join(GRAPHS_DIR, 'validation_loss.jpg'))
        plt.clf()

        plt.plot(history.history["val_accuracy"])
        plt.title("Validation Accuracy")
        plt.ylabel("val_accuracy")
        plt.xlabel("epoch")
        plt.legend()
        plt.savefig(os.path.join(GRAPHS_DIR, 'validation_accuracy.jpg'))
        plt.clf()
    except Exception as e:
        logging.error(f'Error Code : {e}')
        logging.info(f'Removing the directory structure for this instance.')
        os.removedirs(CHECKPOINT_DIR)
        os.removedirs(GRAPHS_DIR)
