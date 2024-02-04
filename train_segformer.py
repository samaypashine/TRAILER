# /**
#  * @file train_segformer.py 
#  * @author Samay Pashine
#  * @updated Samay Pashine
#  * @brief Code to train a SegFormer image segementation model to track the ball in the frames. 
#  * @version 1.4
#  * @date 2023-11-8
#  * @copyright Copyright (c) 2023
#  */

# Importing necessary libraries
import os
from glob import glob
import random
import argparse
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import backend
from transformers import TFSegformerForSemanticSegmentation, SegformerConfig, TFSegformerModel
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import MaxPooling2D, GlobalMaxPool2D
from keras.layers import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from IPython.display import clear_output


# Enabling the memory growth on GPU.
try:
    physical_devices = tf.config.list_physical_devices('GPU')
    print("Devices : ", physical_devices)
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*4)])
except Exception as e:
    print('No GPU device found.')


mean = tf.constant([0.485, 0.456, 0.406])
std = tf.constant([0.229, 0.224, 0.225])

def normalize(input_image, input_mask):
    input_image = tf.image.convert_image_dtype(input_image, tf.float32)
    input_image = (input_image - mean) / tf.maximum(std, backend.epsilon())
    
    uniq_val = np.unique(input_mask)
    input_mask = tf.where(input_mask == uniq_val[1], 1, 0)
    input_mask = tf.where(input_mask == uniq_val[2], 2, 0)
    return input_image, input_mask

# Function to load the images and masks.
def load_data(image_list, mask_list):
    
    image = tf.io.read_file(image_list)
    image = tf.image.decode_png(image, channels=3)

    mask = tf.io.read_file(mask_list)
    mask = tf.image.decode_png(mask, channels=1)

    input_image = tf.image.resize(image, IMAGE_SIZE)
    input_mask = tf.image.resize(
        mask,
        IMAGE_SIZE,
        method="bilinear",
    )

    input_image, input_mask = normalize(input_image, input_mask)
    input_image = tf.transpose(input_image, (2, 0, 1))

    return {"pixel_values": input_image, "labels": tf.squeeze(input_mask)}

# Function to map and batch the list of images to the function above. 
def data_generator(image_list, mask_list):
    dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    return dataset

# Function to display the frame.
def display(display_list, epoch):
    plt.figure(figsize=(15, 15))

    title = ["Input Image", "True Mask", "Predicted Mask"]

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis("off")
    plt.savefig(os.path.join(GRAPHS_DIR, f"infer_{epoch:08d}.jpg"))
    plt.clf()


def create_mask(pred_mask):
    pred_mask = tf.math.argmax(pred_mask, axis=1)
    pred_mask = tf.expand_dims(pred_mask, -1)
    return pred_mask[0]


def show_predictions(dataset=None, epoch=1, num=1):
    for sample in dataset.take(num):
        images, masks = sample["pixel_values"], sample["labels"]
        masks = tf.expand_dims(masks, -1)
        pred_masks = model.predict(images).logits
        images = tf.transpose(images, (0, 2, 3, 1))
        display([images[0], masks[0], create_mask(pred_masks)], epoch=epoch)

# Callback for training to visualize the validation results.
class DisplayCallback(tf.keras.callbacks.Callback):
    def __init__(self, dataset, **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset

    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions(self.dataset, epoch=epoch)
        print("\nSample Prediction after epoch {}\n".format(epoch + 1))

if __name__ == '__main__':

    # Configuring the log handler.
    log_path = r'logs/train-segformer' + os.sep + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.log'
    os.makedirs(r'logs/train-segformer', exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
                        handlers=[logging.FileHandler(log_path), logging.StreamHandler()])

    # Initializing the argument parser.
    parser = argparse.ArgumentParser(description='Script to train and save the ball tracking model.')
    parser.add_argument('-d', '--dataset', default='./dataset', help='Path to dataset.')
    parser.add_argument('-b', '--batch-size', default=1, type=int, help='Batch size of the model.')
    parser.add_argument('-s', '--size', default=(512, 512), help='Image resizing shape for the model.')
    parser.add_argument('-lr', '--learning-rate', default=6e-5, type=float, help='Learning rate')
    parser.add_argument('-wd', '--weight-decay', default=6e-5, type=float, help='Weight Decay')
    parser.add_argument('-e', '--epochs', default=100, type=int, help='Epochs to train the model.')
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

    MODEL_DIR = os.path.join(OUTPUT_DIR, 'models', f'v{folder_name}-segformer')
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'models', f'v{folder_name}-segformer', 'checkpoints')
    GRAPHS_DIR = os.path.join(OUTPUT_DIR, 'graphs', f'v{folder_name}-segformer')
    os.makedirs(rf'{MODEL_DIR}', exist_ok=True)
    os.makedirs(rf'{CHECKPOINT_DIR}', exist_ok=True)
    os.makedirs(rf'{GRAPHS_DIR}', exist_ok=True)

    logging.info('##########################################################################################################')
    logging.info(f'TRAINING VERSION                        :   V-{folder_name}-segformer')
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
        DisplayCallback(val_dataset),
    ]

    # Defining the model configuration
    id2label = {0: "rest", 1: "inner"}
    label2id = {label: id for id, label in id2label.items()}
    num_labels = len(id2label)

    model_checkpoint = "nvidia/mit-b0"
    model = TFSegformerForSemanticSegmentation.from_pretrained(
        model_checkpoint,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )

    model.compile(optimizer=Adam(weight_decay=WEIGHT_DECAY), loss=tf.keras.losses.CategoricalCrossentropy())
    model.summary()
    results = model.fit(train_dataset, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=callbacks,\
                        validation_data=val_dataset)
    model.save_pretrained(os.path.join(MODEL_DIR, 'final_model'), saved_model=True)

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

    plt.plot(results.history["val_loss"])
    plt.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
    plt.title("Validation Loss")
    plt.ylabel("val_loss")
    plt.xlabel("epoch")
    plt.legend()
    plt.savefig(os.path.join(GRAPHS_DIR, 'validation_loss.jpg'))
    plt.clf()
