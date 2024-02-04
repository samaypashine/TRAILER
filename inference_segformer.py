# /**
#  * @file inference_segformer.py 
#  * @author Samay Pashine
#  * @updated Samay Pashine
#  * @brief Code to track the ball in the frames of video using SegFormer model. 
#  * @version 1.5
#  * @date 2023-10-26
#  * @copyright Copyright (c) 2023
#  */

# Importing necessary libraries.
import os
import cv2
import time
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
from tensorflow.keras import backend
from transformers import TFSegformerForSemanticSegmentation


if __name__ == '__main__':

    # Configuring the log handler.
    log_path = r'logs/inference-segformer' + os.sep + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.log'
    os.makedirs(r'logs/inference-segformer', exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
                        handlers=[logging.FileHandler(log_path), logging.StreamHandler()])

    # Enabling the memory growth in the gpu if available and manually configuring the gpu memory limits.
    try:
        tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
        gpu = tf.config.experimental.list_physical_devices('GPU')[0]
    except Exception as e:
        logging.error(f'Exception: {e}')

    # Initializing the argument parser.
    parser = argparse.ArgumentParser(description='Script to infer on data using the segformer ball tracking model.')
    parser.add_argument('-d', '--data', default='./raw_data/videos', help='Path to dataset.')
    parser.add_argument('-m', '--model', required=True, help='Path to saved model.')
    parser.add_argument('-a', '--alpha', default=0.5, type=float, help='Overlay alpha')
    parser.add_argument('--save', action='store_true', help='Save the output as video.')
    parser.add_argument('-s', '--size', default=(512, 512), help='Image resizing shape for the model.')
    args = parser.parse_args()

    # Initializing the variable
    DATA_DIR = args.data
    IMAGE_SIZE = args.size
    MODEL_PATH = args.model
    ALPHA = args.alpha
    SAVE_FLAG = args.save
    FRAME_RATE = 30
    FPS_HIST = list()

    logging.info(f'DATA DIRECTORY       :       {DATA_DIR}')
    logging.info(f'MODEL PATH           :       {MODEL_PATH}')
    logging.info(f'IMAGE SIZE           :       {IMAGE_SIZE}')
    logging.info(f'ALPHA                :       {ALPHA}')
    logging.info(f'FRAME RATE           :       {FRAME_RATE}')
    
    # Checking if data directory and model file exists or not.
    assert os.path.isdir(DATA_DIR)
    assert os.path.isdir(MODEL_PATH)

    # Image Preprocessing configurations.
    mean = tf.constant([0.485, 0.456, 0.406])
    std = tf.constant([0.229, 0.224, 0.225])

    logging.info(f'Loading the Segformer model ...')
    model = TFSegformerForSemanticSegmentation.from_pretrained(MODEL_PATH)

    # Iterating over the video files.
    for video_file in os.listdir(DATA_DIR):
        
        # Opening the video file
        logging.info(f"Video File       :       {video_file}")
        cap = cv2.VideoCapture(os.path.join(DATA_DIR, video_file))

        if SAVE_FLAG:
            # Create a VideoWriter object to save the video
            os.makedirs(rf'{os.path.join("SegFormer Inference Videos")}', exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(os.path.join("./SegFormer Inference Videos", video_file), fourcc, FRAME_RATE, IMAGE_SIZE)

        # Create a blank image with the same dimensions as the mask
        height, width = IMAGE_SIZE
        output = np.zeros((128, 128, 3), dtype=np.uint8)

        while True:
            # Reading the frame
            ret, frame = cap.read()

            # Condition for empty frame
            if not ret: break

            # Processing the frame
            frame_orig = cv2.resize(frame, (128, 128))
            frame = cv2.resize(frame, IMAGE_SIZE)

            frame = (frame - mean) / tf.maximum(std, backend.epsilon())
            frame = tf.transpose(frame, (2, 0, 1))

            # Predicting the mask.
            s_t = time.time()
            pred = model.predict(tf.expand_dims(frame, 0)).logits
            FPS_HIST.append(60 / (time.time() - s_t))
            pred_mask = tf.math.argmax(pred, axis=1)
            pred_mask = tf.squeeze(pred_mask)
            logging.info(f'Average FPS       :       {sum(FPS_HIST) / len(FPS_HIST)}')
            
            # Set the color for the object in the mask (e.g., red)
            color1 = (0, 0, 255) # Red Color

            # Apply the mask to the output image to highlight the object
            output[pred_mask > 0] = color1

            # output = stumps_projection(output)
            cv2.addWeighted(output, ALPHA, frame_orig, 1 - ALPHA, 0, frame_orig)

            # Condition to save the frame in the video file
            if SAVE_FLAG: out.write(frame_orig)

            # Displaying the ball mask
            cv2.imshow("Predicted Output", frame_orig)
            key = cv2.waitKey(1)

            if key == ord('q'): exit()
            elif key == ord('n'): break

        # Release the video writer, close the output file, and destroying all CV windows.
        if SAVE_FLAG: out.release()
        cv2.destroyAllWindows()
