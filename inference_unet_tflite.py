# /**
#  * @file inference_unet_tflite.py 
#  * @author Samay Pashine
#  * @updated Samay Pashine
#  * @brief Code to track the ball in the frames of video using UNET model. 
#  * @version 2.1
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

if __name__ == '__main__':
    # Configuring the log handler.
    log_path = r'logs/inference-unet-tflite' + os.sep + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.log'
    os.makedirs(r'logs/inference-unet-tflite', exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
                        handlers=[logging.FileHandler(log_path), logging.StreamHandler()])

    # Enabling the memory growth in the gpu if available and manually configuring the gpu memory limits.
    try:
        tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
        gpu = tf.config.experimental.list_physical_devices('GPU')[0]
    except Exception as e: logging.error(f'Exception: {e}')

    # Initializing the argument parser.
    parser = argparse.ArgumentParser(description='Script to train and save the ball tracking model.')
    parser.add_argument('-d', '--data', default='./raw_data/videos', help='Path to dataset.')
    parser.add_argument('-m', '--model', required=True, help='Path to saved model.')
    parser.add_argument('-a', '--alpha', default=0.5, type=float, help='Overlay alpha')
    parser.add_argument('-t', '--threshold', default=0.9, type=float, help='Confidence threshold value of prediction to filter mask')
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
    THRESHOLD = args.threshold
    FPS_HIST = list()

    logging.info(f'DATA DIRECTORY           :           {DATA_DIR}')
    logging.info(f'MODEL PATH               :           {MODEL_PATH}')
    logging.info(f'IMAGE SIZE               :           {IMAGE_SIZE}')
    logging.info(f'FRAME RATE               :           {FRAME_RATE}')
    logging.info(f'THRESHOLD                :           {THRESHOLD}')
    logging.info(f'ALPHA                    :           {ALPHA}')

    # Checking if data directory and model file exists or not.
    assert os.path.isdir(DATA_DIR)
    assert os.path.isfile(MODEL_PATH)

    # Loading the model.
    logging.info('Loading the model ...')
    model = tf.lite.Interpreter(MODEL_PATH)

    # Allocate tensors
    model.allocate_tensors()

    # Get input and output tensors
    input_details = model.get_input_details()[0]
    output_details = model.get_output_details()[0]

    # Iterating over the video files.
    for video_file in os.listdir(DATA_DIR):
        
        # Opening the video file
        logging.info(f'VIDEO FILE                    :           {video_file}')
        cap = cv2.VideoCapture(os.path.join(DATA_DIR, video_file))

        if SAVE_FLAG:
            # Create a VideoWriter object to save the video
            os.makedirs(rf'{os.path.join("Saved Videos")}', exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(os.path.join("./Saved Videos", video_file), fourcc, FRAME_RATE, IMAGE_SIZE)

        # Create a blank image with the same dimensions as the mask
        height, width = IMAGE_SIZE
        output = np.zeros((height, width, 3), dtype=np.uint8)
        while True:
            # Reading the frame
            ret, frame = cap.read()

            # Condition for empty frame
            if not ret: break

            # Processing the frame
            frame_orig = cv2.resize(frame, IMAGE_SIZE)
            frame = cv2.resize(frame, IMAGE_SIZE).astype(np.float32)
            frame = tf.expand_dims(frame, axis=0)
            
            # Set the input tensor
            model.set_tensor(input_details['index'], frame)

            # Run inference
            s_t = time.time()
            model.invoke()
            FPS_HIST.append(60 / (time.time() - s_t))

            # Get the output tensor
            output_tensor = model.get_tensor(output_details['index'])
            prediction = output_tensor.astype(float) / output_tensor.max()
            prediction = np.squeeze(prediction)
            logging.info(f'Average FPS       :       {sum(FPS_HIST) / len(FPS_HIST)}')

            # Set the color for the object in the mask (e.g., red)
            color = (0, 0, 255) # Red

            # Apply the mask to the output image to highlight the object
            output[prediction > THRESHOLD] = color
            cv2.addWeighted(output, ALPHA, frame_orig, 1 - ALPHA, 0, frame_orig)

            # Condition to save the frame in the video file
            if SAVE_FLAG: out.write(frame_orig)

            # Displaying the ball mask
            cv2.imshow("Predicted Frame", frame_orig)
            key = cv2.waitKey(1)

            if key == ord('q'): exit()
            elif key == ord('n'): break
        
        # Release the video writer, close the output file, and destroying all CV windows.
        if SAVE_FLAG: out.release()
        cv2.destroyAllWindows() 
