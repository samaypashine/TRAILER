# /**
#  * @file inference_deeplab.py 
#  * @author Samay Pashine
#  * @updated Samay Pashine
#  * @brief Code to track the ball in the frames on video. 
#  * @version 3.5
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

# Function to detect and filter stumps contour, and project the line between them.
def stumps_projection(img, AREA_THRESH = 250, color=(255, 0, 0)):
    # Processing the image to detect contours.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw a bounding box around each contour
    llist = list()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > AREA_THRESH:
            llist.append((x, y, w, h))
            if len(llist) == 2:
                cv2.line(img, (llist[0][0], llist[0][1] + llist[0][3]), (llist[1][0], llist[1][1] + llist[1][3]), color=(255, 0, 0), thickness=1)
                cv2.line(img, (llist[0][0] + llist[0][2], llist[0][1] + llist[0][3]), (llist[1][0] + llist[1][2], llist[1][1] + llist[1][3]), color=(255, 0, 0), thickness=1)

                points = np.array([(llist[1][0], llist[1][1] + llist[1][3]),
                                   (llist[0][0], llist[0][1] + llist[0][3]),
                                   (llist[0][0] + llist[0][2], llist[0][1] + llist[0][3]),
                                   (llist[1][0] + llist[1][2], llist[1][1] + llist[1][3])], dtype=np.int32)
                cv2.fillPoly(img, [points], color, lineType=4)
                break
    return img


if __name__ == '__main__':
    # Configuring the log handler.
    log_path = r'logs/inference-deeplab' + os.sep + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.log'
    os.makedirs(r'logs/inference-deeplab', exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
                        handlers=[logging.FileHandler(log_path), logging.StreamHandler()])

    # Enabling the memory growth in the gpu if available and manually configuring the gpu memory limits.
    try:
        tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
        gpu = tf.config.experimental.list_physical_devices('GPU')[0]
    except Exception as e:
        logging.warn(f'Exception: {e}')

    # Initializing the argument parser.
    parser = argparse.ArgumentParser(description='Script to infer on the data using the deeplab keras ball tracking model.')
    parser.add_argument('-d', '--data', default='./raw_data/videos', help='Path to dataset.')
    parser.add_argument('-m', '--model', required=True, help='Path to saved model.')
    parser.add_argument('-a', '--alpha', default=0.5, type=float, help='Overlay alpha')
    parser.add_argument('-c', '--classes', default=2, type=int, help='Number of classes')
    parser.add_argument('-t', '--area-thresh', default=250, type=int, help='Stumps contour area threshold.')
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
    AREA_THRESH = args.area_thresh
    FPS_HIST = list()

    logging.info(f'DATA DIRECTORY       :       {DATA_DIR}')
    logging.info(f'MODEL PATH           :       {MODEL_PATH}')
    logging.info(f'IMAGE SIZE           :       {IMAGE_SIZE}')
    logging.info(f'ALPHA                :       {ALPHA}')
    logging.info(f'FRAME RATE           :       {FRAME_RATE}')
    logging.info(f'AREA THRESH          :       {AREA_THRESH}')
    
    # Checking if data directory and model file exists or not.
    assert os.path.isdir(DATA_DIR)
    assert os.path.isfile(MODEL_PATH)

    # Loading the model.
    logging.info("Loading the model ...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Set the color for the object in the mask (e.g., red)
    color_dict = {1: (0, 0, 255), 2: (255, 0, 0)}

    # Iterating over the video files.
    for video_file in os.listdir(DATA_DIR):
        # Opening the video file
        logging.info(f"Video File       :       {video_file}")
        cap = cv2.VideoCapture(os.path.join(DATA_DIR, video_file))

        if SAVE_FLAG:
            # Create a VideoWriter object to save the video
            os.makedirs(rf'{os.path.join("DeepLab Inference Videos")}', exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(os.path.join("./DeepLab Inference Videos", video_file), fourcc, FRAME_RATE, IMAGE_SIZE)

        # Create a blank image with the same dimensions as the mask
        height, width = IMAGE_SIZE
        output = np.zeros((height, width, 3), dtype=np.uint8)
        stump_output = np.zeros((height, width, 3), dtype=np.uint8)
        while True:
            # Reading the frame
            ret, frame = cap.read()

            # Condition for empty frame
            if not ret: break

            # Processing the frame
            frame_orig = cv2.resize(frame, IMAGE_SIZE)
            frame = cv2.resize(frame, IMAGE_SIZE)
            frame = tf.keras.applications.resnet50.preprocess_input(frame)
            frame = tf.expand_dims(frame, axis=0)
            
            # Predicting the ball mask
            s_t = time.time()
            prediction = model.predict(frame, verbose=0)
            FPS_HIST.append(60 / (time.time() - s_t))
            prediction = np.squeeze(prediction)
            prediction = np.argmax(prediction, axis=2)
            logging.info(f'Average FPS       :       {sum(FPS_HIST) / len(FPS_HIST)}')

            # Apply the mask to the output image to highlight the object
            pixel_values = np.unique(prediction)
            
            # Assigning the color to pixel values.
            output[prediction == 1] = color_dict[1]
            output[prediction == 2] = color_dict[2]
            stump_output[prediction == 2] = color_dict[2]

            cv2.addWeighted(output, ALPHA, frame_orig, 1 - ALPHA, 0, frame_orig)
            if stump_output.any():
                stump_output = stumps_projection(stump_output, AREA_THRESH, color_dict[1])
                cv2.addWeighted(stump_output, ALPHA, frame_orig, 1 - ALPHA, 0, frame_orig)

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
