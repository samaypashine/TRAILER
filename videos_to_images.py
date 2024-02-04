# /**
#  * @file videos_to_images.py 
#  * @author Samay Pashine
#  * @updated Samay Pashine
#  * @brief Code to traverse video files and save the frames. 
#  * @version 1.0
#  * @date 2023-10-23
#  * 
#  * @copyright Copyright (c) 2023
#  * 
#  */

# Importing the necessary libraries
import os
import cv2
import imutils
import logging
import argparse
import numpy as np
from datetime import datetime


if __name__ == '__main__':
    # Initializing the argument parser.
    parser = argparse.ArgumentParser(description='Script to convert videos to images.')
    parser.add_argument('-v', '--videos', default='./raw_data/videos', help='Path to videos directory')
    parser.add_argument('-n', '--number-format', default=6, help='total numbers for formatting.')
    parser.add_argument('-o', '--output', default='./raw_data', help='Path to output directory to save the masks.')
    parser.add_argument('--debug', default=False, help='Debug mode flag')
    args = parser.parse_args()

    videos_basepath = args.videos
    frame_dir = os.sep.join([args.output, "images"])

    # Verifying the directory structure existence.
    os.makedirs(rf'{args.videos}', exist_ok=True)
    os.makedirs(rf'{frame_dir}', exist_ok=True)

    # Configuring the log handler.
    log_path = r'logs' + os.sep + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.log'
    os.makedirs(r'logs', exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
                        handlers=[logging.FileHandler(log_path), logging.StreamHandler()])

    # Initializing the filenum for both ball masks and not-ball masks.
    frame_filenum = len(os.listdir(frame_dir)) + 1
    logging.info(f'Frame File Num Count        :   {frame_filenum - 1}')

    # Loop to go through every video file in the directory given.
    video_filenum = 1
    for filename in os.listdir(videos_basepath):
        # Base condition to check the file to be video file.
        if os.path.splitext(filename)[1].lower() not in ['.mp4', '.mkv', '.avi', '.mov']:
            logging.warn(f'File : {filename} IS NOT A VIDEO FILE')
            logging.warn(f'Ext  : {os.path.splitext(filename)[1].lower()}')
            continue

        # Initializing the video file to play.
        filepath = os.sep.join([videos_basepath, filename])
        logging.info(f'Video {video_filenum} Filepath                   :   {filepath}')
        logging.info('######################################################################################')
        cap = cv2.VideoCapture(filepath)

        while(True):
            # Capture the next frame
            ret, frame = cap.read()
            flag = False

            # If there are no more frames, break out of the loop
            if not ret: break

            # Saving the frames accordingly.
            frame_copy = imutils.resize(frame, width=600)
            cv2.imshow('Frame', frame_copy)
            cv2.waitKey(1)
            cv2.imwrite(f"{os.sep.join([frame_dir, str(frame_filenum).zfill(args.number_format)])}.jpg", frame_copy)
            logging.info(f'{frame_filenum} Frame Saved At   : {os.sep.join([frame_dir, str(frame_filenum).zfill(args.number_format)])}.jpg')
            frame_filenum += 1

        video_filenum += 1
        cap.release()
        cv2.destroyAllWindows()
