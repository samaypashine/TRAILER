# /**
#  * @file empty_mask_check.py 
#  * @author Samay Pashine
#  * @updated Samay Pashine
#  * @brief Code to filter and remove the empty masks. 
#  * @version 1.0
#  * @date 2023-10-24
#  * 
#  * @copyright Copyright (c) 2023
#  * 
#  */

# Importing necessary libraries.
import os
import cv2
import logging
import argparse
import numpy as np
from datetime import datetime

# Function to check if the image is completely black i.e. blank mask.
def is_image_completely_black(image):
  """Checks if an image is completely black.

  Args:
    image: A NumPy array representing the image.

  Returns:
    True if the image is completely black, False otherwise.
  """
  # Check if all pixels in the image are black.
  return np.all(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) == 0)


if __name__ == '__main__':
    # Initializing the argument parser.
    parser = argparse.ArgumentParser(description='Script to delete empty masks.')
    parser.add_argument('-i', '--images', required=True, help='Path to masks images directory')
    parser.add_argument('-r', '--remove', action='store_true', help='Flag to delete the mask files.')
    parser.add_argument('--debug', action='store_true', help='Debug mode flag')
    args = parser.parse_args()

    # Configuring the log handler.
    log_path = r'logs' + os.sep + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.log'
    os.makedirs(r'logs', exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
                        handlers=[logging.FileHandler(log_path), logging.StreamHandler()])

    # Intiializing the variables.
    mask_basepath = args.images
    img_basepath = os.sep.join(mask_basepath.split(os.sep)[:-1] + ['images'])
    removeFlag = args.remove
    debugFlag = args.debug
    count = 0

    logging.info(f'Mask Path          :   {mask_basepath}')
    logging.info('#####################################################################')

    # Traversing the Image file in the directory.
    for filename in os.listdir(mask_basepath):
        filepath = os.sep.join([mask_basepath, filename])
        img_filepath = os.sep.join([img_basepath, filename.replace('_mask', '')]).replace('png', 'jpg')
        try:
            # Reading the image.
            img = cv2.imread(filepath)
            
            # Checking if the image is blank (completely black).
            if is_image_completely_black(img):
              count += 1
              logging.info(f'Empty File Found :  {filepath},    Total Count : {count}')
              if removeFlag: os.system(f'rm -rf {filepath} {img_filepath}')

        except Exception as e:
            if debugFlag:
                logging.error(f'Exception : {e}')