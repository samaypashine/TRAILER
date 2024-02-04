# /**
#  * @file download_dataset.py 
#  * @author Samay Pashine
#  * @updated Samay Pashine
#  * @brief Code to download the dataset from segments.ai.
#  * @version 1.0
#  * @date 2023-11-14
#  * @copyright Copyright (c) 2023
#  */

# Importing necessary libraries
import os
import logging
import argparse
from datetime import datetime
from segments import SegmentsClient, SegmentsDataset
from segments.utils import export_dataset

if __name__ == '__main__':
    # Configuring the log handler.
    log_path = r'logs/train' + os.sep + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.log'
    os.makedirs(r'logs/train', exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
                        handlers=[logging.FileHandler(log_path), logging.StreamHandler()])

    # Initializing the argument parser.
    parser = argparse.ArgumentParser(description='Script to download dataset from segment.ai')
    parser.add_argument('-t', '--token', default='cbbed6daed80f511f795c566ef73feaced0aabe8', help='Client token from the user account.')
    parser.add_argument('-d', '--dataset', default='rahulanon12345/DRS', help='Dataset to download.')
    parser.add_argument('-r', '--release', default='v1.0', help='Specific release of the dataset to download')
    parser.add_argument('-f', '--format', default="coco-panoptic", help='Format in which dataset should be downloaded.')
    parser.add_argument('-o', '--output', default="./dataset", help='Export the dataset to the specific folder.')
    args = parser.parse_args()

    # Initializing the variables
    TOKEN = args.token
    DATASET = args.dataset
    RELEASE = args.release
    FORMAT = args.format
    OUTPUT_PATH = args.output

    logging.info(f'DATASET          :   {DATASET}')
    logging.info(f'RELEASE          :   {RELEASE}')
    logging.info(f'FORMAT           :   {FORMAT}')
    logging.info(f'OUTPUT           :   {OUTPUT_PATH}')

    # Creating new folder for dataset.
    os.makedirs(rf'{OUTPUT_PATH}', exist_ok=True)
    
    # Initialize a SegmentsDataset from the release file.
    logging.info(f'Initialzing the connection to segments.ai')
    client = SegmentsClient(TOKEN)
    release = client.get_release(DATASET, RELEASE)
    dataset = SegmentsDataset(release, labelset='ground-truth', filter_by=['labeled', 'reviewed'])

    # Export the dataset in a specific format.
    logging.info(f'####################### Starting the export #######################')
    export_dataset(dataset, export_format=FORMAT, export_folder=OUTPUT_PATH)
    logging.info(f'######################### Export Finished #########################')
