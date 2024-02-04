# Tracking and Recognition AI for LBW Evaluation and Review

This codebases focuses on tracking the ball and stumps with the projection line between stumps to help umpires in critical decisions regarding gameplay. It utilizes the power of neural nets to find the segmentation mask and then process those masks using OpenCV to project line between wicket to assist users..

## Prerequisites

- Python 3.10.13
- OpenCV 4.8.0.74
- NumPy 1.24.4
- tensorflow 2.14.0
- tensorflow-datasets 4.9.3
- tensorflow-estimator 2.14.0
- tensorflow-io-gcs-filesystem 0.34.0
- tensorflow-metadata 1.14.0
- keras 2.14.0
- opencv-python 4.8.1
- imutils 0.5.4
- matplotlib 3.7.3

## Installation

1. Create a conda environment to run the codebase:

   ```bash
   conda create -n aiCrik python=3.10.13
   ```

2. Activate the conda environment after creation:

    ```bash
   conda activate aiCrik
   ```

3. Install the dependencies
    
    ```bash
    pip install -r requirements.txt
    ```

4. To train a certain network, run the script train_{choice_of_network}.py

    ```bash
    python3 train_{option}.py --help (To look at the arguments to pass for training.)
    ```

5. To infer using a certain model, run the script inference_{choice_of_model - type}.py

    ```bash
    python3 inference_{option}_{type}.py --help (To look at the arguments to pass for inference.)
    ```

