from scipy.misc import imread
from preprocess.normalize import preprocess_signature
import signet
from cnn_model import CNNModel
import numpy as np
import sys
import os
import scipy.io
import classifier
import random
from process_helper import filter_array_of_folders

datasets_paths = ["datasets/MCYT/", "datasets/GPDS160/", "datasets/GPDS300/"] #All datasets needed
model_path = "models/signet.pkl" #Always will use this model
canvas_size = (952, 1360)  # Maximum signature size

model = CNNModel(signet, model_path)

print("Loading MCYT-75")
mcyt_folders = os.listdir(datasets_paths[0])
mcyt_folders = filter_array_of_folders(mcyt_folders, datasets_paths[0]) #Remove files other than signatures
print("Loading GPDS-160")
gpds_160_folders = os.listdir(datasets_paths[1])
gpds_160_folders = filter_array_of_folders(gpds_160_folders, datasets_paths[1]) #Remove files other than signatures
print("Loading GPDS-300")
gpds_300_folders = os.listdir(datasets_paths[2])
gpds_300_folders = filter_array_of_folders(gpds_300_folders, datasets_paths[2]) #Remove files other than signatures
