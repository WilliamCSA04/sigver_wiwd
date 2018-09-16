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
from generate_set import split_into_train_test

datasets_paths = ["datasets/MCYT/", "datasets/GPDS160/", "datasets/GPDS300/"] #All datasets needed
model_path = "models/signet.pkl" #Always will use this model
canvas_size = (952, 1360)  # Maximum signature size

model = CNNModel(signet, model_path)

print("Loading MCYT-75")
mcyt_path = datasets_paths[0]
mcyt_folders = os.listdir(datasets_paths[0])
mcyt_sets = split_into_train_test(mcyt_folders, mcyt_path, [10, 5], [0, 15], [10, 0])
print("Loading GPDS-160")
gpds_160_folders = os.listdir(datasets_paths[1])
mcyt_sets = split_into_train_test(mcyt_folders, mcyt_path, [10, 5], [0, 15], [10, 0])
print("Loading GPDS-300")
gpds_300_folders = os.listdir(datasets_paths[2])
mcyt_sets = split_into_train_test(mcyt_folders, mcyt_path, [10, 5], [0, 15], [10, 0])
