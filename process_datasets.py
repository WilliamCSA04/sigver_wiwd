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
mcyt_folders = [folder + "/" for folder in mcyt_folders]
mcyt_genuine_options=[10, 5]
mcyt_forgery_options=[0, 15]
mcyt_random_options=[10, 0]
mcyt_sets_classification = split_into_train_test(mcyt_folders, mcyt_path, mcyt_genuine_options, mcyt_forgery_options, mcyt_random_options)
mcyt_sets = mcyt_sets_classification[0]
mcyt_classification = mcyt_sets_classification[1]

print("Loading GPDS-160")
gpds_160_path = datasets_paths[1]
gpds_160_folders = os.listdir(gpds_160_path)
gpds_160_folders = [folder + "/" for folder in gpds_160_folders]
gpds_160_sets = split_into_train_test(gpds_160_folders, gpds_160_path, [14, 10], [0, 10], [14, 10])
print("Loading GPDS-300")
gpds_300_path = datasets_paths[2]
gpds_300_folders = os.listdir(gpds_300_path)
gpds_300_folders = [folder + "/" for folder in gpds_300_folders]
#For each array in next line is [number_of_samples_for_train, number_of_samples_for_test]
gpds_300_sets = split_into_train_test(gpds_300_folders, gpds_300_path, [14, 10], [0, 10], [14, 10])

print("Starting preprocess images for train of MCYT")
mcyt_train = []
for image in mcyt_sets[0]:
    original = imread(image, flatten=1)
    processed = preprocess_signature(original, canvas_size)
    mcyt_train.append(model.get_feature_vector(processed)[0])

mcyt_train = np.array(mcyt_train)
print("Dataset for mcyt_train: " + str(len(mcyt_train)) + " samples")

print("Starting preprocess images for test of MCYT")
mcyt_genuine_for_test = []
mcyt_forgery_for_test = []
mcyt_random_for_test = []
for index, set in enumerate(mcyt_sets[1]):
    for image in set:
        original = imread(image, flatten=1)
        processed = preprocess_signature(original, canvas_size)
        feature_vector = model.get_feature_vector(processed)
        if(index == 0):
            mcyt_genuine_for_test.append(feature_vector[0])
        elif(index == 1):
            mcyt_forgery_for_test.append(feature_vector[0])
        else:
            mcyt_random_for_test.append(feature_vector[0])

mcyt_test = mcyt_genuine_for_test + mcyt_forgery_for_test[:mcyt_forgery_options[1]] + mcyt_random_for_test[:mcyt_random_options[1]]
mcyt_test = np.array(mcyt_test)
print("Dataset for mcyt_test: " + str(len(mcyt_test)) + " samples")

classifier.knn(mcyt_train, mcyt_test, mcyt_classification[0], mcyt_classification[1])

