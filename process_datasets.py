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
#For each array in next line is [number_of_samples_for_train, number_of_samples_for_test]
mcyt_genuine_options=[10, 5]
mcyt_forgery_options=[0, 15]
mcyt_random_options=[10, 0]
mcyt_sets_classification = split_into_train_test(mcyt_folders, mcyt_path, mcyt_genuine_options[0], mcyt_forgery_options[0], mcyt_random_options[0])
mcyt_sets = mcyt_sets_classification[0]
mcyt_train_set = mcyt_sets[0]
mcyt_test_set = mcyt_sets[1]
mcyt_train_classification = mcyt_sets_classification[1]

print("Loading GPDS-160")
gpds_160_path = datasets_paths[1]
gpds_160_folders = os.listdir(gpds_160_path)
gpds_160_folders = [folder + "/" for folder in gpds_160_folders]
#For each array in next line is [number_of_samples_for_train, number_of_samples_for_test]
gpds_160_genuine_options = [14, 10]
gpds_160_forgery_options = [0, 10]
gpds_160_random_options = [14, 10]
gpds_160_sets = split_into_train_test(gpds_160_folders, gpds_160_path, gpds_160_genuine_options[0], gpds_160_forgery_options[0], gpds_160_random_options[0])
gpds_160_train_set = gpds_160_sets[0][0]
gpds_160_test_set = gpds_160_sets[0][1]
gpds_160_train_classification = gpds_160_sets[1]

print("Loading GPDS-300")
gpds_300_path = datasets_paths[2]
gpds_300_folders = os.listdir(gpds_300_path)
gpds_300_folders = [folder + "/" for folder in gpds_300_folders]
#For each array in next line is [number_of_samples_for_train, number_of_samples_for_test]
gpds_300_genuine_options = [14, 10]
gpds_300_forgery_options = [0, 10]
gpds_300_random_options = [14, 10]
gpds_300_sets = split_into_train_test(gpds_300_folders, gpds_300_path, gpds_300_forgery_options[0], gpds_300_forgery_options[0], gpds_300_random_options[0])
gpds_300_train_set = gpds_300_sets[0][0]
gpds_300_test_set = gpds_300_sets[0][1]
gpds_300_train_classification = gpds_300_sets[1]


train_sets = [mcyt_train_set, gpds_160_train_set, gpds_300_train_set]
train_sets_processed = [[],[],[]]
for index, set in enumerate(train_sets):
    if(index == 0):
        print("Starting preprocess images for train of MCYT")
    elif(index == 1):
        print("Starting preprocess images for train of GPDS-160")
    else:
        print("Starting preprocess images for train of GPDS-300")
    for image in set:
        original = imread(image, flatten=1)
        processed = preprocess_signature(original, canvas_size)
        train_sets_processed[index].append(model.get_feature_vector(processed)[0])

classifications = [mcyt_train_classification, gpds_160_train_classification, gpds_300_train_classification]
options = [mcyt_forgery_options[1] + mcyt_random_options[1], gpds_160_forgery_options[1] + gpds_160_random_options[1], gpds_300_forgery_options[1] + gpds_300_random_options[1]]
test_sets = [mcyt_test_set, gpds_160_test_set, gpds_300_test_set]
for i, test_set in enumerate(test_sets):
    if(i == 0):
        print("Starting preprocess images for test of MCYT")
    elif(i == 1):
        print("Starting preprocess images for test of GPDS-160")
    else:
        print("Starting preprocess images for test of GPDS-300")
    genuine_for_test = []
    forgery_for_test = []
    random_for_test = []
    for index, set in enumerate(test_set):
        for image in set:
            original = imread(image, flatten=1)
            processed = preprocess_signature(original, canvas_size)
            feature_vector = model.get_feature_vector(processed)
            if(index == 0):
                genuine_for_test.append(feature_vector[0])
            elif(index == 1):
                forgery_for_test.append(feature_vector[0])
            else:
                random_for_test.append(feature_vector[0])


    for j in range(100):
        print("Interation: " + str(j))
        random.shuffle(forgery_for_test)
        random.shuffle(random_for_test)
        #TODO: Check if data are correct
        option = options[i]
        test = genuine_for_test + forgery_for_test[:option]
        test_classification = []
        for k in range(len(genuine_for_test)):
            test_classification.append(1)
        for k in range(option):
            test_classification.append(0)
        classifier.knn(np.array(train_sets_processed[i]), test, classifications[i], test_classification, k=3)

