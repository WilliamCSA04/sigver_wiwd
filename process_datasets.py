import signet
import sys
import numpy as np
import os
import scipy.io
import classifier
import random
from scipy.misc import imread
from preprocess.normalize import preprocess_signature
from cnn_model import CNNModel
from generate_set import split_into_train_test
from process_helper import validate_train_test
from metrics import average, standard_deviation

datasets_paths = [] 
model_path = "models/signet.pkl" #Always will use this model
#canvas_size = (952, 1360)  # Maximum signature size
canvas_size = (1768, 2176)  # Maximum signature size
canvas = []
dataset = ""
if(len(sys.argv) == 1):
    datasets_paths = ["datasets/MCYT/", "datasets/GPDS160/", "datasets/GPDS300/"]#All datasets needed
else:
    dataset = sys.argv[1]
    datasets_paths = ["datasets/"+ dataset +"/"]

model = CNNModel(signet, model_path)
train_sets = []
test_sets = []
classifications = []
options = []
train_message = []
test_message = []

if(dataset == "MCYT"  or dataset == ""):
    print("Loading MCYT")
    mcyt_path = "datasets/MCYT/"
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
    validate_train_test(mcyt_train_set, mcyt_test_set)
    mcyt_train_classification = mcyt_sets_classification[1]
    train_sets.append(mcyt_train_set)
    test_sets.append(mcyt_test_set)
    train_message.append("Starting preprocess images for train of MCYT")
    test_message.append("Starting preprocess images for test of MCYT")
    classifications.append(mcyt_train_classification)
    options.append([mcyt_forgery_options[1], mcyt_random_options[1]])
    canvas.append((600, 850))
    print("This dataset has for test: genuine samples: " + str(len(mcyt_test_set[0])) + " ,Forgery: " + str(len(mcyt_test_set[1])) + " ,Random: " + str(len(mcyt_test_set[2])))

if(dataset == "GPDS160" or dataset == ""):
    print("Loading GPDS-160")
    gpds_160_path = "datasets/GPDS160/"
    gpds_160_folders = os.listdir(gpds_160_path)
    gpds_160_folders = [folder + "/" for folder in gpds_160_folders]
    #For each array in next line is [number_of_samples_for_train, number_of_samples_for_test]
    gpds_160_genuine_options = [14, 10]
    gpds_160_forgery_options = [0, 10]
    gpds_160_random_options = [14, 10]
    gpds_160_sets = split_into_train_test(gpds_160_folders, gpds_160_path, gpds_160_genuine_options[0], gpds_160_forgery_options[0], gpds_160_random_options[0])
    gpds_160_train_set = gpds_160_sets[0][0]
    gpds_160_test_set = gpds_160_sets[0][1]
    validate_train_test(gpds_160_train_set, gpds_160_test_set)
    train_message.append("Starting preprocess images for train of GPDS160")
    test_message.append("Starting preprocess images for test of GPDS160")
    gpds_160_train_classification = gpds_160_sets[1]
    train_sets.append(gpds_160_train_set)
    test_sets.append(gpds_160_test_set)
    classifications.append(gpds_160_train_classification)
    options.append([gpds_160_forgery_options[1], gpds_160_random_options[1]])
    canvas.append((1768, 2176))
    print("This dataset has for test: genuine samples: " + str(len(gpds_160_test_set[0])) + " ,Forgery: " + str(len(gpds_160_test_set[1])) + " ,Random: " + str(len(gpds_160_test_set[2])))

if(dataset == "GPDS300" or dataset == ""):
    print("Loading GPDS-300")
    gpds_300_path = "datasets/GPDS300/"
    gpds_300_folders = os.listdir(gpds_300_path)
    gpds_300_folders = [folder + "/" for folder in gpds_300_folders]
    #For each array in next line is [number_of_samples_for_train, number_of_samples_for_test]
    gpds_300_genuine_options = [14, 10]
    gpds_300_forgery_options = [0, 10]
    gpds_300_random_options = [14, 10]
    gpds_300_sets = split_into_train_test(gpds_300_folders, gpds_300_path, gpds_300_genuine_options[0], gpds_300_forgery_options[0], gpds_300_random_options[0])
    gpds_300_train_set = gpds_300_sets[0][0]
    gpds_300_test_set = gpds_300_sets[0][1]
    validate_train_test(gpds_300_train_set, gpds_300_test_set)
    train_message.append("Starting preprocess images for train of GPDS300")
    test_message.append("Starting preprocess images for test of GPDS300")
    gpds_300_train_classification = gpds_300_sets[1]
    train_sets.append(gpds_300_train_set)
    test_sets.append(gpds_300_test_set)
    classifications.append(gpds_300_train_classification)
    options.append([gpds_300_forgery_options[1], gpds_300_random_options[1]])
    canvas.append((1768, 2176))
    print("This dataset has for test: genuine samples: " + str(len(gpds_300_test_set[0])) + " ,Forgery: " + str(len(gpds_300_test_set[1])) + " ,Random: " + str(len(gpds_300_test_set[2])))

train_sets_processed = [[],[],[]]
for index, set in enumerate(train_sets):
    print(train_message[index])
    for image in set:
        original = imread(image, flatten=1)
        processed = preprocess_signature(original, canvas[index])
        train_sets_processed[index].append(model.get_feature_vector(processed)[0])

for i, test_set in enumerate(test_sets):
    print(test_message[i])
    genuine_for_test = []
    forgery_for_test = []
    random_for_test = []
    for index, set in enumerate(test_set):
        for image in set:
            original = imread(image, flatten=1)
            processed = preprocess_signature(original, canvas[i])
            if(index == 0):
                feature_vector = model.get_feature_vector(processed)
                genuine_for_test.append(feature_vector[0])
            elif(index == 1):
                feature_vector = model.get_feature_vector(processed)
                forgery_for_test.append(feature_vector[0])
            else:
                random_for_test.append(processed)
    accs_knn = [[], []]
    for j in range(100):
        print("Interation: " + str(j))
        random.shuffle(forgery_for_test)
        random.shuffle(random_for_test)
        #TODO: Check if data are correct
        option = options[i]
        random_signatures_for_test = [model.get_feature_vector(processed)[0] for processed in random_for_test[:option[1]]]
        test = genuine_for_test + forgery_for_test[:option[0]] + random_signatures_for_test
        test_classification = []
        genuine_quantity = len(genuine_for_test)
        for k in range(genuine_quantity):
            test_classification.append(1)
        for k in range(option[0] + option[1]):
            test_classification.append(0)
        classifier.knn(np.array(train_sets_processed[i]), test, classifications[i], test_classification, genuine_quantity, option[0], option[1])

    

