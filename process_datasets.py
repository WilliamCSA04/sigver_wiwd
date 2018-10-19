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

dataset = ""

frr_metrics_local = []
far_skilled_local = []
far_random_local = []
eer_local = []

frr_metrics_global = []
far_skilled_metrics_global = []
far_random_metrics_global = []
eer_metrics_global = []

frr_metrics_local_sd = []
far_skilled_local_sd = []
far_random_local_sd = []
eer_local_sd = []

frr_metrics_global_sd = []
far_skilled_metrics_global_sd = []
far_random_metrics_global_sd = []
eer_metrics_global_sd = []

auc_metrics = []

number_of_interations = 1
if(len(sys.argv) == 1):
    datasets_paths = ["datasets/MCYT/", "datasets/GPDS160/", "datasets/GPDS300/"]#All datasets needed
else:
    dataset = sys.argv[1]
    number_of_interations = int(sys.argv[2])
    datasets_paths = ["datasets/"+ dataset +"/"]

model = CNNModel(signet, model_path)

mcyt_path = "datasets/MCYT/"
mcyt_folders = os.listdir(datasets_paths[0])
mcyt_folders = [folder + "/" for folder in mcyt_folders]
mcyt_genuine_candidates = list(mcyt_folders)

gpds_160_path = "datasets/GPDS160/"
gpds_160_folders = os.listdir(gpds_160_path)
gpds_160_folders = [folder + "/" for folder in gpds_160_folders]
gpds_160_genuine_candidates = list(gpds_160_folders)

gpds_300_path = "datasets/GPDS300/"
gpds_300_folders = os.listdir(gpds_300_path)
gpds_300_folders = [folder + "/" for folder in gpds_300_folders]
gpds_300_genuine_candidates = list(gpds_300_folders)

gpds_50_path = "datasets/GPDS50/"
gpds_50_folders = os.listdir(gpds_50_path)
gpds_50_folders = [folder + "/" for folder in gpds_50_folders]
gpds_50_genuine_candidates = list(gpds_50_folders)

images_dictionary = {}

def add_feature_vector_from_a_image(image, canvas, sets_processed):
    if image in images_dictionary.keys():
        sets_processed.append(images_dictionary[image])
    else:
        original = imread(image, flatten=1)
        processed = preprocess_signature(original, canvas)
        images_dictionary[image] = model.get_feature_vector(processed)[0]
        sets_processed.append(images_dictionary[image])

for number_of_interation in range(number_of_interations):
    train_sets = []
    test_sets = []
    classifications = []
    options = []
    train_message = []
    test_message = []
    svm_weights = []
    canvas = []
    print("Number of interation:" + str(number_of_interation))
    if(dataset == "MCYT"  or dataset == ""):
        print("Loading MCYT")
        #For each array in next line is [number_of_samples_for_train, number_of_samples_for_test]
        mcyt_genuine_options=[10, 5]
        mcyt_forgery_options=[0, 15]
        mcyt_random_options=[10, 0]
        mcyt_sets_classification = split_into_train_test(mcyt_folders, mcyt_genuine_candidates.pop(), mcyt_path, mcyt_genuine_options[0], mcyt_forgery_options[0], mcyt_random_options[0])
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
        svm_genuine_weight = (mcyt_random_options[0]*74)/mcyt_genuine_options[0]
        svm_weights.append({0: 1, 1: svm_genuine_weight})
        print("This dataset has for test: genuine samples: " + str(len(mcyt_test_set[0])) + " ,Forgery: " + str(len(mcyt_test_set[1])) + " ,Random: " + str(len(mcyt_test_set[2])))
    if(dataset == "GPDS160" or dataset == ""):
        print("Loading GPDS-160")
        gpds_160_genuine_options = [14, 10]
        gpds_160_forgery_options = [0, 10]
        gpds_160_random_options = [14, 10]
        gpds_160_sets = split_into_train_test(gpds_160_folders, gpds_160_genuine_candidates.pop(), gpds_160_path, gpds_160_genuine_options[0], gpds_160_forgery_options[0], gpds_160_random_options[0])
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
        svm_genuine_weight = (gpds_160_random_options[0]*159)/gpds_160_genuine_options[0]
        svm_weights.append({0: 1, 1: svm_genuine_weight})
        print("This dataset has for test: genuine samples: " + str(len(gpds_160_test_set[0])) + " ,Forgery: " + str(len(gpds_160_test_set[1])) + " ,Random: " + str(len(gpds_160_test_set[2])))

    if(dataset == "GPDS300" or dataset == ""):
        print("Loading GPDS-300")
        gpds_300_genuine_options = [14, 10]
        gpds_300_forgery_options = [0, 10]
        gpds_300_random_options = [14, 10]
        gpds_300_sets = split_into_train_test(gpds_300_folders, gpds_300_genuine_candidates.pop(), gpds_300_path, gpds_300_genuine_options[0], gpds_300_forgery_options[0], gpds_300_random_options[0])
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
        svm_genuine_weight = (gpds_300_random_options[0]*299)/gpds_300_genuine_options[0]
        svm_weights.append({0: 1, 1: svm_genuine_weight})
        print("This dataset has for test: genuine samples: " + str(len(gpds_300_test_set[0])) + " ,Forgery: " + str(len(gpds_300_test_set[1])) + " ,Random: " + str(len(gpds_300_test_set[2])))

    if(dataset == "GPDS50"):
        print("Loading GPDS-50")
        gpds_50_genuine_options = [14, 10]
        gpds_50_forgery_options = [0, 10]
        gpds_50_random_options = [14, 10]
        gpds_50_sets = split_into_train_test(gpds_50_folders, gpds_50_genuine_candidates.pop(), gpds_50_path, gpds_50_genuine_options[0], gpds_50_forgery_options[0], gpds_50_random_options[0])
        gpds_50_train_set = gpds_50_sets[0][0]
        gpds_50_test_set = gpds_50_sets[0][1]
        validate_train_test(gpds_50_train_set, gpds_50_test_set)
        train_message.append("Starting preprocess images for train of GPDS50")
        test_message.append("Starting preprocess images for test of GPDS50")
        gpds_50_train_classification = gpds_50_sets[1]
        train_sets.append(gpds_50_train_set)
        test_sets.append(gpds_50_test_set)
        classifications.append(gpds_50_train_classification)
        options.append([gpds_50_forgery_options[1], gpds_50_random_options[1]])
        canvas.append((1768, 2176))
        svm_genuine_weight = (gpds_50_random_options[0]*299)/gpds_50_genuine_options[0]
        svm_weights.append({0: 1, 1: svm_genuine_weight})
        print("This dataset has for test: genuine samples: " + str(len(gpds_50_test_set[0])) + " ,Forgery: " + str(len(gpds_50_test_set[1])) + " ,Random: " + str(len(gpds_50_test_set[2])))

    train_sets_processed = [[],[],[]]
    for index, set in enumerate(train_sets):
        print(train_message[index])
        for image in set:
            add_feature_vector_from_a_image(image, canvas[index], train_sets_processed[index])

    for i, test_set in enumerate(test_sets):
        print(test_message[i])
        genuine_for_test = []
        forgery_for_test = []
        random_for_test = []
        for index, set in enumerate(test_set):
            for image in set:
                if(index == 0):
                    add_feature_vector_from_a_image(image, canvas[i], genuine_for_test)
                elif(index == 1):
                    add_feature_vector_from_a_image(image, canvas[i], forgery_for_test)
                else:
                    add_feature_vector_from_a_image(image, canvas[i], random_for_test)

        frr_metrics = [[],[],[],[]]
        far_skilled_metrics = [[],[],[],[]]
        far_random_metrics = [[],[],[],[]]
        eer_metrics = [[],[],[],[]]
        print("Starting classification")
        for j in range(100):
            random.shuffle(forgery_for_test)
            random.shuffle(random_for_test)
            option = options[i]
            test = genuine_for_test + forgery_for_test[:option[0]] + random_for_test[:option[1]]
            test_classification = []
            genuine_quantity = len(genuine_for_test)
            for k in range(genuine_quantity):
                test_classification.append(1)
            for k in range(option[0] + option[1]):
                test_classification.append(0)
          #  metrics = classifier.knn(np.array(train_sets_processed[i]), test, classifications[i], test_classification, genuine_quantity, option[0], option[1])
          #  frr_metrics[0].append(metrics[0])
          #  far_skilled_metrics[0].append(metrics[1])
          #  far_random_metrics[0].append(metrics[2])
          #  eer_metrics[0].append(metrics[3])

          #  metrics = classifier.tree(np.array(train_sets_processed[i]), test, classifications[i], test_classification, genuine_quantity, option[0], option[1])
          #  frr_metrics[1].append(metrics[0])
          #  far_skilled_metrics[1].append(metrics[1])
          #  far_random_metrics[1].append(metrics[2])
          #  eer_metrics[1].append(metrics[3])

            metrics = classifier.svm(np.array(train_sets_processed[i]), test, classifications[i], test_classification, genuine_quantity, option[0], option[1], weights=svm_weights[i])
            frr_metrics[2].append(metrics[0])
            far_skilled_metrics[2].append(metrics[1])
            far_random_metrics[2].append(metrics[2])
            eer_metrics[2].append(metrics[3])

            frr_metrics_global.append(metrics[4])
            far_skilled_metrics_global.append(metrics[5])
            far_random_metrics_global.append(metrics[6])
            eer_metrics_global.append(metrics[7])

            frr_metrics_global_sd.append(metrics[4])
            far_skilled_metrics_global_sd.append(metrics[5])
            far_random_metrics_global_sd.append(metrics[6])
            eer_metrics_global_sd.append(metrics[7])

            auc_metrics.append(metrics[8])

           # metrics = classifier.mlp(np.array(train_sets_processed[i]), test, classifications[i], test_classification, genuine_quantity, option[0], option[1])
           # frr_metrics[3].append(metrics[0])
           # far_skilled_metrics[3].append(metrics[1])
           # far_random_metrics[3].append(metrics[2])
           # eer_metrics[3].append(metrics[3])
        print("results")
        for p in range(4):
            types = ["KNN", "Tree", "SVM", "MLP"]
            if(types[p] == "SVM"):

                frr_avg = average(frr_metrics[p])
                far_skilled_avg = average(far_skilled_metrics[p])
                far_random_avg = average(far_random_metrics[p])
                eer_avg = average(eer_metrics[p])

                frr_sd = standard_deviation(frr_metrics[p])
                far_skilled_sd = standard_deviation(far_skilled_metrics[p])
                far_random_sd = standard_deviation(far_random_metrics[p])
                eer_sd = standard_deviation(eer_metrics[p])
                
                frr_metrics_local += frr_metrics[p]
                far_skilled_local += far_skilled_metrics[p]
                far_random_local += far_random_metrics[p]
                eer_local += eer_metrics[p]

                frr_metrics_local_sd += frr_metrics[p]
                far_skilled_local_sd += far_skilled_metrics[p]
                far_random_local_sd += far_random_metrics[p]
                eer_local_sd += eer_metrics[p]

                print("averages " + types[p])
                print(frr_avg)
                print(far_skilled_avg)
                print(far_random_avg)
                print(eer_avg)

                print("standard deviations " + types[p])
                print(frr_sd)
                print(far_skilled_sd)
                print(far_random_sd)
                print(eer_sd)

        
print("Using user Threshold: ")
#Average of all users
print("Average")
print("FRR " + str(average(frr_metrics_local))) 
print("FAR Skilled " + str(average(far_skilled_local)))
print("FAR Random " + str(average(far_random_local)))
print("EER local " + str(average(eer_local)))

#Standard derivation based on user threshold
print("Standard deviation")
print("FRR " + str(standard_deviation(frr_metrics_local_sd))) 
print("FAR Skilled " + str(standard_deviation(far_skilled_local_sd)))
print("FAR Random " + str(standard_deviation(far_random_local_sd)))
print("EER " + str(standard_deviation(eer_local_sd)))

print("Using global Threshold: ")
#Average of all users using global threshold
print("Average")
print("FRR " + str(average(frr_metrics_global)))
print("FAR Skilled " + str(average(far_skilled_metrics_global)))
print("FAR Random " + str(average(far_random_metrics_global)))
print("EER " + str(average(eer_metrics_global)))

#Standard derivation based on global threshold
print("Standard deviation")
print("FRR " + str(standard_deviation(frr_metrics_global)))
print("FAR Skilled " + str(standard_deviation(far_skilled_metrics_global)))
print("FAR Random " + str(standard_deviation(far_random_metrics_global)))
print("EER " + str(standard_deviation(eer_metrics_global)))

print("Area under curve")
print("AUC AVG: " + str(average(auc_metrics)))
print("AUC SD: " + str(standard_deviation(auc_metrics)))
