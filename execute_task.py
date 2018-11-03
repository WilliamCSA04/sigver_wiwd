import sys
import signet
import random
import classifier
from cnn_model import CNNModel
from gpds_signatures import *
from signature_image_features import add_feature_vector_from_a_image
from metrics import *
from process_helper import validate_train_test

dataset = sys.argv[1]

config = {}
if dataset.lower() == "gpds160":
    from gpds_config import gpds160_config
    config = gpds160_config()
    print("Get GPDS-160 configurations")
elif dataset.lower() == "gpds300":
    from gpds_config import gpds300_config
    config = gpds300_config()
    print("Get GPDS-300 configurations")
else:
    print("Error: No valid dataset selected")
    exit()

train_set = {
    "genuines": [],
    "skilled": [],
    "random": []
}

results = [[], [], [], [], [], [], [], [], []]

svm = config["svm_linear"]
print(svm)
model = CNNModel(signet, svm["model_path"])
images_dictionary = {}

list_of_signatures_use_on_train = []
list_of_signatures_use_on_test = []

random_users = get_signature_folders(config["dataset_for_random_path"])
print("Loading list for random users to train")

train_config = config["train_config"]

print("Starting preprocess random signatures for train")
random_users_size = len(random_users)
for count, user in enumerate(random_users):
    
    print("Processing Random Signatures " + str(count) + "/" +str(random_users_size))
    path = config["dataset_for_random_path"] + user
    random_signatures = get_genuines(path, train_config["random"])
    for image in random_signatures:
        image_path = path+"/"+image
        list_of_signatures_use_on_train.append(image_path)
        add_feature_vector_from_a_image(images_dictionary, image_path, config["max_image_size"], config["canvas"], train_set["random"], model)

train_genuine_users = get_signature_folders(config["dataset_path"])
print("Loading list for genuine users to train")

for user in train_genuine_users:
    print("Starting preprocess genuine signatures for train " + user)
    train_set["genuines"] = []
    path = config["dataset_path"] + user
    genuine_for_train = train_config["genuine"]
    genuine_signatures = get_genuines(path, genuine_for_train)
    for image in genuine_signatures:
        image_path = path+"/"+image
        list_of_signatures_use_on_train.append(image_path)
        add_feature_vector_from_a_image(images_dictionary, image_path, config["max_image_size"], config["canvas"], train_set["genuines"], model)

    print("Train set:")
    print("Train Genuines: " + str(len(train_set["genuines"])))
    print("Train Skilled: " + str(len(train_set["skilled"])))
    print("Train Random: " + str(len(train_set["random"])))
    max_signature_numbers = config["signature_numbers_by_user"]
    data_train = train_set["genuines"] + train_set["random"]
    train_classes = []
    for i in train_set["genuines"]:
        train_classes.append(1)
    for i in train_set["random"]:
        train_classes.append(0)
    c_plus = len(train_set["random"])/len(train_set["genuines"])
    weights = {0: svm["c-minus"], 1: c_plus}
    clf = classifier.svm(data_train, train_classes, gamma = svm["gamma"], weights = weights)
    test_sets = []
    for time in range(0, config["number_of_tests_by_user"]):
        print("starting test" + str(time) + "/" + str(config["number_of_tests_by_user"]))
        test_set = {
            "genuines": [],
            "skilled": [],
            "random": []
        }
        test_config = config["test_config"]
        print("Loading genuine signatures to test")
        genuine_signatures = get_genuines(path, max_signature_numbers["genuine"])[genuine_for_train:]
        print("Starting preprocess genuine signatures to test")
        for image in genuine_signatures:
            image_path = path+"/"+image
            list_of_signatures_use_on_test.append(image_path)
            add_feature_vector_from_a_image(images_dictionary, image_path, config["max_image_size"], config["canvas"], test_set["genuines"], model)
        
        print("Loading skilled signatures to test")
        skilled_signatures = get_skilled(path, test_config['skilled'])
        print("Starting preprocess skilled signatures to test")
        for image in genuine_signatures:
            image_path = path+"/"+image
            list_of_signatures_use_on_test.append(image_path)
            add_feature_vector_from_a_image(images_dictionary, image_path, config["max_image_size"], config["canvas"], test_set["skilled"], model)
        
        print("Loading random signatures to test")
        
        temp_random_images = []
        for user in random_users:
            path = config["dataset_for_random_path"] + user
            random_signatures = get_genuines(path, max_signature_numbers["genuine"])[genuine_for_train:]
            for image in random_signatures:
                image_path = path+"/"+image
                list_of_signatures_use_on_test.append(image_path)
                temp_random_images.append(image_path)

        random.shuffle(temp_random_images)
        for image_path in temp_random_images[:test_config["random"]]:
            add_feature_vector_from_a_image(images_dictionary, image_path, config["max_image_size"], config["canvas"], test_set["random"], model)

        print("Test set:")
        print("Test Genuines: " + str(len(test_set["genuines"])))
        print("Test Skilled: " + str(len(test_set["skilled"])))
        print("Test Random: " + str(len(test_set["random"])))

        validate_train_test(list_of_signatures_use_on_train, list_of_signatures_use_on_test)
        data_test = test_set["genuines"] + test_set["skilled"] + test_set["random"]
        test_classes = []
        for i in test_set["genuines"]:
            test_classes.append(1)
        for i in test_set["skilled"]:
            test_classes.append(0)
        for i in test_set["random"]:
            test_classes.append(0)
        test_sets.append(data_test)
    partial_results = classifier.test(clf, test_sets, test_classes, test_config["genuine"], test_config["skilled"], test_config["random"], svm["global_threshhold"])
    results[0] += (partial_results[0])
    results[1] += (partial_results[1])
    results[2] += (partial_results[2])
    results[3] += (partial_results[3])
    results[4] += (partial_results[4])
    results[5] += (partial_results[5])
    results[6] += (partial_results[6])
    results[7] += (partial_results[7])
    results[8] += (partial_results[8])        

print(results)

print("Results: ")
print("===USER AVG===: ")
print("FRR: " + str(average(results[0])))
print("FAR_SKILLED: " + str(average(results[1])))
print("FAR_RANDOM: " + str(average(results[2])))
print("EER: " + str(average(results[3])))
print("===USER SD===: ")
print("FRR: " + str(standard_deviation(results[0])))
print("FAR_SKILLED: " + str(standard_deviation(results[1])))
print("FAR_RANDOM: " + str(standard_deviation(results[2])))
print("EER: " + str(standard_deviation(results[3])))

print("===GLOBAL AVG===: ")
print("FRR: " + str(average(results[4])))
print("FAR_SKILLED: " + str(average(results[5])))
print("FAR_RANDOM: " + str(average(results[6])))
print("EER: " + str(average(results[7])))
print("===GLOBAL SD===: ")
print("FRR: " + str(standard_deviation(results[4])))
print("FAR_SKILLED: " + str(standard_deviation(results[5])))
print("FAR_RANDOM: " + str(standard_deviation(results[6])))
print("EER: " + str(standard_deviation(results[7])))
print("===AUC===")
print("AVG: " + str(average(results[8])))
print("SD: " + str(standard_deviation(results[8])))
