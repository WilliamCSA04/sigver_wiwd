import sys
import signet
import random
from cnn_model import CNNModel
from gpds_signatures import *
from signature_image_features import add_feature_vector_from_a_image

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



svm = config["svm_linear"]

model = CNNModel(signet, svm["model_path"])
images_dictionary = {}

random_users = get_signature_folders(config["dataset_for_random_path"])
print("Loading list for random users to train")

train_config = config["train_config"]

print("Starting preprocess random signatures for train")
for user in random_users:
    path = config["dataset_for_random_path"] + user
    random_signatures = get_genuines(path, train_config["random"])
    for image in random_signatures:
        image_path = path+"/"+image
        add_feature_vector_from_a_image(images_dictionary, image_path, config["max_image_size"], config["canvas"], train_set["random"], model)


train_genuine_users = get_signature_folders(config["dataset_path"])
print("Loading list for genuine users to train")

print("Starting preprocess genuine signatures for train")
for user in train_genuine_users:
    train_set["genuines"] = []
    path = config["dataset_path"] + user
    genuine_for_train = train_config["genuine"]
    genuine_signatures = get_genuines(path, genuine_for_train)
    for image in genuine_signatures:
        image_path = path+"/"+image
        add_feature_vector_from_a_image(images_dictionary, image_path, config["max_image_size"], config["canvas"], train_set["genuines"], model)

    print("Train set:")
    print("Train Genuines: " + str(len(train_set["genuines"])))
    print("Train Skilled: " + str(len(train_set["skilled"])))
    print("Train Random: " + str(len(train_set["random"])))
    test_set = {
        "genuines": [],
        "skilled": [],
        "random": []
    }
    max_signature_numbers = config["signature_numbers_by_user"]
    test_config = config["test_config"]
    print("Loading genuine signatures to test")
    genuine_signatures = get_genuines(path, max_signature_numbers["genuine"])[genuine_for_train:]
    print("Starting preprocess genuine signatures to test")
    for image in genuine_signatures:
        image_path = path+"/"+image
        add_feature_vector_from_a_image(images_dictionary, image_path, config["max_image_size"], config["canvas"], test_set["genuines"], model)
    
    print("Loading skilled signatures to test")
    skilled_signatures = get_skilled(path, test_config['skilled'])
    print("Starting preprocess skilled signatures to test")
    for image in genuine_signatures:
        image_path = path+"/"+image
        add_feature_vector_from_a_image(images_dictionary, image_path, config["max_image_size"], config["canvas"], test_set["skilled"], model)
    
    print("Loading random signatures to test")
    
    temp_random_images = []
    for user in random_users:
        path = config["dataset_for_random_path"] + user
        random_signatures = get_genuines(path, max_signature_numbers["genuine"])[genuine_for_train:]
        for image in random_signatures:
            image_path = path+"/"+image
            temp_random_images.append(image_path)

    random.shuffle(temp_random_images)
    for image_path in temp_random_images[:test_config["random"]]:
        add_feature_vector_from_a_image(images_dictionary, image_path, config["max_image_size"], config["canvas"], test_set["random"], model)

    print("Test set:")
    print("Test Genuines: " + str(len(test_set["genuines"])))
    print("Test Skilled: " + str(len(test_set["skilled"])))
    print("Test Random: " + str(len(test_set["random"])))