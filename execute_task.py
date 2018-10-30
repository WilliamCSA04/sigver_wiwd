import sys
import signet
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

test_set = {
    "genuines": [],
    "skilled": [],
    "random": []
}

svm = config["svm_linear"]

model = CNNModel(signet, svm["model_path"])
images_dictionary = {}

train_genuine_users = get_signature_folders(config["dataset_path"])
print("Loading list for genuine users to train")
train_random_users = get_signature_folders(config["dataset_for_random_path"])
print("Loading list for random users to train")

print("Starting preprocess genuine signatures for train")
for user in train_genuine_users:
    path = config["dataset_path"] + user
    train_config = config["train_config"]
    genuine_signatures = get_genuines(path, train_config["genuine"])
    for image in genuine_signatures:
        image_path = path+"/"+image
        add_feature_vector_from_a_image(images_dictionary, image_path, config["max_image_size"], config["canvas"], train_set["genuines"], model)
    print("Train set:")
    print("Train Genuines: " + str(len(train_set["genuines"])))
    print("Train Skilled: " + str(len(train_set["skilled"])))
    print("Train Random: " + str(len(train_set["random"])))