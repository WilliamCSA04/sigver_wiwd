import sys
from gpds_signatures import *
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

train_genuine_users = get_signature_folders(config["dataset_path"])
train_random_users = get_signature_folders(config["dataset_for_random_path"])