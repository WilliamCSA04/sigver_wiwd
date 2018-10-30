def gpds160_config():
    config = {
        "dataset_genuine_path": "/datasets/gpds160/",
        "dataset_random_path": "/datasets/gpds160-RANDOM/",
        "train_config": {
            "genuine": 14,
            "skilled": 0,
            "random": 14
        },
        "test_config": {
            "genuine": 10,
            "skilled": 10,
            "random": 10
        },
        "number_of_interations": 160,
        "max_image_size": (819, 1137),
        "canvas": (952, 1360)
    }
    return config

def gpds300_config():
    config = {
        "dataset_genuine_path": "/datasets/gpds300/",
        "dataset_random_path": "/datasets/gpds300-RANDOM/",
        "train_config": {
            "genuine": 14,
            "skilled": 0,
            "random": 14
        },
        "test_config": {
            "genuine": 10,
            "skilled": 10,
            "random": 10
        },
        "number_of_interations": 300,
        "max_image_size": (819, 1137),
        "canvas": (952, 1360)
    }
    return config