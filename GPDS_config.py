svm_linear = {
            "model_path": "models/signetf_lambda0.95.pkl",
            "c-minus": 1,
            "gamma": "auto"
        }
svm_rbf = {
            "model_path": "models/signetf_lambda0.999.pkl",
            "c-minus": 1,
            "gamma": 2**(-11)
        }
signature_numbers_by_user = {
            "genuine": 24,
            "skilled": 30,
        }
def gpds160_config():
    config = {
        "dataset_path": "datasets/gpds160/",
        "dataset_for_random_path": "datasets/gpds160-RANDOM/",
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
        "signature_numbers_by_user": signature_numbers_by_user,
        "number_of_trains": 160,
        "number_of_tests_by_user": 100,
        "max_image_size": (819, 1137),
        "canvas": (952, 1360),
        "svm_linear": svm_linear,
        "svm_rbf": svm_rbf
    }
    return config

def gpds300_config():
    config = {
        "dataset_path": "datasets/gpds300/",
        "dataset_for_random_path": "datasets/gpds300-RANDOM/",
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
        "signature_numbers_by_user": signature_numbers_by_user,
        "number_of_trains": 300,
        "number_of_tests_by_user": 100,
        "max_image_size": (819, 1137),
        "canvas": (952, 1360),
        "svm_liner": svm_linear,
        "svm_rbf": svm_rbf
    }
    return config