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
svm_kernel = sys.argv[2]

config = {}
if dataset.lower() == "gpds160":
    from gpds_dataset_config import gpds160_config
    config = gpds160_config()
    print("Get GPDS-160 configurations")
elif dataset.lower() == "gpds300":
    from gpds_dataset_config import gpds300_config
    config = gpds300_config()
    print("Get GPDS-300 configurations")
elif dataset.lower() == "gpds50":
    from gpds_dataset_config import gpds50_config
    config = gpds50_config()
    print("Get GPDS-50 configurations")
else:
    print("Error: No valid dataset selected")
    exit()

train_set = {
    "genuines": [],
    "skilled": [],
    "random": []
}

results = {
    "svm_linear": [[], [], [], [], [], [], []],
    "svm_rbf": [[], [], [], [], [], [], []],
    "mlp": [[], [], [], [], [], [], []],
    "knn": [[], [], [], [], [], [], []],
    "tree": [[], [], [], [], [], [], []],
}

svm = None
if svm_kernel == "linear":
    svm = config["svm_linear"]
elif svm_kernel == "rbf":
    svm = config["svm_rbf"]
else:
    print("Invalid kernel for svm")
    exit()


print(svm)
model = CNNModel(signet, svm["model_path"])
images_dictionary = {}

list_of_signatures_use_on_train = []
list_of_signatures_use_on_test = []

weights = {1: config["c-plus"], 0: svm["c-minus"]}
svc_linear = classifier.svm(gamma = 'auto', weights = weights, kernel="linear")
print(svc_linear)
svc_rbf = classifier.svm(gamma = 2**(-11), weights = weights, kernel="rbf")
print(svc_rbf)
mlp = classifier.mlp()
print(mlp)
knn = classifier.knn()
print(knn)
tree = classifier.tree(weights = weights)
print(tree)

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
    clf_svm_linear = svc_linear.fit(data_train, train_classes)
    clf_svm_rbf = svc_rbf.fit(data_train, train_classes)
    clf_mlp = mlp.fit(data_train, train_classes)
    clf_knn = knn.fit(data_train, train_classes)
    clf_tree = tree.fit(data_train, train_classes)
    test_sets = []
    print(c_plus)
    for time in range(0, config["number_of_tests_by_user"]):
        print("starting test" + str(time) + "/" + str(config["number_of_tests_by_user"]))
        test_set = {
            "genuines": [],
            "skilled": [],
            "random": []
        }
        test_config = config["test_config"]
        genuine_signatures = get_genuines(path, max_signature_numbers["genuine"])[genuine_for_train:]
        for image in genuine_signatures:
            image_path = path+"/"+image
            list_of_signatures_use_on_test.append(image_path)
            add_feature_vector_from_a_image(images_dictionary, image_path, config["max_image_size"], config["canvas"], test_set["genuines"], model)
        
        skilled_signatures = get_skilled(path, test_config['skilled'])
        for image in skilled_signatures:
            image_path = path+"/"+image
            list_of_signatures_use_on_test.append(image_path)
            add_feature_vector_from_a_image(images_dictionary, image_path, config["max_image_size"], config["canvas"], test_set["skilled"], model)
        
        
        temp_random_images = []
        for user_test in random_users:
            path_random = config["dataset_for_random_path"] + user_test
            random_signatures = get_genuines(path_random, max_signature_numbers["genuine"])[genuine_for_train:]
            for image in random_signatures:
                image_path = path_random+"/"+image
                temp_random_images.append(image_path)

        random.shuffle(temp_random_images)
        for image_path in temp_random_images[:test_config["random"]]:
            list_of_signatures_use_on_test.append(image_path)
            add_feature_vector_from_a_image(images_dictionary, image_path, config["max_image_size"], config["canvas"], test_set["random"], model)

        data_test = test_set["genuines"] + test_set["skilled"] + test_set["random"]
        test_classes = []
        for i in test_set["genuines"]:
            test_classes.append(1)
        for i in test_set["skilled"]:
            test_classes.append(0)
        for i in test_set["random"]:
            test_classes.append(0)
        test_sets.append(data_test)
    partial_results = classifier.test(clf_mlp, test_sets, test_classes, test_config["genuine"], test_config["skilled"], test_config["random"])
    results["mlp"][0] += (partial_results[0])
    results["mlp"][1] += (partial_results[1])
    results["mlp"][2] += (partial_results[2])
    results["mlp"][3] += (partial_results[3])
    results["mlp"][4] += (partial_results[4])
    results["mlp"][5] += (partial_results[5])       
    results["mlp"][6] += (partial_results[6])       
    partial_results = classifier.test(clf_svm_linear, test_sets, test_classes, test_config["genuine"], test_config["skilled"], test_config["random"])
    results["svm_linear"][0] += (partial_results[0])
    results["svm_linear"][1] += (partial_results[1])
    results["svm_linear"][2] += (partial_results[2])
    results["svm_linear"][3] += (partial_results[3])
    results["svm_linear"][4] += (partial_results[4])
    results["svm_linear"][5] += (partial_results[5])       
    results["svm_linear"][6] += (partial_results[6])
    partial_results = classifier.test(clf_svm_rbf, test_sets, test_classes, test_config["genuine"], test_config["skilled"], test_config["random"])
    results["svm_rbf"][0] += (partial_results[0])
    results["svm_rbf"][1] += (partial_results[1])
    results["svm_rbf"][2] += (partial_results[2])
    results["svm_rbf"][3] += (partial_results[3])
    results["svm_rbf"][4] += (partial_results[4])
    results["svm_rbf"][5] += (partial_results[5])       
    results["svm_rbf"][6] += (partial_results[6])   
    partial_results = classifier.test(clf_knn, test_sets, test_classes, test_config["genuine"], test_config["skilled"], test_config["random"])
    results["knn"][0] += (partial_results[0])
    results["knn"][1] += (partial_results[1])
    results["knn"][2] += (partial_results[2])
    results["knn"][3] += (partial_results[3])
    results["knn"][4] += (partial_results[4])
    results["knn"][5] += (partial_results[5])       
    results["knn"][6] += (partial_results[6])  
    partial_results = classifier.test(clf_tree, test_sets, test_classes, test_config["genuine"], test_config["skilled"], test_config["random"])
    results["tree"][0] += (partial_results[0])
    results["tree"][1] += (partial_results[1])
    results["tree"][2] += (partial_results[2])
    results["tree"][3] += (partial_results[3])
    results["tree"][4] += (partial_results[4])
    results["tree"][5] += (partial_results[5])       
    results["tree"][6] += (partial_results[6])  

print("results")
print(results)
print("Results MLP: ")
print("===AVG===: ")
print("FRR: " + str(average(results["mlp"][0])))
print("FAR_SKILLED: " + str(average(results["mlp"][1])))
print("FAR_RANDOM: " + str(average(results["mlp"][2])))
print("EER: " + str(average(results["mlp"][3])))
print("EER_userthresholds: " + str(average(results["mlp"][4])))
print("AUC: " + str(average(results["mlp"][5])))
print("Threshold: " + str(average(results["mlp"][6])))
print("===SD===: ")
print("FRR: " + str(standard_deviation(results["mlp"][0])))
print("FAR_SKILLED: " + str(standard_deviation(results["mlp"][1])))
print("FAR_RANDOM: " + str(standard_deviation(results["mlp"][2])))
print("EER: " + str(standard_deviation(results["mlp"][3])))
print("EER_userthresholds: " + str(standard_deviation(results["mlp"][4])))
print("AUC: " + str(standard_deviation(results["mlp"][5])))
print("Threshold: " + str(standard_deviation(results["mlp"][6])))

print("Results SVM LINEAR: ")
print("===AVG===: ")
print("FRR: " + str(average(results["svm_linear"][0])))
print("FAR_SKILLED: " + str(average(results["svm_linear"][1])))
print("FAR_RANDOM: " + str(average(results["svm_linear"][2])))
print("EER: " + str(average(results["svm_linear"][3])))
print("EER_userthresholds: " + str(average(results["svm_linear"][4])))
print("AUC: " + str(average(results["svm_linear"][5])))
print("Threshold: " + str(average(results["svm_linear"][6])))
print("===SD===: ")
print("FRR: " + str(standard_deviation(results["svm_linear"][0])))
print("FAR_SKILLED: " + str(standard_deviation(results["svm_linear"][1])))
print("FAR_RANDOM: " + str(standard_deviation(results["svm_linear"][2])))
print("EER: " + str(standard_deviation(results["svm_linear"][3])))
print("EER_userthresholds: " + str(standard_deviation(results["svm_linear"][4])))
print("AUC: " + str(standard_deviation(results["svm_linear"][5])))
print("Threshold: " + str(standard_deviation(results["svm_linear"][6])))

print("Results SVM RBF: ")
print("===AVG===: ")
print("FRR: " + str(average(results["svm_rbf"][0])))
print("FAR_SKILLED: " + str(average(results["svm_rbf"][1])))
print("FAR_RANDOM: " + str(average(results["svm_rbf"][2])))
print("EER: " + str(average(results["svm_rbf"][3])))
print("EER_userthresholds: " + str(average(results["svm_rbf"][4])))
print("AUC: " + str(average(results["svm_rbf"][5])))
print("Threshold: " + str(average(results["svm_rbf"][6])))
print("===SD===: ")
print("FRR: " + str(standard_deviation(results["svm_rbf"][0])))
print("FAR_SKILLED: " + str(standard_deviation(results["svm_rbf"][1])))
print("FAR_RANDOM: " + str(standard_deviation(results["svm_rbf"][2])))
print("EER: " + str(standard_deviation(results["svm_rbf"][3])))
print("EER_userthresholds: " + str(standard_deviation(results["svm_rbf"][4])))
print("AUC: " + str(standard_deviation(results["svm_rbf"][5])))
print("Threshold: " + str(standard_deviation(results["svm_rbf"][6])))

print("Results KNN: ")
print("===AVG===: ")
print("FRR: " + str(average(results["knn"][0])))
print("FAR_SKILLED: " + str(average(results["knn"][1])))
print("FAR_RANDOM: " + str(average(results["knn"][2])))
print("EER: " + str(average(results["knn"][3])))
print("EER_userthresholds: " + str(average(results["knn"][4])))
print("AUC: " + str(average(results["knn"][5])))
print("Threshold: " + str(average(results["knn"][6])))
print("===SD===: ")
print("FRR: " + str(standard_deviation(results["knn"][0])))
print("FAR_SKILLED: " + str(standard_deviation(results["knn"][1])))
print("FAR_RANDOM: " + str(standard_deviation(results["knn"][2])))
print("EER: " + str(standard_deviation(results["knn"][3])))
print("EER_userthresholds: " + str(standard_deviation(results["knn"][4])))
print("AUC: " + str(standard_deviation(results["knn"][5])))
print("Threshold: " + str(standard_deviation(results["knn"][6])))

print("Results TREE: ")
print("===AVG===: ")
print("FRR: " + str(average(results["tree"][0])))
print("FAR_SKILLED: " + str(average(results["tree"][1])))
print("FAR_RANDOM: " + str(average(results["tree"][2])))
print("EER: " + str(average(results["tree"][3])))
print("EER_userthresholds: " + str(average(results["tree"][4])))
print("AUC: " + str(average(results["tree"][5])))
print("Threshold: " + str(average(results["tree"][6])))
print("===SD===: ")
print("FRR: " + str(standard_deviation(results["tree"][0])))
print("FAR_SKILLED: " + str(standard_deviation(results["tree"][1])))
print("FAR_RANDOM: " + str(standard_deviation(results["tree"][2])))
print("EER: " + str(standard_deviation(results["tree"][3])))
print("EER_userthresholds: " + str(standard_deviation(results["tree"][4])))
print("AUC: " + str(standard_deviation(results["tree"][5])))
print("Threshold: " + str(standard_deviation(results["tree"][6])))
