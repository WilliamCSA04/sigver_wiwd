""" This example extract features for all signatures in a folder,
    using the CNN trained on the GPDS dataset. Results are saved in a matlab
    format.

    Usage: python process_folder.py <signatures_path> <save_path>
                                    <model_path> [canvas_size]

    Example:
    python process_folder.py signatures/ features/ models/signet.pkl

    This example will process all signatures in the "signatures" folder, using
    the SigNet model, and saving the resutls to the features folder

"""
from scipy.misc import imread
from preprocess.normalize import preprocess_signature
import signet
from cnn_model import CNNModel
import numpy as np
import sys
import os
import scipy.io

from sklearn import neighbors
from sklearn.metrics import confusion_matrix


if len(sys.argv) not in [5,7]:
    print('Usage: python process_folder.py <signatures_path> <save_path> <signature_user_path> '
          '<model_path> [canvas_size]')
    exit(1)


dataset_path = sys.argv[1]
signatures_folder = sys.argv[4]
signatures_path = dataset_path + signatures_folder
save_path = sys.argv[2]
model_path = sys.argv[3]


if len(sys.argv) == 5:
    canvas_size = (952, 1360)  # Maximum signature size
else:
    canvas_size = (int(sys.argv[5]), int(sys.argv[6]))

print('Processing images from folder "%s" and saving to folder "%s"' % (signatures_path, save_path))
print('Using model %s' % model_path)
print('Using canvas size: %s' % (canvas_size,))

# Load the model
model_weight_path = 'models/signet.pkl'
model = CNNModel(signet, model_weight_path)

paths = os.listdir(signatures_path)
files_signature = paths[:15]
files_skilled = paths[15:]
files_random = os.listdir(dataset_path)

# Note: it there is a large number of signatures to process, it is faster to
# process them in batches (i.e. use "get_feature_vector_multiple")
k = 1
data = list()
expected = list()

print("Generate Train Set")
print("Adding Genuine")
count = 0
for f in files_signature:
    # Load and pre-process the signature
    filename = os.path.join(signatures_path, f)
    if(count == 10):
        break
    if("f" in f):
        continue
    original = imread(filename, flatten=1)
    processed = preprocess_signature(original, canvas_size)

    # Use the CNN to extract features
    feature_vector = model.get_feature_vector(processed)
    data.append(feature_vector[0])
    count += 1
    expected.append(0)

count = 0
print("Adding Skilled")

for index, f in enumerate(files_skilled):
    # Load and pre-process the signature
    filename = os.path.join(signatures_path, f)
    if(count == 10):
        break
    if("f" not in f):
        continue
    original = imread(filename, flatten=1)
    processed = preprocess_signature(original, canvas_size)
    # Use the CNN to extract features
    feature_vector = model.get_feature_vector(processed)
    data.append(feature_vector[0])
    count += 1
    expected.append(1)

print("Adding Random")

for p in files_random:
    validate_p = p + "/"
    if(validate_p == signatures_folder):
        continue
    folder_path = dataset_path + p
    folder = os.listdir(folder_path)
    count = 0 
    for f in folder:
        # Load and pre-process the signature
        if(count == 10):
            break
        if("f" in f):
            continue
        filename = os.path.join(folder_path, f)
        original = imread(filename, flatten=1)
        processed = preprocess_signature(original, canvas_size)

        # Use the CNN to extract features
        feature_vector = model.get_feature_vector(processed)
        data.append(feature_vector[0])
        count += 1
        expected.append(1)



data_train = np.array(data)


data = list()

count_g = 0
count_s = 0
print("Generate Test Set")
correct_class = list()
for f in paths:
    # Load and pre-process the signature
    filename = os.path.join(signatures_path, f)
    if(count_g == 5 and count_s == 15):
        break
    elif("f" not in f and count_g == 5):
        continue
    elif("f" in f and count_s == 15):
        continue   
    original = imread(filename, flatten=1)
    processed = preprocess_signature(original, canvas_size)

    # Use the CNN to extract features
    feature_vector = model.get_feature_vector(processed)
    data.append(feature_vector[0])
    if("f" not in f):
        count_g += 1
        correct_class.append(0)
    else:
        count_s += 1
        correct_class.append(1)

    

data_test = np.array(data)

print("Size of train set: " + str(len(data_train)))
print("Size of test set: " + str(len(data_test)))

print("KNN Classifier") 
for weights in ['uniform', 'distance']:
    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(k, weights=weights)
    clf.fit(data_train, expected)
    prediction = clf.predict(data_test)
    print("Prediction: " + weights)   
    print(confusion_matrix(correct_class, prediction))
    tn, fp, fn, tp = confusion_matrix(correct_class, prediction).ravel()
    print("true positive: " + str(tp))
    print("true negative: " + str(tn))
    print("false positive: " + str(fp))
    print("false negative: " + str(fn))

