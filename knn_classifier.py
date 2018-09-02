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


if len(sys.argv) not in [4,6]:
    print('Usage: python process_folder.py <signatures_path> <save_path> '
          '<model_path> [canvas_size]')
    exit(1)

signatures_path = sys.argv[1]
save_path = sys.argv[2]
model_path = sys.argv[3]


if len(sys.argv) == 4:
    canvas_size = (952, 1360)  # Maximum signature size
else:
    canvas_size = (int(sys.argv[4]), int(sys.argv[5]))

print('Processing images from folder "%s" and saving to folder "%s"' % (signatures_path, save_path))
print('Using model %s' % model_path)
print('Using canvas size: %s' % (canvas_size,))

# Load the model
model_weight_path = 'models/signet.pkl'
model = CNNModel(signet, model_weight_path)

files = os.listdir(signatures_path)

# Note: it there is a large number of signatures to process, it is faster to
# process them in batches (i.e. use "get_feature_vector_multiple")
k = 1
data = list()
for f in files:
    # Load and pre-process the signature
    filename = os.path.join(signatures_path, f)
    if(f == "Thumbs.db"):
        print("Continue")
        continue
    print(f)
    original = imread(filename, flatten=1)
    processed = preprocess_signature(original, canvas_size)

    # Use the CNN to extract features
    feature_vector = model.get_feature_vector(processed)
    data.append(feature_vector[0])
    # Save in the matlab format
    save_filename = os.path.join(save_path, os.path.splitext(f)[0] + '.mat')
    scipy.io.savemat(save_filename, {'feature_vector':feature_vector})

data_train = np.array([data[0], data[1], data[2], data[3], data[4], data[15], data[16], data[17], data[18], data[19], data[20], data[21], data[22], data[23], data[24]])
data_test = np.array([data[5], data[6], data[7], data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[25], data[26], data[27], data[28], data[29]])
expected = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
for weights in ['uniform', 'distance']:
    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(k, weights=weights)
    clf.fit(data_train, expected)
    prediction = clf.predict(data_test)
    print(prediction) # shouldn't error anymore

