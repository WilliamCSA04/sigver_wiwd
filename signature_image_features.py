from scipy.misc import imread
from scipy.misc import imresize
from preprocess.normalize import preprocess_signature

def add_feature_vector_from_a_image(images_dictionary, image_path, img_max_size, canvas, sets_processed, model):
    if image_path in images_dictionary.keys():
        sets_processed.append(images_dictionary[image_path])
    else:
        original = imread(image_path, flatten=1)
        height, width = original.shape
        if height > img_max_size[0]:
            diff = height - img_max_size[0]
            percentage = (100*diff)/height
            original = imresize(original, 100-percentage)
            height, width = original.shape
        if width > img_max_size[1]:
            diff = width - img_max_size[1]
            percentage = (100*diff)/width
            original = imresize(original, 100-percentage)
            height, width = original.shape

        
        processed = preprocess_signature(original, canvas)
        images_dictionary[image_path] = model.get_feature_vector(processed)[0]
        sets_processed.append(images_dictionary[image_path])