import random
import os
from process_helper import filter_genuine, filter_forgery, filter_by_text, remove_invalid_files

def split_into_train_test(array, genuine_user, dataset_path, genuine_train_quantity, forgery_train_quantity, random_train_quantity):
    print("user: " + genuine_user)
    signature_images = os.listdir(dataset_path+genuine_user) #Get images from folder
    signature_images = filter(remove_invalid_files, signature_images)
    signature_images = [dataset_path + genuine_user + file for file in signature_images]

    #Split genuine signature for train and test
    genuine_signature_images = get_images_splited(signature_images, genuine_train_quantity, filter_genuine)
    genuine_signature_images_for_train = genuine_signature_images[0]
    genuine_signature_images_for_test = genuine_signature_images[1]
    
    #Split forgery signature for train and test
    forgery_signature_images = get_images_splited(signature_images, forgery_train_quantity, filter_forgery)
    forgery_signature_images_for_train = forgery_signature_images[0]
    forgery_signature_images_for_test = forgery_signature_images[1]

    #Split random signature for train and test
    array.remove(genuine_user) #Removing genuine_user to avoid get a invalid random signature
    random_signature_images = get_random_signatures(array, dataset_path, random_train_quantity)
    random_signature_images_for_train = random_signature_images[0]
    random_signature_images_for_test = random_signature_images[1]

    #Merge lists to create train and test set
    train_set = genuine_signature_images_for_train + forgery_signature_images_for_train + random_signature_images_for_train
    test_set = [genuine_signature_images_for_test, forgery_signature_images_for_test, random_signature_images_for_test]
    
    #Creating classification list
    number_of_genuines_for_train = len(genuine_signature_images_for_train)
    number_of_forgeries_and_randoms_for_train = len(forgery_signature_images_for_train) + len(random_signature_images_for_train)
    train_classification_list = generate_classes_list(number_of_genuines_for_train, number_of_forgeries_and_randoms_for_train)
    array.append(genuine_user) #Removing genuine_user to avoid get a invalid random signature
    return [[train_set, test_set], train_classification_list]

def generate_classes_list(number_of_genuine, number_of_forgery_and_random):
    genuine = list()
    for i in range(0, number_of_genuine):
        genuine.append(1)
    forgery = list()
    for i in range(0, number_of_forgery_and_random):
        forgery.append(0)
    return genuine + forgery

def get_random_signatures(folders, dataset_path, number_for_train):
    random_signatures_for_train = []
    random_signatures_for_test = []
    for folder in folders:
        path = dataset_path + folder
        signature_images = os.listdir(dataset_path + folder)
        signature_images = filter(remove_invalid_files, signature_images)
        signature_images = filter(filter_genuine, signature_images)
        signature_images = [path + file for file in signature_images]
        random.shuffle(signature_images)
        random_signatures_for_train = random_signatures_for_train + signature_images[:number_for_train]
        random_signatures_for_test = random_signatures_for_test + signature_images[number_for_train:]
    return [random_signatures_for_train, random_signatures_for_test]
    


def get_images_splited(signature_images, number_for_train, filter_function):
    signature_images = filter(filter_function, signature_images)
    random.shuffle(signature_images)
    signature_images_for_train = signature_images[:number_for_train]
    signature_images_for_test = signature_images[number_for_train:]
    return [signature_images_for_train, signature_images_for_test]