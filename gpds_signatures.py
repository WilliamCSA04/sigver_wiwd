import os
from process_helper import filter_genuine, filter_forgery, remove_invalid_files
import random

def get_signature_folders(path):
    return os.listdir(path)

def get_genuines(path, number_of_signatures):
    signatures = __get_signatures(path)
    filtered_signatures = __filter_signatures(filter_genuine, signatures, number_of_signatures)
    return filtered_signatures

def get_skilled(path, number_of_signatures):
    signatures = __get_signatures(path)
    filtered_signatures = __filter_signatures(filter_forgery, signatures, number_of_signatures, should_shuffle = True)
    return filtered_signatures

def __get_signatures(path):
    signatures = os.listdir(path)
    filtered_signatures = filter(remove_invalid_files, signatures)
    return filtered_signatures

def __filter_signatures(filter_method, signatures, number_of_signatures, should_shuffle = False):
    __validate_number_of_signatures(len(signatures), number_of_signatures)
    if should_shuffle:
        filtered_signatures = filter(filter_method, signatures)
        random.shuffle(filtered_signatures)
        return filtered_signatures[:number_of_signatures]
    
    return filter(filter_method, signatures)[:number_of_signatures]

def __validate_number_of_signatures(max_limit, number_of_signatures):
    min_limit = 0
    if(number_of_signatures < min_limit):
        print("error: number of signature is lower than " + str(min_limit))
        exit()
    elif(number_of_signatures > max_limit):
        print("error: number of signature is higher than " + str(max_limit))
        exit()
    