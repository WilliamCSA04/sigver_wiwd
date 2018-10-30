import os
from process_helper import filter_genuine, filter_forgery, remove_invalid_files

def get_genuines(path, number_of_signatures):
    signatures = __get_signatures(path)
    __validate_number_of_signatures(len(signatures), number_of_signatures)
    filtered_signatures = filter(filter_genuine, signatures)[:number_of_signatures]
    return filtered_signatures

def get_skilled(path, number_of_signatures):
    signatures = __get_signatures(path)
    __validate_number_of_signatures(len(signatures), number_of_signatures)
    filtered_signatures = filter(filter_forgery, signatures)[:number_of_signatures]
    return filtered_signatures

def __get_signatures(path):
    signatures = os.listdir(path)
    filtered_signatures = filter(remove_invalid_files, signatures)
    return filtered_signatures

def __validate_number_of_signatures(max_limit, number_of_signatures):
    min_limit = 0
    if(number_of_signatures < 0):
        print("error: number of signature is higher than 0")
        exit()
    elif(number_of_signatures > max_limit):
        print("error: number of signature is lower than " + str(max_limit))
        exit()
    