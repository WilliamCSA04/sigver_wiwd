import os
from process_helper import filter_genuine, filter_forgery, remove_invalid_files

def get_genuines(path, number_of_genuines):
    signatures = __get_signatures(path)
    filtered_signatures = filter(filter_genuine, signatures)[:number_of_genuines]
    return filtered_signatures

def get_skilled(path, number_of_genuines):
    signatures = __get_signatures(path)
    filtered_signatures = filter(filter_forgery, signatures)[:number_of_genuines]
    return filtered_signatures

def __get_signatures(path):
    signatures = os.listdir(path)
    filtered_signatures = filter(remove_invalid_files, signatures)
    return filtered_signatures
    