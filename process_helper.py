import os

def filter_forgery(text):
    return "f" in text.lower()

def filter_genuine(text):
    return "f" not in text.lower()

def filter_by_text(text_full, text_part):
    return text_part.lower() in text_full.lower()

def remove_invalid_files(name):
    return ".bmp" in name.lower() or ".jpg" in name.lower()

def validate_train_test(train_set, test_set):
    for image in test_set:
        if(image in train_set):
            print("Image " + image + " found at train set and test set")
            exit(1)