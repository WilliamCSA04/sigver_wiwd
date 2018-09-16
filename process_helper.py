import os

def filter_forgery(text):
    return "f" in text

def filter_genuine(text):
    return "f" not in text

def filter_by_text(text_full, text_part):
    return text_part in text_full

def remove_invalid_files(name):
    return ".bmp" in name or ".jpg" in name