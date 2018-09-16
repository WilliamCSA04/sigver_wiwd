import os

def filter_forgery(text):
    return "f" in text.lower()

def filter_genuine(text):
    return "f" not in text.lower()

def filter_by_text(text_full, text_part):
    return text_part.lower() in text_full.lower()

def remove_invalid_files(name):
    return ".bmp" in name.lower() or ".jpg" in name.lower()