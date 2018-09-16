import os


def filter_array_of_folders(folders, path):
    for folder in folders:
        files = os.listdir(path + folder)
        files = filter(remove_invalid_files, files)
    return folders



def remove_invalid_files(name):
    return ".bmp" in name or ".jpg" in name