import os
from glob import glob
import shutil
import cv2
import math

if __name__ == '__main__':
    os.system("wget http://cs231n.stanford.edu/tiny-imagenet-200.zip")
    
    # # Unzip raw zip file
    os.system("unzip -qq 'tiny-imagenet-200.zip'")

    # Define main data directory
    DATA_DIR = 'tiny-imagenet-200' # Original images come in shapes of [3,64,64]

    # Define training and validation data paths
    TRAIN_DIR = os.path.join(DATA_DIR, 'train') 
    VALID_DIR = os.path.join(DATA_DIR, 'val')

    # Create separate validation subfolders for the validation images based on
    # their labels indicated in the val_annotations txt file
    val_img_dir = os.path.join(VALID_DIR, 'images')

    # Open and read val annotations text file
    fp = open(os.path.join(VALID_DIR, 'val_annotations.txt'), 'r')
    data = fp.readlines()

    # Create dictionary to store img filename (word 0) and corresponding
    # label (word 1) for every line in the txt file (as key value pair)
    val_img_dict = {}
    for line in data:
        words = line.split('\t')
        val_img_dict[words[0]] = words[1]
    fp.close()

    # Display first 10 entries of resulting val_img_dict dictionary
    {k: val_img_dict[k] for k in list(val_img_dict)[:10]}

    # Create subfolders (if not present) for validation images based on label,
    # and move images into the respective folders
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(val_img_dir, folder))
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        if os.path.exists(os.path.join(val_img_dir, img)):
            os.rename(os.path.join(val_img_dir, img), os.path.join(newpath, img))