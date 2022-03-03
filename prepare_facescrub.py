import os
from glob import glob
import shutil
import cv2
import math

def transform_file(image_file, target_dir, resize_shape, resize = True, gray = False):
    image = cv2.imread(image_file)
    image_file_name = os.path.basename(image_file)
    
    try:
        if resize:
            image = cv2.resize(image, resize_shape)
        if gray:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        new_file = os.path.join(target_dir, image_file_name)
        cv2.imwrite(new_file, image)
        #shutil.copy(image_file, new_face_dir)
    except:
        print('error resizing image, will not save it'.format())

def copy_and_transform_files(files, training_dir, validate_dir, train_segment_size, resize_shape, resize = True, gray = False):
    files_count = len(files)
    training_count = math.ceil(files_count * train_segment_size)
    training_files = files[:training_count]
    validate_files = files[training_count:]

    for image_file in training_files:
        transform_file(image_file, training_dir, resize_shape, resize = resize, gray = gray)
        
    for image_file in validate_files:
        transform_file(image_file, validate_dir, resize_shape, resize = resize, gray = gray)
    return files_count


if __name__ == '__main__':
    #total number of train file is 39969
    #total number of validation file is 4170
    resize = True
    resize_shape = (32,32)
    # resize_shape = (64,64)
    gray = False
    new_dir = './facescrub-dataset/32x32/'
    # new_dir = './facescrub-dataset/64x64/'
    
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    train_dir = os.path.join(new_dir, 'train')
    validate_dir = os.path.join(new_dir, 'validate')

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    if not os.path.exists(validate_dir):
        os.makedirs(validate_dir)


    train_segment_size = 1.0
    parent_dir = './facescrub-dataset/raw/train'
    # Loop over all the directories of each person
    total_num_files = 0
    for class_dir in glob(os.path.join(parent_dir, "*")):
        face_dir = os.path.join(class_dir, '')
        class_name = os.path.basename(class_dir)

        new_training_face_dir = os.path.join(train_dir, class_name)
        new_validate_face_dir = os.path.join(validate_dir, class_name)

        # make a new training class directory
        if not os.path.exists(new_training_face_dir):
            os.makedirs(new_training_face_dir)
        
        # mae a new validate class directory
        if not os.path.exists(new_validate_face_dir):
            os.makedirs(new_validate_face_dir)
        
        files = glob(os.path.join(face_dir, '*.jpg'))
        
        #print(files, new_training_face_dir, new_validate_face_dir, train_segment_size)
        files_count = copy_and_transform_files(files, new_training_face_dir, new_validate_face_dir, train_segment_size, resize_shape, resize = resize, gray = gray)
        total_num_files += files_count
    print("total number of train file is {}".format(total_num_files))

    train_segment_size = 0.0
    parent_dir = './facescrub-dataset/raw/validate'
    # Loop over all the directories of each person
    total_num_files = 0
    for class_dir in glob(os.path.join(parent_dir, "*")):
        face_dir = os.path.join(class_dir, '')
        class_name = os.path.basename(class_dir)

        new_training_face_dir = os.path.join(train_dir, class_name)
        new_validate_face_dir = os.path.join(validate_dir, class_name)

        # make a new training class directory
        if not os.path.exists(new_training_face_dir):
            os.makedirs(new_training_face_dir)
        
        # mae a new validate class directory
        if not os.path.exists(new_validate_face_dir):
            os.makedirs(new_validate_face_dir)
        
        files = glob(os.path.join(face_dir, '*.jpg'))
        
        #print(files, new_training_face_dir, new_validate_face_dir, train_segment_size)
        files_count = copy_and_transform_files(files, new_training_face_dir, new_validate_face_dir, train_segment_size, resize_shape, resize = resize, gray = gray)
        total_num_files += files_count
    
    print("total number of validation file is {}".format(total_num_files))
        
