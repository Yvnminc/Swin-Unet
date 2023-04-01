'''
File name: img_to_npz.py
Authors: Yanming Guo
Description: Compress the dataset into npz format.
'''
import os
import numpy as np
from PIL import Image

def load_images_from_folder(path):
    '''
    Load the images from folder to NumPy array.
    '''
    images = []
    for filename in os.listdir(path):
        if filename.lower().endswith(('.tiff', '.tif', '.png', '.jpg', '.jpeg')):
            img = Image.open(os.path.join(path, filename))
            img_array = np.array(img)
            img_shape = img_array.shape
            images.append(img_array)

    images = np.stack(images, axis=0)
    return images, img_shape

def save_images_to_npz():
    '''
    Comparess NumPy array to npz.
    '''
    image_path, label_path, output_file = get_path()
    images = load_images_from_folder(image_path)
    labels = load_images_from_folder(label_path)
    np.savez_compressed(output_file, image = images, label = labels)
    return images, labels

def get_path(data_name = "DRIVE"):
    '''
    Return the path to contruct the npz file.
    '''
    path = os.getcwd()
    folder_name = "/data/retina/"

    image_path = f"{path}{folder_name}/{data_name}/image"
    label_path = f"{path}{folder_name}/{data_name}/label"
    npz_path = f"{path}/data/{data_name}_images_and_labels.npz"

    return image_path, label_path, npz_path
    
def print_path(data_name = "DRIVE"):
    image_path, label_path, npz_path = get_path(data_name)
    print(f"Image path: {image_path}")
    print(f"Label path: {label_path}")
    print(f"npz path: {npz_path}")
   