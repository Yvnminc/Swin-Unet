'''
File name: img_plot.py
Authors: Yanming Guo
Description: Plot the image from NumPy array.
'''
import numpy as np
import matplotlib.pyplot as plt
import random

# The dictionary is contructed by following:
# "Data set name" : ((image shape), (label shape))
img_arg = {
    "DRIVE": ((584, 565, 3), (584,565))
       }

def plot_1d_image(image, img_name = "DRIVE"):
    '''
    Print the given 1d image and reshape.

    Param: image in NumPy format.
    '''
    # Given Height, Weight and Channel of image
    H, W, C = img_arg[img_name][0]

    # Reshape the image to given dimensions
    image_array = image.reshape((H, W, C))

    # Plot matplotlib to show RGB image
    plt.imshow(image_array)
    plt.axis('off')  # Cancel axis
    plt.show()

def plot_image(image):
    '''
    Print the given image.

    Param: image in NumPy format.
    '''
    # image = np.transpose(image, (1, 0, 2))
    # Plot matplotlib to show RGB image
    plt.imshow(image)
    plt.show()

def plot_random_image(images, labels = None, img_name = "DRIVE", index = None):
    '''
    Print the image or label in the image dataset.

    Param: 
    
    images: images is a NumPy format array,
    where with the shape (image_features, number_of_images).
    '''
    if index == None:
        image_num = images.shape[1] # get the image number
        index = random.randint(0, image_num - 1)
    image = images[:, index]

    if labels is not None:
        label = labels[:, index]
        plot_image_with_label(image, label, img_name)
    else:
        plot_image(image)

def plot_1d_image_with_label(image, label, img_name = "DRIVE"):
    # Given Height, Weight and Channel of image
    H, W, C = img_arg[img_name][0]

    # Reshape the 1D NumPy arrays into the specified dimensions
    image_array_1 = image.reshape((H, W, C))
    image_array_2 = label.reshape((H, W))

    # Use matplotlib to display RGB and grayscale images side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(image_array_1)
    ax1.axis('off')  # Remove axis
    ax1.set_title(f"{img_name} image (RGB)")

    ax2.imshow(image_array_2, cmap='gray')
    ax2.axis('off')  # Remove axis
    ax2.set_title(f"{img_name} label (Grayscale)")

def plot_image_with_label(image, label, img_name = "DRIVE"):
    # image = np.transpose(image, (1, 0, 2))
    # Use matplotlib to display RGB and grayscale images side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(image)
    ax1.axis('off')  # Remove axis
    ax1.set_title(f"{img_name} image (RGB)")

    ax2.imshow(label, cmap='gray')
    ax2.set_title(f"{img_name} label (Grayscale)")

def plot_tensor(image):
    image = image.numpy()

    # ensure the shape (H, W, C)
    # assert image.shape == (584, 565, 3)

    # Use Matplotlib to plot
    plt.imshow(image)
    plt.show()

def plot_tensor_image_with_label(image, label, img_name = "DRIVE"):
    # image = np.transpose(image, (1, 0, 2))
    # Use matplotlib to display RGB and grayscale images side by side
    image = image.numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(image)
    ax1.set_title(f"{img_name} image (RGB)")

    ax2.imshow(label, cmap='gray')
    ax2.set_title(f"{img_name} label (Grayscale)")