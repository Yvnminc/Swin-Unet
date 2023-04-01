'''
File name: npz_dataloader.py
Authors: Yanming Guo
Description: Wrapping numpy data into a pytorch dataset.
'''
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import numpy as np
import random
from .img_to_npz import get_path
from .img_plot import plot_tensor_image_with_label
import h5py
import os

class Npz_dataset(Dataset):
    def __init__(self, data_name = "DRIVE",
                 split_ratio = (0.7, 0.2, 0.1),
                 transform = transforms.ToTensor(),
                 pirnt_info = False):
        """
        Init data instance from pytorch Dataset and load data,
        and split the train and validation set.
        Args:
            img_dir: the path of the dataset.
            transform: whether need to apply transform.
        """
        self.transform = transform
        self.data_name = data_name
        self.image_path, self.label_path, self.img_dir = get_path(data_name)
        self.split_ratio = split_ratio
        self.dataset = np.load(self.img_dir)
        
        # Assign image and label
        self.image = self.dataset["image"]
        self.label = self.dataset["label"]

        # The data is in shape of (n_features, n_samples)
        self.n_sample = self.image.shape[0]

        # Cal train, validation, test set data length
        self.train_valid_test_len()
        self.random_split()

        # Print information of the dataset
        if pirnt_info:
            self.print_info()

    def __getitem__(self, index):
        """ 
        Get item for iteration.
        """
        image = self.image[index]
        label = self.label[index]

        # From 3D image to tensor
        image_tensor = torch.from_numpy(image)

        # If image type is uint8（range [0, 255]），zoom to [0.0, 1.0]
        if image_tensor.dtype == torch.uint8:
            image_tensor = image_tensor.float() / 255.0

        return image_tensor, label
    
    def __len__(self):
        """
        Total number of training instance.
        """
        return self.n_sample
    
    def random_split(self):
        """
        Random split the data.
        """
        self.train, self.valid, self.test = random_split(self, 
        [self.n_train, self.n_valid, self.n_test],
        generator=torch.Generator().manual_seed(42))
    
    def get_loader(self, batch_size =128, num_workers = 1):
        """
        Create the dataloader for each dataset.
        Args:
            batch_size:     the sample size for each bath.
            num_workers:    multi-process.
        """
        train_loader = DataLoader(self.train, batch_size= batch_size, num_workers= num_workers)
        val_loader = DataLoader(self.valid, batch_size= batch_size, num_workers= num_workers)
        test_loader = DataLoader(self.valid, batch_size= batch_size, num_workers= num_workers)

        return train_loader, val_loader, test_loader
    
    def train_valid_test_len(self):
        # dataset length
        self.n_train = int(self.n_sample * self.split_ratio[0])
        self.n_valid = int(self.n_sample *  self.split_ratio[1])
        self.n_test = self.n_sample - self.n_train - self.n_valid
        

    def print_info(self):
        print(f"-----------------------{self.data_name}--------------------------")
        print(f"Number of total data: {self.n_sample}")
        print(f"Image shape: {self.image.shape}")
        print(f"Label shape: {self.label.shape}")
        print(f"--------------Split by {self.split_ratio}---------------")
        print(f"Train size: {self.n_train}")
        print(f"Validation size: {self.n_valid}")
        print(f"Test size: {self.n_test}")

    def plot_random_image(self):
        random_index = random.randint(0, self.n_sample -1)

        # Get train_dataset's image and label
        image, label = self[random_index]

        plot_tensor_image_with_label(image, label, self.data_name)

class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample