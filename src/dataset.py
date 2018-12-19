from config import *
import os.path as osp
import numpy as np
from PIL import Image
import glob

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class RoomImageDataset(Dataset):
    def __init__(self, imgs, labels, transform=None):
        self.imgs = imgs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

class Rooms(object):

    def __init__(self, **kwargs):
        print("Room dataset is being loaded")

        self.config = Config()

        if not osp.exists(self.config.data_path):
            raise RuntimeError("'{}' does not exist".format(self.config.data_path))

        full_X, full_Y = self.readDataFolder(self.config.data_path)
        self.train_set, self.train_labels, self.test_set, self.test_labels,  self.val_set, self.val_labels = self.splitDataSet(full_X, full_Y)

        if len(self.train_set) != len(self.train_labels):
            raise RuntimeError("Train set and train labels have different sizes")

        if len(self.test_set) != len(self.test_labels):
            raise RuntimeError("Test set and test labels have different sizes")

        if len(self.val_set) != len(self.val_labels):
            raise RuntimeError("Val set and val labels have different sizes")

        self.train_size = len(self.train_set)
        self.test_size = len(self.test_set)
        self.val_size = len(self.val_set)

        print("Room dataset is loaded")
        print("Train set contains {:5d} images".format(self.train_size))
        print("Test set contains {:5d} images".format(self.test_size))
        print("Val set contains {:5d} images".format(self.val_size))

    def getActiveLabels(self):
        return ["fastfood_restaurant", "children_room", "bathroom", "closet", "tv_studio", "computerroom", "clothingstore", "gym", "auditorium", "classroom", "bar", "garage", "dining_room", "florist", "bakery"]


    def readDataFolder(self, path):
        label_directories = self.getActiveLabels()

        x_list = []
        y_list = []
        for label in range(len(label_directories)):
            label_str = label_directories[label]
            #We use only a subset of all labels


            label_directory = osp.join(path, label_str)
            label_image_paths = glob.glob(osp.join(label_directory, '*.jpg'))
            for label_image_path in label_image_paths:
                img = Image.open(label_image_path).convert('RGB')
                x_list.append(img)
                y_list.append(label)
        return x_list, y_list

    def splitDataSet(self, full_X, full_Y):
        #Use stratify to balance labels
        X_train, X_test, Y_train, Y_test = train_test_split(full_X, full_Y, test_size = 0.2, random_state = 42, stratify=full_Y)
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.2, random_state = 42, stratify=Y_train)

        return X_train, Y_train, X_test, Y_test, X_val, Y_val
