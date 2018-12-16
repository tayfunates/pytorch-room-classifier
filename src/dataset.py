from config import *
import os.path as osp

class Room(object):

    def __init__(self, **kwargs):
        self.config = Config()
        self.train_path = osp.join(self.config.data_path, 'train')
        self.test_path = osp.join(self.config.data_path, 'test')
        self.val_path = osp.join(self.config.data_path, 'val')

        self.checkDataExists()

        self.train_set = self.readDataFolder(self.train_path)
        self.test_set = self.readDataFolder(self.test_path)
        self.val_set = self.readDataFolder(self.validation_path)

        self.train_size = len(self.train_set)
        self.test_size = len(self.test_set)
        self.val_size = len(self.val_set)

        print("Room dataset is loaded")
        print("Train set contains {:5d} images", self.train_size)
        print("Test set contains {:5d} images", self.test_size)
        print("Val set contains {:5d} images", self.val_size)

    def checkDataExists(self):
        if not osp.exists(self.config.data_path):
            raise RuntimeError("'{}' does not exist".format(self.config.data_path))
        if not osp.exists(self.train_path):
            raise RuntimeError("'{}' does not exist".format(self.train_path))
        if not osp.exists(self.test_path):
            raise RuntimeError("'{}' does not exist".format(self.test_path))
        if not osp.exists(self.val_path):
            raise RuntimeError("'{}' does not exist".format(self.val))

    def readDataFolder(self, path):
        dataset = []
        return dataset

