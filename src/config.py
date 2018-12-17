import os
import torch
import os.path as osp
import sys
import torchvision.transforms as T

class Config(object):
    def __init__(self):
        # Basic Parameters
        self.data_path = ""
        self.test_only = False
        self.model_path = "" #If test only mode is on, test the performance of this model, otherwise finetune this model
        self.seed = 1
        self.no_of_classes = 6
        self.model_input_width = 32
        self.model_input_height = 32
        self.model_input_channels = 3

        #train parameters
        self.learning_rate = 0.00001
        self.no_of_train_batches = 128
        self.no_of_epochs = 20
        self.validation_frequency = 10
        self.drop_prob = 0.0
        self.augment_data = False

        self.pretrained_data_mean = []
        self.pretrained_data_std = []
        pretrained = (len(self.pretrained_data_mean) > 0) and (len(self.pretrained_data_std) > 0)

        #cuda parameters
        self.devices = "0"
        torch.manual_seed(self.seed)
        os.environ['CUDA_VISIBLE_DEVICES'] = self.devices
        self.run_gpu = torch.cuda.is_available()

        #log paramaters
        self.log_path = osp.joint("log", "lr{lr:7.5f}_nb{nb}_dp{dp}_da{da}".format(lr=self.learning_rate, nb=self.no_of_train_batches, dp=self.drop_prob, da=self.augment_data))

        #train transform parameters
        transform_array_train = []
        transform_array_train.append(T.Resize((self.model_input_height, self.model_input_width)))
        if self.augment_data:
            transform_array_train.append(T.RandomHorizontalFlip())
            #To do add other types of augmentation

        transform_array_train.append(T.ToTensor())
        if pretrained:
            transform_array_train.append(T.Normalize(mean=self.pretrained_data_mean, std=self.pretrained_data_std))
        self.trainTransform = T.Compose(transform_array_train)

        #test transform parameters
        transform_array_test = []
        transform_array_test.append(T.Resize((self.model_input_height, self.model_input_width)))
        transform_array_test.append(T.ToTensor())
        if pretrained:
            transform_array_test.append(T.Normalize(mean=self.pretrained_data_mean, std=self.pretrained_data_std))
        self.testTransform = T.Compose(transform_array_test)