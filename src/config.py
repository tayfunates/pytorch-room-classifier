import os
import torch

class Config(object):
    def __init__(self):
        # Basic Parameters
        self.data_path = ''
        self.test_only = False
        self.model_path = '' #If test only mode is on, test the performance of this model, otherwise finetune this model
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

        #cuda parameters
        self.devices = "0"
        torch.manual_seed(self.seed)
        os.environ['CUDA_VISIBLE_DEVICES'] = self.devices
        self.run_gpu = torch.cuda.is_available()