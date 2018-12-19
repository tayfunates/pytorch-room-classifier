import torch
import torch.backends.cudnn as cudnn
from config import *
from dataset import Rooms
from dataset import RoomImageDataset
from torch.utils.data import DataLoader
from model import Room_Classifier
import time
import numpy as np
import datetime
from train import *

if __name__ == '__main__':
    config = Config()

    cudnn.benchmark = True
    torch.cuda.manual_seed_all(config.seed)

    data = Rooms()

    trainLoader = DataLoader(
        RoomImageDataset(data.train_set, data.train_labels, transform=config.trainTransform),
        batch_size=config.no_of_train_batches, shuffle=True, num_workers=config.workers,
        pin_memory=config.run_gpu, drop_last=True,
    )

    testLoader = DataLoader(
        RoomImageDataset(data.test_set, data.test_labels, transform=config.testTransform),
        batch_size=1, shuffle=False, num_workers=config.workers,
        pin_memory=config.run_gpu, drop_last=False,
    )

    valLoader = DataLoader(
        RoomImageDataset(data.val_set, data.val_labels, transform=config.testTransform),
        batch_size=1, shuffle=False, num_workers=config.workers,
        pin_memory=config.run_gpu, drop_last=False,
    )

    #ToDo: Consider loading already existing model for fine tuning and testing
    model = Room_Classifier(config)

    if config.run_gpu:
        model.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    if config.test_only:
        if config.model_path != "":
            checkpoint = torch.load(config.model_path)
            model.load_state_dict(checkpoint['state_dict'])
            #ToDO: Run test code

    start_epoch = 0
    start_time = time.time()
    train_time = 0
    best_acc = -np.inf
    print("Started training...")

    for epoch in range(start_epoch, config.no_of_epochs):
        start_train_time = time.time()
        train(epoch, model, criterion, optimizer, trainLoader, config)
        train_time += round(time.time() - start_train_time)

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))