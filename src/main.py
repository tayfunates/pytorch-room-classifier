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
from test import *
import os
import os.path as osp
import shutil

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

        if config.validation_frequency > 0 and (epoch + 1) % config.validation_frequency == 0 or (epoch + 1) == config.no_of_epochs:
            print("==> Test")
            acc = test(model, valLoader, config)
            is_current_best = acc.avg >= best_acc
            if is_current_best:
                best_acc = acc.avg
                best_epoch = epoch + 1

            state_dict = model.state_dict()

            state = {'state_dict': state_dict,  'acc': acc, 'epoch': epoch }
            checkpoint_path = osp.join(config.log_path, 'epoch_' + str(epoch + 1) + '.pth.tar')

            if not osp.exists(config.log_path):
                os.makedirs(config.log_path)

            torch.save(state, checkpoint_path)
            if is_current_best:
                shutil.copy(checkpoint_path, osp.join(osp.dirname(checkpoint_path), 'best_model.pth.tar'))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))