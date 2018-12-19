from stats import  AverageMeter
import time
import torch

def train(epoch, model, criterion, optimizer, trainloader, config):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()

    end = time.time()

    for batch_idx, (imgs, labels) in enumerate(trainloader):
        data_time.update(time.time() - end)

        if config.run_gpu:
            imgs, labels = imgs.cuda(), labels.cuda()

        outputs = model(imgs)

        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), 1)

        batch_time.update(time.time() - end)
        end = time.time()

        if (batch_idx +1) % config.train_print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch+1, batch_idx +1, len(trainloader), batch_time=batch_time,
                data_time=data_time, loss=losses))