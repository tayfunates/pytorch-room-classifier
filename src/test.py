from stats import AverageMeter
import time
import torch

def test(model, testloader, config):
    batch_time = AverageMeter()
    acc = AverageMeter()

    model.eval()

    with torch.no_grad():
        for test_idx, (imgs, labels) in enumerate(testloader):
            end = time.time()

            if config.run_gpu:
                imgs, labels = imgs.cuda(), labels.cuda()

            y = model(imgs)

            accuracy = (labels.eq(torch.argmax(y, 1))).sum()
            acc.update(accuracy.item(), labels.shape[0])

            batch_time.update(time.time() - end)

    print("Final Accuracy:", acc.avg)
    return acc


