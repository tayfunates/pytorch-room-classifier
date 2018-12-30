from stats import AverageMeter
import time
import torch

import itertools
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def test(model, testloader, config):
    batch_time = AverageMeter()
    acc = AverageMeter()

    model.eval()

    y_test = []
    y_pred = []

    with torch.no_grad():
        for test_idx, (imgs, labels) in enumerate(testloader):
            end = time.time()
            if config.draw_confusion_matrix:
                print(test_idx)

            if config.run_gpu:
                imgs, labels = imgs.cuda(), labels.cuda()

            y_test.append(labels.data.cpu().numpy()[0])

            y = model(imgs)
            max_y = torch.argmax(y, 1)

            y_pred.append(max_y.data.cpu().numpy()[0])

            accuracy = (labels.eq(max_y)).sum()
            acc.update(accuracy.item(), labels.shape[0])

            batch_time.update(time.time() - end)

    print("Final Accuracy:", acc.avg)

    if config.draw_confusion_matrix:
        class_names = ["fastfood_restaurant", "children_room", "bathroom", "closet", "tv_studio", "computerroom",
                       "clothingstore", "gym", "auditorium", "classroom", "bar", "garage", "dining_room", "florist",
                       "bakery"]
        # Compute confusion matrix
        cnf_matrix = confusion_matrix(y_test, y_pred)
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization')

        # Plot normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Normalized confusion matrix')

        plt.show()

    return acc


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

