# room-classifier

Includes networks used for room classification developed for a homework in the Deparment of Computer Engineering, Hacettepe University.

./src folder contains a couple of source files for the implementation of the homework.

./src/config.py contains the values of several parameters with which training and testing are running. Several experiments are run by updating this file.
./src/main.py contains the main function which responsible training and testing the model.
./src/dataset.py contains two classes which are responsible from reading and iterating over the room dataset.
./src/stats.py contains AverageMeter class which is used to hold information for several values such as loss and accuracy. It is adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
./src/model.py contains my implementation of the classifier.
./src/resnet18.py contains resnet18 implementation of the classifier. For the first run, model downloads and caches parameter values which are trained on Imagenet dataset.
./src/train.py contains train function of the models for an epoch.
./src/test.py contains test function for a pass of the data provided.
./src/plot_train.py draws some plots on top of training process text files

During execution, software creates a folder ./src/log and saves useful information such as checkpoints of the models. Training and testing process can be watched from command line.

Software is executed with both CPU during development and with GPU for extensive experiments. A CPU version of pytorch 1.0.0 is used to develop the software and a GPU version of pytorch 0.4.1.post2 is used for extensive testing. Numpy version of the environment is 1.15.4. Sklearn version of the environment is 0.20.1. By providing a similar environment, software is ready to be run after only changing the folder for the dataset from config.py.

Besides information regarding software, report.pdf contains information about the experiments conducted and their results.
