import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

filepath = "myTrainingProcess.txt"

losses = []
loss_epochs = list(range(0, 1000))

val_accs = []
val_acc_epochs = list(range(0, 1000, 5))

with open(filepath) as fp:
   for cnt, line in enumerate(fp):
       loss_index=line.find("Loss")
       val_acc_index = line.find("Final Accuracy")
       if loss_index>0:
           lossStr = line[loss_index+5:loss_index+11]
           losses.append(float(lossStr))
       if val_acc_index > 0:
           valStr = line[val_acc_index+18:val_acc_index+25]
           val_accs.append(float(valStr) * 100.0)



plt.figure(1)
plt.subplot(211)
plt.plot(loss_epochs, losses, 'r')
plt.axis([0, 1001, 0, 3])
plt.ylabel('Train Batch Loss')

plt.subplot(212)
plt.plot(val_acc_epochs, val_accs, 'b')
plt.axis([0, 1001, 10, 65])
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy %')
plt.show()