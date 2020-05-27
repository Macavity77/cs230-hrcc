#this is used to plot the graph of the training and testing loss
#import numpy as np
import matplotlib.pyplot as plt

def getLoss(loss):
    return float(loss[0:len(loss) - 4])

def getAccuracy(top1, top5, top10):
    accurancy = []
    accurancy.append(float(top1))
    accurancy.append(float(top5))
    accurancy.append(float(top10[3:].rstrip('.')))
    return accurancy

steps = []
losses = []
accurancy_top1 = []
accurancy_top5 = []
accurancy_top10 = []
f_train = open("m2_model_train.txt")
for line in f_train:
    line = line.split()
    steps.append(line[1])
    losses.append(getLoss(line[8]))
    t1, t5, t10 = getAccuracy(line[13], line[15], line[16])
    accurancy_top1.append(t1)
    accurancy_top5.append(t5)
    accurancy_top10.append(t10)

def getWin(top1, top5, top10):
    win = []
    win.append(float(top1))
    win.append(float(top5))
    win.append(float(top10.rstrip('.')))
    return win

pics = []
result_top1 = []
result_top5 = []
result_top10 = []
f_test = open("m2_model_test.txt")
for line in f_test:
    line = line.split()
    pics.append(line[5])
    t1, t5, t10 = getWin(line[9], line[12], line[14])
    result_top1.append(t1)
    result_top5.append(t5)
    result_top10.append(t10)

plt.plot(steps, losses, 'r-', linewidth=2.0)
plt.xlabel('training steps')
plt.ylabel('training loss')
plt.show()


# Create plots with pre-defined labels.
fig, ax = plt.subplots()
ax.plot(steps, accurancy_top1, 'r-', linewidth=1.5, label='accurancy for top1 result')
ax.plot(steps, accurancy_top5, 'g-', linewidth=1.5, label='accurancy for top5 result')
ax.plot(steps, accurancy_top10, 'b-', linewidth=1.5, label='accurancy for top10 result')

legend = ax.legend(loc='lower right', shadow=True, fontsize='x-large')

# Put a nicer background color on the legend.
legend.get_frame()

plt.show()

fig, ax = plt.subplots()
ax.plot(pics, result_top1, 'r-', linewidth=1.5, label='accurancy for top1 test result')
ax.plot(pics, result_top5, 'g-', linewidth=1.5, label='accurancy for top5 test result')
ax.plot(pics, result_top10, 'b-', linewidth=1.5, label='accurancy for top10 test result')

legend = ax.legend(loc='lower right', shadow=True, fontsize='x-large')
ax.set_ylim(0.9, 1)
ax.set_title("Model performance",fontsize=14)

ax.set_xlabel("Number of test files",fontsize=12)

ax.set_ylabel("Accurancy",fontsize=12)

# Put a nicer background color on the legend.
legend.get_frame()

plt.show()
