import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import argparse
import random

class_num = 3
size = 60

centers = [[-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]]


for i in range(class_num):
    theta = 0
    radius = 1
    X = []
    thetas_all = np.array
    for j in range(7):
        thetas = []
        num = math.ceil((radius * size) / 28)
        thetas = np.random.uniform(theta, theta + 0.5 * math.pi, size=num)
        x1 = radius * np.cos(thetas)
        x2 = radius * np.sin(thetas)
        x = np.c_[x1, x2]
        x = x + centers[j]
        X.extend(x)

        theta += 0.5 * math.pi
        radius += 1



    cut_num = len(X) - size
    cut_indexes = random.sample(range(size), cut_num)

    for k in range(cut_num):
        del X[cut_indexes[k]]

    X = np.array(X)

    if i == 0:
        X1 = X
    elif i == 1:
        X2 = X

        rotate_angle = (2 / 3) * math.pi
        rotate_matrix = [[np.cos(rotate_angle), np.sin(rotate_angle)], [-np.sin(rotate_angle), np.cos(rotate_angle)]]
        X2 = np.dot(X2, rotate_matrix)

        # for j in range(len(X2[:, 0])):
        #     chord = math.sqrt(math.pow(X2[j, 0], 2) + math.pow(X2[j, 1], 2))
        #     if X2[j, 1] >= 0:
        #         theta_self = math.acos(X2[j, 0]/chord)
        #     else:
        #         theta_self = 2 * math.pi - math.acos(X2[j, 0] / chord)
        #     theta_target = theta_self + (2/3) * math.pi
        #     X2[j, 0] = chord * math.cos(theta_target)
        #     X2[j, 1] = chord * math.sin(theta_target)
    elif i == 2:
        X3 = X

        rotate_angle = (4 / 3) * math.pi
        rotate_matrix = [[np.cos(rotate_angle), np.sin(rotate_angle)], [-np.sin(rotate_angle), np.cos(rotate_angle)]]
        X3 = np.dot(X3, rotate_matrix)

        # for j in range(len(X3[:, 0])):
        #     chord = math.sqrt(math.pow(X3[j, 0], 2) + math.pow(X3[j, 1], 2))
        #     if X3[j, 1] >= 0:
        #         theta_self = math.acos(X3[j, 0]/chord)
        #     else:
        #         theta_self = 2 * math.pi - math.acos(X3[j, 0] / chord)
        #     theta_target = theta_self + (4/3) * math.pi
        #     X3[j, 0] = chord * math.cos(theta_target)
        #     X3[j, 1] = chord * math.sin(theta_target)

X = np.concatenate((X1, X2, X3), axis=0)

y1 = np.zeros(size)
y2 = np.ones(size)
y3 = y2 * 2
y = np.concatenate((y1, y2, y3), axis=0)

print(len(X1[:, 0]))
print(len(X2[:, 0]))
print(len(X3[:, 0]))

save_X = pd.DataFrame(X)
save_y = pd.DataFrame(y)
save_X.to_csv('train_samples.csv', index=False, header=False)
save_y.to_csv('train_labels.csv', index=False, header=False)

plt.scatter(X[:, 0], X[:, 1], edgecolors='k', c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim([-12, 12])
plt.ylim([-12, 12])
plt.title("trainset")
plt.savefig('trainset.png')
plt.show()