import time
import matplotlib.pyplot as plt
import numpy as np
import toml

from astar import Astar
from dwa import DWA
# from itertools import izip

config_params = toml.load("config.toml")['params']
print(config_params)
locals().update(config_params)

import cv2

# Load image
img = cv2.imread("./circuit.png")

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.

plt.imshow(img, cmap="gray")
plt.show()

grid_res = 0.05

img2 = cv2.resize(
    img, (int(img.shape[1]*grid_res), int(img.shape[0]*grid_res)), cv2.INTER_NEAREST)

img2[img2 < 0.5] = 0
img2[img2 > 0] = 1

astar_ = Astar(1-img2)

print(img2.shape)
path = astar_.shortest_path((3, 3), (30, 24))

x, y = path
interp_range = len(x)*2
x = np.interp(np.linspace(0, 1, interp_range), np.linspace(0, 1, len(x)), x)
y = np.interp(np.linspace(0, 1, interp_range), np.linspace(0, 1, len(y)), y)
pre_x, pre_y = 0, 0

t = []
for x_c, y_c in zip(x, y):
    t.append(np.arctan2(y_c-pre_y, x_c-pre_x))
    pre_x, pre_y = x_c, y_c
t = np.array(t)

path = np.array([x, y, t], dtype=np.float).T

dwa_obj = DWA(1-img2.T, path, path[0])


plt.imshow(1-dwa_obj.grid_data.T, cmap="Accent")
plt.show()

plt.figure(figsize=(20, 20))
plt.imshow(1-dwa_obj.grid_data.T, cmap="gray", interpolation=None)
plt.scatter(x, y)
plt.legend()

i = 0
for progress, distances, target_path in dwa_obj:

    tracked_x, tracked_y = progress[:, 0], progress[:, 1]
    plt.clf()
    plt.imshow(1-dwa_obj.grid_data.T, cmap="gray", interpolation=None)
    plt.scatter(x, y, label="A* path")
    plt.plot(tracked_x, tracked_y, c="red", label="tracked")
    plt.scatter(tracked_x[-1], tracked_y[-1], label="Robot")
    idx = int(progress[-1, 5])
    # print(idx, type(idx))
    x_target = x[idx: idx+pred_horizon]
    y_target = y[idx: idx+pred_horizon]
    if target_path is not None:
        plt.plot(target_path[:,0],target_path[:,1], label="Target path")
    # print(target_path.shape)
    plt.scatter(x_target, y_target, label="Prediction Horizon")
    # plt.scatter(tracked_x[-1], tracked_y[-1], label="Robot")
    plt.legend()
    plt.pause(0.001)
    i += 1
    if i > 400:
        break


print(np.unique(dwa_obj.grid_data.T))
# plt.figure()
plt.plot(tracked_x, tracked_y, c="red", label="tracked")
plt.scatter(x, y)
plt.legend()
plt.show()
