import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import toml

from astar import Astar
from dwa import DWA

config_params = toml.load("config.toml")['params']
print(config_params)
locals().update(config_params)

# Load image
img = cv2.imread("./circuit.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.

# plt.imshow(img, cmap="gray")
# plt.show()

grid_res = 0.05
downsampled_image_shape = (
    int(img.shape[1]*grid_res), int(img.shape[0]*grid_res))
img_downsampled = cv2.resize(
    img, downsampled_image_shape, cv2.INTER_NEAREST)

# Making it binary
img_downsampled[img_downsampled < 0.5] = 0
img_downsampled[img_downsampled > 0] = 1

# Running A* algorithm
astar_ = Astar(1-img_downsampled)
path = astar_.shortest_path((3, 3), (30, 24))

x, y = path
interp_range = len(x)*2
x = np.interp(np.linspace(0, 1, interp_range), np.linspace(0, 1, len(x)), x)
y = np.interp(np.linspace(0, 1, interp_range), np.linspace(0, 1, len(y)), y)

# Calculate thetas
pre_x, pre_y = 0, 0
t = []
for x_c, y_c in zip(x, y):
    t.append(np.arctan2(y_c-pre_y, x_c-pre_x))
    pre_x, pre_y = x_c, y_c
t = np.array(t)


path = np.array([x, y, t], dtype=np.float).T

# Create DWA object
dwa_ = DWA(1-img_downsampled.T, path, path[0])

# plt.imshow(1-dwa_.grid_data.T, cmap="Accent")
# plt.show()

plt.figure(figsize=(20, 20))
plt.imshow(1-dwa_.grid_data.T, cmap="Dark2", interpolation=None)
plt.scatter(x, y)
plt.legend()

i = 0
for progress, distances, target_path in dwa_:

    tracked_x, tracked_y = progress[:, 0], progress[:, 1]
    plt.clf()
    plt.imshow(1-dwa_.grid_data.T, cmap="gray", interpolation=None)
    plt.scatter(x, y, label="A* path")
    plt.plot(tracked_x, tracked_y, c="red", label="tracked")
    plt.scatter(tracked_x[-1], tracked_y[-1], label="Robot")
    idx = int(progress[-1, 5])
    # print(idx, type(idx))
    x_target = x[idx: idx+pred_horizon]
    y_target = y[idx: idx+pred_horizon]
    # print(target_path.shape)
    plt.scatter(x_target, y_target, label="Prediction Horizon")
    x_c, y_c, theta_c = dwa_.pose
    for dist, angle in zip(distances, dwa_.lidar.beam_angles):
        t = angle + theta_c
        plt.plot(np.array([x_c, x_c+dist*np.cos(t)]), np.array([y_c,
                                                                y_c+dist*np.sin(t)]), c="green")
    if target_path is not None:
        plt.plot(target_path[:, 0], target_path[:, 1], label="Target path")
    # plt.scatter(tracked_x[-1], tracked_y[-1], label="Robot")
    plt.legend()
    plt.pause(0.001)
    i += 1
    if i > 400:
        break


plt.legend()
plt.show()
