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
img = cv2.imread("./circuit3.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.
img = np.flip(img, 0)

img_reality = cv2.imread("./circuit3_dynamic_obstacles.png")
img_reality = cv2.cvtColor(img_reality, cv2.COLOR_BGR2GRAY) / 255.
img_reality = np.flip(img_reality, 0)

# plt.imshow(img, cmap="gray")
# plt.show()

grid_res = 0.05
downsampled_image_shape = (
    round(img.shape[1]*grid_res), round(img.shape[0]*grid_res))
img_downsampled = cv2.resize(
    img, downsampled_image_shape, cv2.INTER_NEAREST)

# Making it binary
img_downsampled[img_downsampled > 0.3] = 1
img_downsampled[img_downsampled < 0.99] = 0

reality_downsampled = cv2.resize(
    img_reality, downsampled_image_shape, cv2.INTER_NEAREST)

# Making it binary
reality_downsampled[reality_downsampled > 0.3] = 1
reality_downsampled[reality_downsampled < 0.99] = 0

print("Downsampled image shape", img_downsampled.shape)
plt.imshow(img_downsampled, origin="lower")
plt.title("Track")
plt.show()

# Running A* algorithm
astar_ = Astar(1-img_downsampled)
# circuit 1
start = (5, 3)
goal = (25, 20)
# start = (100,65)
# goal = (542, 370)

# circuit 2
start = (5, 10)
goal = (15, 11)

# circuit 3
start = (5, 5)
goal = (3, 15)

path = astar_.shortest_path( start, goal)

x, y = path
interp_range = len(x)*2
x = np.interp(np.linspace(0, 1, interp_range), np.linspace(0, 1, len(x)), x)
y = np.interp(np.linspace(0, 1, interp_range), np.linspace(0, 1, len(y)), y)

# Calculate thetas
pre_x, pre_y = x[0], y[0]
t = [np.arctan2(y[1]-pre_y, x[1]-pre_x)]
for x_c, y_c in zip(x[1:], y[1:]):
    t.append(np.arctan2(y_c-pre_y, x_c-pre_x))
    pre_x, pre_y = x_c, y_c
t = np.array(t)


path = np.array([x, y, t], dtype=np.float).T

# Create DWA object
dwa_ = DWA(1-img_downsampled.T, path, path[0], goal_threshold=goal_threshold, reality=1-reality_downsampled.T)

plt.figure(figsize=(20, 20))
plt.imshow(1-dwa_.grid_data.T, cmap="Dark2", interpolation=None)
plt.scatter(x, y)
plt.legend()

i = 0
for progress, distances, target_path in dwa_:

    tracked_x, tracked_y = progress[:, 0], progress[:, 1]
    plt.clf()
    plt.imshow(1-dwa_.grid_data.T, cmap="gray", origin="lower")
    plt.imshow(1-dwa_.reality.T, cmap="gray", origin="lower", alpha=0.5)
    plt.scatter(x, y, label="A* path")
    plt.plot(tracked_x, tracked_y, c="red", label="tracked")
    plt.scatter(tracked_x[-1], tracked_y[-1], label="Robot")
    idx = int(progress[-1, 5])

    x_target = x[idx: idx+pred_horizon]
    y_target = y[idx: idx+pred_horizon]

    plt.scatter(x_target, y_target, label="Prediction Horizon")
    x_c, y_c, theta_c = dwa_.pose
    for dist, angle in zip(distances, dwa_.lidar.beam_angles):
        t = angle + theta_c
        plt.plot(np.array([x_c, x_c+dist*np.cos(t)]),
                 np.array([y_c, y_c+dist*np.sin(t)]), c="green")
    if target_path is not None:
        plt.plot(target_path[:, 0], target_path[:, 1], label="Target path")

    plt.legend()
    plt.pause(0.001)
    i += 1
    # if i > 600:
    #     break

print("Simulation Ended!")
plt.legend()
plt.show()
