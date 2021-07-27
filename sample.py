from astar import Astar
import matplotlib.pyplot as plt
from dwa import DWA
import numpy as np
# from itertools import izip

import cv2

# Load image
img = cv2.imread("./circuit.png")

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.

plt.imshow(img, cmap="gray")
plt.show()

grid_res = 0.05

img2 = cv2.resize(
    img, (int(img.shape[1]*grid_res), int(img.shape[0]*grid_res)), cv2.INTER_NEAREST)

img2[img2<0.5] = 0
img2[img2 > 0] = 1

astar_ = Astar(1-img2)

print(img2.shape)
path = astar_.shortest_path((3, 3), (30, 24))

x, y = path
interp_range = len(x)*3
x = np.interp( np.linspace(0,1,interp_range), np.linspace(0,1,len(x)), x)
y = np.interp( np.linspace(0,1,interp_range), np.linspace(0,1,len(y)), y)
pre_x, pre_y = 0, 0

t = []
for x_c, y_c in zip(x, y):
    t.append(np.arctan2(y_c-pre_y, x_c-pre_x))
    pre_x, pre_y = x_c, y_c
t = np.array(t)
# print((x))
# print((y))
# print((t))
path = np.array([x, y, t], dtype=np.float).T
# print(path)
# print(path.shape, path.dtype)
# print(path[0])

dwa_obj = DWA(1-img2.T, path, path[0])

import time

plt.imshow(1-dwa_obj.grid_data.T, cmap="Accent")
plt.show()
i = 0
for progress, distances in dwa_obj:
    # progress, distances = next(dwa_obj)
    # plt.pause(0.001)
    # print(progress)
    # time.sleep(0.001)
    tracked_x, tracked_y = progress[:,0], progress[:,1]
    # i += 10
    # if i > 10000:
    #   break
    # pass

print(np.unique(dwa_obj.grid_data.T))
plt.figure()
plt.imshow(1-dwa_obj.grid_data.T, cmap="gray", interpolation=None)
plt.plot(tracked_x, tracked_y, C="red", label="tracked")
plt.scatter(x, y)
plt.show()