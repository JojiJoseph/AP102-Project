from astar import Astar
import matplotlib.pyplot as plt
from dwa import DWA
import numpy as np

import cv2

# Load image
img = cv2.imread("./circuit.png")

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.

plt.imshow(img, cmap="gray")
plt.show()

grid_res = 0.05

img2 = cv2.resize(
    img, (int(img.shape[1]*grid_res), int(img.shape[0]*grid_res)), cv2.INTER_NEAREST)

astar_ = Astar(1-img2)

print(img2.shape)
path = astar_.shortest_path((3, 3), (30, 24))

x, y = path

dwa_obj = DWA(img2.T, np.array([x, y]), (3, 3))

print(x, y)
