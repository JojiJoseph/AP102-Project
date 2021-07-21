import numpy as np
import matplotlib.pyplot as plt
# choose resolution and dimensions
grid_res = 1
grid_span = 15  # square circuit dimensions in m

# calculate grid_shape from grid
# dimensions have to be integers
grid_shape = (np.array([grid_span]*2)/grid_res).astype('int')
# Initialize
grid_data = np.zeros(grid_shape)

# Create rectangular obstacles in world co-ordinates
#xmin, xmax, ymin, ymax
obstacles = np.array([[5, 6, 7, 9]])  # [[25, 26, 10, 40],
# [2, 8, 16, 20]])
for obs in obstacles:
    # calculate obstacles extent in pixel coords
    xmin, xmax, ymin, ymax = (obs/grid_res).astype('int')
    # mark them as occupied
    grid_data[xmin:xmax, ymin:ymax] = 1.0

# calculate the extents
x1, y1 = 0, 0
x2, y2 = grid_span, grid_span



w = 0.8
l = 1.2
mule_extents = np.array([[-w/2, -l/2],
                         [w/2, -l/2],
                         [w/2, l/2],
                         [-w/2, l/2],
                         [-w/2, -l/2]])


r = 0.5
l = 0.4
circles = [(0, 0, r), (0, l, r), (0, -l, r)]


def circle_collision_check(grid, local_traj, grid_res=grid_res):
    xmax, ymax = grid.shape
    all_x = np.arange(xmax)
    all_y = np.arange(ymax)
    X, Y = np.meshgrid(all_x, all_y)

    # print("In circle collision check")
    occupied_pt = grid[X, Y] == 1

    min_distance_2 = np.inf
    x, y, t = local_traj[:, 0], local_traj[:, 1], local_traj[:, 2]
    rot = np.array([[np.sin(t), -np.cos(t)], [np.cos(t), np.sin(t)]])
    rot = rot.transpose(2, 0, 1)
    n = local_traj.shape[0]
    for xc, yc, rc in circles:
        xy_rot = rot @ np.array([xc, yc]) + np.array([x, y]).T
        xc_rot, yc_rot = xy_rot[:, 0], xy_rot[:, 1]
        xc_pix, yc_pix = (xc_rot/grid_res), (yc_rot / grid_res)
        rc_pix = rc / grid_res
        X_dash = np.repeat(X[:, :, np.newaxis], n, axis=2)
        Y_dash = np.repeat(Y[:, :, np.newaxis], n, axis=2)
        inside_circle = ((X_dash-xc_pix)**2 + (Y_dash-yc_pix)
                         ** 2 - rc_pix**2 < 0).any(axis=2)
        if np.sum(np.multiply(inside_circle, occupied_pt)):
            return True, np.inf
    distance_squared = np.min(
        (X_dash[occupied_pt]-xc_pix)**2+(Y_dash[occupied_pt]-yc_pix)**2)
    return False, np.sqrt(distance_squared)
