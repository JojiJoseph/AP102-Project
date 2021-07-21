
import numpy as np
import toml
import matplotlib.pyplot as plt
import time

from unicycle import simulate_unicycle
from collision_check import circle_collision_check

dt = 0.1
pred_horizon = 10

config_params = toml.load("config.toml")['params']
print(config_params)
locals().update(config_params)
print(dt, V_MAX)

grid_data = None
grid_res = 1


def command_window(v, w, dt=0.1):
    """Returns acceptable v,w commands given current v,w"""
    # velocity can be (0, V_MAX)
    # ACC_MAX = max linear acceleration
    v_max = min(V_MAX, v + ACC_MAX*dt)
    v_min = max(0, v - ACC_MAX*dt)
    # omega can be (-W_MAX, W_MAX)
    # W_DOT_MAX = max angular acceleration
    epsilon = 1e-6
    w_max = min(W_MAX, w + W_DOT_MAX*dt)
    w_min = max(-W_MAX, w - W_DOT_MAX*dt)

    # generate quantized range for v and omega
    vs = np.linspace(v_min, v_max, num=11)
    ws = np.linspace(w_min, w_max, num=11)

    # cartesian product of [vs] and [ws]
    # remember there are 0 velocity entries which have to be discarded eventually
    commands = np.transpose([np.tile(vs, len(ws)), np.repeat(ws, len(vs))])

    # calculate kappa for the set of commands
    kappa = commands[:, 1]/(commands[:, 0]+epsilon)

    # returning only commands < max curvature
    return commands[(kappa < K_MAX) & (commands[:, 0] != 0)]


def track(ref_path, pose, v, w, dt=0.1, grid_data=grid_data,
          detect_collision=True, grid_res=grid_res):
    commands = command_window(v, w, dt)
    # initialize path cost
    best_cost, best_command = np.inf, None
    for i, (v, w) in enumerate(commands):
        # Number of steps = prediction horizon
        local_path = simulate_unicycle(pose, v, w, pred_horizon, dt)

        if detect_collision:
            # ignore colliding paths
            hit, distance = circle_collision_check(
                grid_data, local_path, grid_res=grid_res)
            if hit:
                # print("local path has a collision")
                continue
        else:
            distance = np.inf
        # calculate cross-track error
        # can use a simplistic definition of
        # how close is the last pose in local path from the ref path

        cte = np.linalg.norm(
            ref_path[-1, 0:2]-local_path[-1, 0:2]) / len(local_path)
        # print(cte)

        # other cost functions are possible
        # can modify collision checker to give distance to closest obstacle
        cost = w_cte*cte + w_speed*np.abs(V_MAX - v) + w_obs / distance

        # check if there is a better candidate
        if cost < best_cost:
            best_cost, best_command = cost, [v, w]

    if best_command:
        return best_command
    else:
        return [0, 0]


def dwa(grid_data, ref_path, start_pose, goal_threshold=0.3, grid_res=1,
        animate=False):
    pose = start_pose
    logs = []
    jump_distance = 4
    path_index = 0
    v, w = 0.0, 0.0
    failed_attempts = -1

    while path_index < 10:#0.001*len(ref_path)-1:
        print(path_index/len(ref_path))
        local_ref_path = ref_path[path_index:path_index+pred_horizon]
        if goal_threshold > np.min(np.hypot(local_ref_path[:, 0]-pose[0],
                                            local_ref_path[:, 1]-pose[1])):
            candidate_jump = np.argmin(
                np.hypot(local_ref_path[:, 0]-pose[0],
                         local_ref_path[:, 1]-pose[1]))
            path_index = path_index + 1 + \
                candidate_jump*(candidate_jump < jump_distance)

        failed_attempts += 1
        if failed_attempts > 160:
            path_index += 1
            failed_attempts = -1
        # get next command
        v, w = track(local_ref_path, pose, v, w, dt,
                     detect_collision=True, grid_data=grid_data)

        # simulate vehicle for 1 step
        # remember the function now returns a trajectory, not a single pose
        pose = simulate_unicycle(pose, v, w, N=1, dt=dt)[0]

        # update logs
        logs.append([*pose, v, w])
        if animate:
            plt.clf()
            plt.imshow(grid_data.T, cmap="Dark2")
            plt.scatter(pose[0], pose[1], c="red", label="Robot")
            plt.plot(ref_path[:,0],ref_path[:,1], label="Astar Path")
            plt.pause(0.00001)
            # fig.canvas.flush_events()
            # time.sleep(0.1)
    if animate:
        plt.show()
    return np.array(logs)
