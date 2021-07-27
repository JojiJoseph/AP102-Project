
import numpy as np
import toml
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
import time

from unicycle import simulate_unicycle
from collision_check import circle_collision_check
from lidar import Lidar

dt = 0.1
# pred_horizon = 10

config_params = toml.load("config.toml")['params']
print(config_params)
locals().update(config_params)
print(dt, V_MAX)

grid_data = None
grid_res = 1


class DWA:
    def __init__(self, grid_data, ref_path, start_pose, goal_threshold=0.3, grid_res=1) -> None:
        self.grid_data = grid_data
        self.reality = grid_data.copy()
        self.ref_path = ref_path
        self.start_pose = start_pose
        self.goal_threshold = goal_threshold
        self.grid_res = grid_res
        self.path_index = 0
        self.pose = start_pose
        self.v, self.w = 0.0, 0.0
        self.failed_attempts = -1
        self.logs = []
        self.path_index = 0
        self.lidar = Lidar()
        self.lidar.set_env(self.reality, self.grid_res)

    def _command_window(self, v, w, dt=0.1):
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

    def _track(self, ref_path, pose, v, w, dt=0.1, grid_data=grid_data,
               detect_collision=True, grid_res=grid_res):
        commands = self._command_window(v, w, dt)
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
                    print("local path has a collision")
                    continue
            else:
                distance = np.inf
            # calculate cross-track error
            # can use a simplistic definition of
            # how close is the last pose in local path from the ref path

            cte = np.linalg.norm(
                ref_path[-1, 0:2]-local_path[-1, 0:2]) #/ len(local_path)
            # print(cte)

            # other cost functions are possible
            # can modify collision checker to give distance to closest obstacle
            cost = w_cte*cte + w_speed*np.abs(V_MAX - v)**2 + w_obs / distance

            # check if there is a better candidate
            if cost < best_cost:
                best_cost, best_command = cost, [v, w]

        if best_command:
            return best_command
        else:
            return [0, 0]

    def __iter__(self):
        self.path_index = 0
        self.logs = []
        return self

    def reset(self):
        self.path_index = 0
        self.logs = []
        return self

    def __next__(self):
        # pred_horizon = 20
        print(pred_horizon)

        if self.path_index > len(self.ref_path)-1:
            raise StopIteration
        local_ref_path = self.ref_path[self.path_index:self.path_index+pred_horizon]
        # print(self.ref_path.shape, local_ref_path.shape)
        # print(self.pose[1])
        # a = (local_ref_path[:, 0]-self.pose[0])
        # b = (local_ref_path[:, 1]-self.pose[1])
        # c = (np.hypot(a,b))
        # print(self.goal_threshold, np.min(c))
        if self.goal_threshold > np.min(np.hypot(local_ref_path[:, 0]-self.pose[0],
                                                 local_ref_path[:, 1]-self.pose[1])):
            candidate_jump = np.argmin(
                np.hypot(local_ref_path[:, 0]-self.pose[0],
                         local_ref_path[:, 1]-self.pose[1]))
            self.path_index = self.path_index + 1 + \
                1*candidate_jump#*(candidate_jump < jump_distance)

        self.failed_attempts += 1
        if self.failed_attempts > 1600:
            self.path_index += 1
            self.failed_attempts = -1
        # get next command
        self.v, self.w = self._track(local_ref_path, self.pose, self.v, self.w, dt,
                                     detect_collision=True, grid_data=self.grid_data)

        # simulate vehicle for 1 step
        # remember the function now returns a trajectory, not a single pose
        self.pose = simulate_unicycle(self.pose, self.v, self.w, N=1, dt=dt)[0]

        self.lidar.set_env(self.reality, self.grid_res)
        distances, collision_points = self.lidar.sense_obstacles(self.pose)
        # Add obstacles to grid data
        for point in collision_points:
            if point[0] != -1:
                i, j = point
                self.grid_data[i, j] = 1
                self.reality = self.grid_data.copy()

        # update logs
        self.logs.append([*self.pose, self.v, self.w, self.path_index])
        print(self.path_index)
        return np.array(self.logs), distances
