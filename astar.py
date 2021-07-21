import numpy as np
import heapq

v = 1
dt = 0.1
num_st_pts = int(v/dt)
num_pts = 50

DEBUG = False


def cubic_spiral(theta_i, theta_f, n=10):
    x = np.linspace(0, 1, num=n)
    # -2*x**3 + 3*x**2
    return (theta_f-theta_i)*(-2*x**3 + 3*x**2) + theta_i


def straight(dist, curr_pose, n=num_st_pts, v=v, dt=dt):
    # the straight-line may be along x or y axis
    # curr_theta will determine the orientation
    # At present v is a dummy parameter
    x0, y0, t0 = curr_pose
    xf, yf = x0 + dist*np.cos(t0), y0 + dist*np.sin(t0)
    x = (xf - x0) * np.linspace(0, 1, n) + x0
    y = (yf - y0) * np.linspace(0, 1, n) + y0
    return x, y, t0*np.ones_like(x)


def turn(change, curr_pose, n=num_pts, v=v, dt=dt):
    # adjust scaling constant for desired turn radius
    x0, y0, t0 = curr_pose
    theta = cubic_spiral(t0, t0 + np.deg2rad(change), n)
    x = x0 + np.cumsum(v*np.cos(theta)*dt)
    y = y0 + np.cumsum(v*np.sin(theta)*dt)
    return x, y, theta


def inplace(change, curr_pose, n=num_pts, v=v, dt=dt):
    # v is a dummy variable
    return turn(change, curr_pose, n, 0, dt)


def reverse(dist, curr_pose, n=num_st_pts, v=v,dt=dt):
    return straight(-dist, curr_pose, n=num_st_pts, v=v,dt=dt)


def change_lane(side, curr_pose, n=10, v=v, dt=dt):
    if side > 0: # new lane is in 
        theta = cubic_spiral(curr_pose[2], curr_pose[2]-np.pi/2, n=n).tolist()
        theta.extend(cubic_spiral(curr_pose[2]-np.pi/2,curr_pose[2], n=n).tolist())
        x = curr_pose[0] + np.cumsum(v*np.cos(theta)*dt)
        y = curr_pose[1] + np.cumsum(v*np.sin(theta)*dt)  
    else:
        theta = cubic_spiral(curr_pose[2],curr_pose[2]+np.pi/2,n=n).tolist()
        theta.extend( cubic_spiral(curr_pose[2]+np.pi/2,curr_pose[2],n=n).tolist() )
        x = curr_pose[0] + np.cumsum(v*np.cos(theta)*dt)
        y = curr_pose[1] + np.cumsum(v*np.sin(theta)*dt)
    return x, y, theta


def generate_trajectory(route, init_pose = (0, 0,np.pi/2),v_straight=v,v_turn=v,dt=dt):
    curr_pose = init_pose
    func = {'straight': straight, 'turn': turn, 'inplace': inplace, "reverse":reverse, "change-lane": change_lane}
    x, y, t = np.array([]), np.array([]),np.array([])
    for manoeuvre, command in route:
        px, py, pt = func[manoeuvre](command, curr_pose,v=v_straight if manoeuvre == "straight" or manoeuvre == "reverse" else v_turn,dt=dt)
        curr_pose = px[-1],py[-1],pt[-1]
        x = np.concatenate([x, px])
        y = np.concatenate([y, py])
        t = np.concatenate([t, pt])
        
    return x, y, t


def get_corners(x,y):
    corners = [ [x[0], y[0]] ]
    for x1, x2, x3, y1, y2, y3 in zip(x[:-2],x[1:-1],x[2:],y[:-2],y[1:-1],y[2:]):
        slope = (x2-x1)*(y3-y2) - (x3-x2)*(y2-y1)
        if np.abs(slope) > 0.0:
            corners.append([x2, y2])
    corners.append( [ x[-1], y[-1] ] )
    return corners


epsilon = 1e-9


def corners_to_route(corners,r=0.1):
    pre_x, pre_y = corners[0]
    route = []
    x,y = corners[1]
    init_pose = (pre_x,pre_y,np.pi/2)
    
    if x > pre_x + epsilon:
        init_pose = (pre_x, pre_y,0)
    elif x < pre_x - epsilon:
        init_pose = (pre_x, pre_y,np.pi)
    elif y > pre_y + epsilon:
        init_pose = (pre_x, pre_y,np.pi/2)
    elif y < pre_y - epsilon:
        init_pose = (pre_x, pre_y,-np.pi/2)

    for i in range(1,len(corners)):
        R = r if i==1 or i == len(corners)-1 else 2*r
        x, y = corners[i]
        if abs(pre_x -x) < epsilon:
            route.append(("straight",abs(y-pre_y)-R))
        else:
            route.append(("straight",abs(x-pre_x)-R))
        if i < len(corners)-1:
            next_x = corners[i+1][0]
            next_y = corners[i+1][1]
            slope = (x-pre_x)*(next_y-y) - (next_x-x)*(next_y-pre_y)
            if slope > 0:
                route.append(("turn",90))
            elif slope < 0:
                route.append(("turn",-90))

        pre_y = y
        pre_x = x
    return route, init_pose

def euclidean(node1, node2):
    x1,y1 = node1
    x2,y2 = node2
    # return 0
    # return abs(x1-y1) + abs(x2-y2)
    return np.sqrt((x1-y1)**2 + (x2-y2)**2)

class Astar:
    def __init__(self, occupancy_grid):
        self.occupancy_grid = occupancy_grid

        print(self.occupancy_grid)

    def shortest_path(self,start, goal, norm=euclidean):
        self.predecessors = []
        self.predecessors.append(start)
        heap = []

        self.finalized = np.zeros_like(self.occupancy_grid)

        print("self.finalized", self.finalized.shape)
        self.predecessors = -np.ones((self.occupancy_grid.shape[0],self.occupancy_grid.shape[1],2))
        print("pred",self.predecessors.shape)
        self.distance = np.inf * np.ones_like(self.occupancy_grid)

        heapq.heappush(heap,(0+norm(start, goal),start))
        self.distance[start[1],start[0]] = 0

        print(self.occupancy_grid.shape)
        print(start)
        while len(heap):
            node = heapq.heappop(heap)[1]
            if self.finalized[node[1],node[0]] == 1:
                continue
            # print(heap)

            if node[0] == goal[0] and node[1] == goal[1]:
                print("Goal Reached!")
                break

            self.finalized[node[1],node[0]] = 1

            if node[1] < self.occupancy_grid.shape[0]-1:
                if not self.finalized[node[1]+1, node[0]] and not self.occupancy_grid[node[1]+1,node[0]]:
                    if self.distance[node[1]+1, node[0]] >= 1+self.distance[ node[1], node[0]]:
                        self.predecessors[node[1]+1, node[0]] = node
                        print(node[1])
                        heapq.heappush(heap,(self.distance[node[1],node[0]]+1 + euclidean((node[0],node[1]+1),goal),(node[0],node[1]+1)))
                    self.distance[ node[1]+1, node[0]] = min(self.distance[node[1]+1, node[0]], 1+self.distance[ node[1], node[0]])

            if node[1] > 0:
                if not self.finalized[node[1]-1, node[0]] and not self.occupancy_grid[node[1]-1,node[0]]:
                    if self.distance[node[1]-1, node[0]] >= 1+self.distance[node[1], node[0]]:
                        self.predecessors[node[1]-1, node[0]] = node
                        heapq.heappush(heap, (self.distance[node[1],node[0]]+1 + euclidean((node[0], node[1]-1), goal), (node[0], node[1]-1)))
                    self.distance[node[1]-1, node[0]] = min(self.distance[ node[1]-1, node[0]], 1+self.distance[ node[1], node[0]])

            if node[0] < self.occupancy_grid.shape[1]-1:
                if not self.finalized[node[1], node[0]+1] and not self.occupancy_grid[node[1],node[0]+1]:
                    if self.distance[ node[1], node[0]+1] >= 1+self.distance[ node[1], node[0]]:
                        self.predecessors[node[1], node[0]+1] = node
                        heapq.heappush(heap,(self.distance[node[1],node[0]]+1 + euclidean((node[0]+1,node[1]),goal),(node[0]+1,node[1])))
                    self.distance[ node[1], node[0]+1] = min(self.distance[ node[1], node[0]+1], 1+self.distance[ node[1], node[0]])

            if node[0] > 0:
                if not self.finalized[node[1],node[0]-1] and not self.occupancy_grid[node[1],node[0]-1]:
                    if self.distance[ node[1], node[0]-1] >= 1+self.distance[ node[1], node[0]]:
                        self.predecessors[node[1], node[0]-1] = node
                        heapq.heappush(heap,(self.distance[node[1],node[0]]+1 + euclidean((node[0]-1,node[1]),goal),(node[0]-1,node[1])))
                    self.distance[ node[1], node[0]-1] = min(self.distance[ node[1], node[0]-1], 1+self.distance[ node[1], node[0]])

            # print(self.distance)

        # self.distance[self.distance > 1000 ] = 1000

        print(self.predecessors[goal[1],goal[0]])

        path = []

        node = goal
        while node[0] != -1:
            print(node)
            path.append(node)
            node = self.predecessors[int(node[1]),int(node[0])]

        path.reverse()
        print("path",path)
        path = np.array(path)
        x,y = path[:,0],path[:,1]
        if DEBUG:
            import matplotlib.pyplot as plt
            plt.imshow(self.distance, origin="lower")
            plt.figure()
            plt.imshow(self.occupancy_grid, origin="lower")
            plt.scatter(x,y)

            corners = get_corners(x.tolist(),y.tolist())
            route,init_pose = corners_to_route(corners)
            print(route)
            # plt.show()
            print(self.distance)

            x, y, _ = generate_trajectory(route,init_pose,v_turn=0.1)
            # plt.figure()
            plt.plot(x, y)
            plt.axis("equal")
            # plt.grid()
            plt.show()
        return path[:,0],path[:,1]

# Fix a turn radius r
# Shorten the straight segments by r
# convert this into {("straight", s1), ("turn", +/- 90), ("straight", s2)}

route = []
pre_x, pre_y, theta = 0, 0, 90
# corners.append([5,5])


            
# route,init_pose = corners_to_route(corners,r=0.01)
    

# use generate_trajectory() and plot the smooth path
# print(route)
# x, y, _ = generate_trajectory(route,init_pose,v_turn=0.1)
# plt.figure()
# plt.plot(x, y)
# plt.axis("equal")
# plt.grid()
        
# import numpy as np
# import networkx as nx
# import matplotlib.pyplot as plt
# from IPython.display import Image
# from networkx.algorithms.shortest_paths.generic import *

# def euclidean(node1, node2):
#     x1,y1 = node1
#     x2,y2 = node2
#     return np.sqrt((x1-y1)**2 + (x2-y2)**2)