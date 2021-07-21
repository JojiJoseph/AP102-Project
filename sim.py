from dwa import DWA
import tkinter as tk
from tkinter import PhotoImage
from tkinter import ttk
import toml
import matplotlib.pyplot as plt
import numpy as np
import imageio
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from astar import Astar, generate_trajectory

from astar import corners_to_route, get_corners

from astar import generate_trajectory as traj

config = toml.load("config.toml")
maps = config["maps"]


root = tk.Tk()


icon_file_name = "icon.png"
icon = PhotoImage(file=icon_file_name)

# FIXME: Why icon is not showing up on titlebar?
root.iconphoto(False, icon)
root.geometry("1280x960")
root.title(" AP102 Project")

combo_map = ttk.Combobox(master=root)

combo_map["values"] = list(maps.keys())
combo_map.set(combo_map["values"][0])
combo_map.state(['readonly'])
combo_map.pack()

fig = plt.figure()
ax = plt.gca()
canvas = FigureCanvasTkAgg(fig, master=root)

canvas.get_tk_widget().pack()

SIMULATION_IS_STOPPED = 0
SIMULATION_IN_PROGRESS = 1

simulation_status = SIMULATION_IS_STOPPED


def simulate():
    global simulation_status
    if simulation_status == SIMULATION_IN_PROGRESS:
        print("Simulation is already running!")
        return
    map_name = maps[combo_map.get()]
    map_img = imageio.imread(map_name)
    map_img = np.array(map_img)[:, :, 0]/255.

    astar_ = Astar(map_img)

    x, y = astar_.shortest_path((0, 0), (19, 19))

    # plt.sca(ax)
    corners = get_corners(x, y)
    route, init_pose = corners_to_route(corners)
    x_, y_, t = generate_trajectory(route, init_pose, v_turn=0.01, dt=0.1)

    ref_path = ref_path = np.hstack(
        [np.array(axis).reshape(-1, 1) for axis in (x_, y_, t)])

    simulation_status = SIMULATION_IN_PROGRESS

    dwa_obj = DWA(map_img.T, ref_path, init_pose, grid_res=1)

    root.after(100, update_plot, dwa_obj, ax)


def stop_simulation():
    global simulation_status
    if simulation_status == SIMULATION_IN_PROGRESS:
        simulation_status = SIMULATION_IS_STOPPED
        print("Simulation is being stopped!")


def update_plot(dwa_obj, ax):
    if simulation_status != SIMULATION_IN_PROGRESS:
        dwa_obj.reset()
        return
    try:
        progress = next(dwa_obj)
        ax.imshow(dwa_obj.grid_data.T, cmap="Accent")
        ax.plot(progress[:, 0], progress[:, 1])
        ax.scatter(progress[-1, 0], progress[-1, 1], label="robot")
        plt.legend()
        canvas.draw()
        ax.clear()
        root.after(1, update_plot, dwa_obj, ax)
    except StopIteration:
        pass


button_sim = ttk.Button(root, text="Simulate", command=simulate)
button_stop_sim = ttk.Button(root, text="Stop", command=stop_simulation)
button_sim.pack()
button_stop_sim.pack()
root.mainloop()
