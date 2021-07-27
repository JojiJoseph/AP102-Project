import tkinter as tk
from tkinter import PhotoImage
from tkinter import ttk
import toml
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import imageio
from PIL import Image, ImageTk
import cv2
import numpy as np


from astar import Astar, generate_trajectory, corners_to_route, get_corners
from dwa import DWA

config = toml.load("config.toml")
maps = config["maps"]


root = tk.Tk()

SIMULATION_IS_STOPPED = 0
SIMULATION_IN_PROGRESS = 1
SIMULATION_IS_PAUSED = 2  # TODO

simulation_status = SIMULATION_IS_STOPPED

icon_file_name = "icon.png"
icon = PhotoImage(file=icon_file_name)

# FIXME: Why icon is not showing up on titlebar?
root.iconphoto(False, icon)
root.geometry("1280x960")
root.title(" AP102 Project")

combo_frame = ttk.Frame(root)
combo_label = ttk.Label(master=combo_frame, text=" Select a map ")
combo_label.pack(side=tk.LEFT)

combo_map = ttk.Combobox(master=combo_frame)

combo_map["values"] = list(maps.keys())
combo_map.set(combo_map["values"][0])
combo_map.state(['readonly'])
combo_map.pack(side=tk.LEFT)


def load_map():
    canvas.delete('all')
    if simulation_status == SIMULATION_IN_PROGRESS or simulation_status == SIMULATION_IS_PAUSED:
        tk.messagebox.showwarning(
            "Warning", "Please stop the simulation to load a map.")
    map_name = maps[combo_map.get()]
    map_img = imageio.imread(map_name)
    map_img = np.array(map_img)[:, :, 0]/255.
    global img
    img = cv2.resize(1-map_img, (400, 400),
                     interpolation=cv2.INTER_NEAREST)*255.
    img = ImageTk.PhotoImage(image=Image.fromarray(img))
    canvas.create_image(20, 20, image=img, anchor="nw")
    canvas.update()


button_load = ttk.Button(master=combo_frame, text="Load", command=load_map)
button_load.pack(side=tk.RIGHT)

combo_frame.pack(pady=10)

fig = plt.figure()
ax = plt.gca()
dwa_obj = None

canvas = tk.Canvas(master=root, height=400, width=400, bg="white")


def update_obstacle(event):
    global dwa_obj
    x = event.x // 50
    y = event.y // 50
    if dwa_obj is not None:
        dwa_obj.grid_data[x:x+2, y:y+2] = 1


canvas.bind("<Button-1>", update_obstacle)


canvas.pack()


scale_x, scale_y = 20, 20


def simulate():
    global simulation_status
    if simulation_status == SIMULATION_IN_PROGRESS:
        print("Simulation is already running!")
        return
    map_name = maps[combo_map.get()]
    map_img = imageio.imread(map_name)
    map_img = np.array(map_img)[:, :, 0]/255.

    astar_ = Astar(map_img)

    global x_star, y_star

    x, y = astar_.shortest_path((0, 0), (19, 19))
    plot(x, y, "astar")

    corners = get_corners(x, y)
    route, init_pose = corners_to_route(corners)
    x_, y_, t = generate_trajectory(route, init_pose, v_turn=0.1, dt=0.1)
    x_star = x_
    y_star = y_

    ref_path = ref_path = np.hstack(
        [np.array(axis).reshape(-1, 1) for axis in (x_, y_, t)])

    simulation_status = SIMULATION_IN_PROGRESS
    global dwa_obj
    global img  # To not lose img
    dwa_obj = DWA(map_img.T, ref_path, init_pose, grid_res=1)
    img = cv2.resize(1-dwa_obj.grid_data.T, (400, 400),
                     interpolation=cv2.INTER_NEAREST)*255.
    img = ImageTk.PhotoImage(image=Image.fromarray(img))
    canvas.create_image(20, 20, image=img, anchor="nw")
    root.after(100, update_plot, dwa_obj, ax)


def scatter(x, y, tag="scatter", color="red"):
    print(x.shape)
    global canvas
    n = len(x)
    assert len(x) == len(y)
    for i in range(n):
        canvas.create_oval(20-10+x[i]*scale_x, 20-10+y[i]*scale_y,
                           20+10+x[i]*scale_x, 20+10+y[i]*scale_y, fill=color, tag=tag)
    canvas.update()


def plot(x, y, tag="lines", color="black"):
    print(x.shape)
    global canvas
    n = len(x)
    assert len(x) == len(y)
    for i in range(1, n):
        canvas.create_line(20+x[i-1]*scale_x + scale_x/2, 20+y[i-1]*scale_y + scale_y/2,
                           20+x[i]*scale_x + scale_x/2, 20+y[i]*scale_y + scale_y/2, fill=color, width=2, tag=tag)
    canvas.update()


def stop_simulation():
    global simulation_status
    if simulation_status == SIMULATION_IN_PROGRESS:
        simulation_status = SIMULATION_IS_STOPPED
        print("Simulation is being stopped!")


def pause_simulation():
    global simulation_status
    simulation_status = SIMULATION_IS_PAUSED


def update_plot(dwa_obj, ax):
    # canvas.update()
    global simulation_status
    if simulation_status != SIMULATION_IN_PROGRESS:
        dwa_obj.reset()
        return
    try:
        progress, distances = next(dwa_obj)
        ax.imshow(dwa_obj.grid_data.T, cmap="Accent")
        # ax.plot(progress[:, 0], progress[:, 1])
        canvas.delete("lines")
        canvas.delete("robot")
        canvas.delete("lidar_beam")
        plot(progress[:, 0], progress[:, 1])
        scatter(progress[None, -1, 0], progress[None, -1, 1], tag="robot")
        plot(x_star, y_star, "astar")
        x, y, theta = dwa_obj.pose
        for dist, angle in zip(distances, dwa_obj.lidar.beam_angles):
            t = angle + theta
            plot(np.array([x, x+dist*np.cos(t)]), np.array([y,
                 y+dist*np.sin(t)]), color="green", tag="lidar_beam")
        root.after(1, update_plot, dwa_obj, ax)
    except StopIteration:
        simulation_status = SIMULATION_IS_STOPPED
        print("Simulation Completed!")


button_frame = ttk.Frame(root)
button_frame.pack()
button_sim = ttk.Button(button_frame, text="Simulate", command=simulate)
# button_pause = ttk.Button(button_frame, text="Pause", command=pause_simulation) TODO implement
button_stop_sim = ttk.Button(
    button_frame, text="Stop", command=stop_simulation)
button_sim.pack(side=tk.LEFT)
# button_pause.pack()
button_stop_sim.pack(side=tk.LEFT)

button_add_obstacle = ttk.Button(button_frame, text="Add Obstacle (TODO)")
button_add_obstacle.pack(side=tk.LEFT)

button_mark_start = ttk.Button(button_frame, text="Mark Start (TODO)")
button_mark_start.pack(side=tk.LEFT)

button_mark_end = ttk.Button(button_frame, text="Mark End (TODO)")
button_mark_end.pack(side=tk.LEFT)

button_zoom_in = ttk.Button(button_frame, text="Zoom In (TODO)")
button_zoom_in.pack(side=tk.LEFT)

button_zoom_fit = ttk.Button(button_frame, text="Fit on Screen (TODO)")
button_zoom_fit.pack(side=tk.LEFT)

button_zoom_out = ttk.Button(button_frame, text="Zoom Out (TODO)")
button_zoom_out.pack(side=tk.LEFT)

root.mainloop()
