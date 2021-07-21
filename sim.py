import tkinter as tk
from tkinter import PhotoImage
from tkinter import ttk
import toml
import matplotlib.pyplot as plt
import numpy as np
import imageio

config = toml.load("config.toml")
maps = config["maps"]

print(config)

root = tk.Tk()

assets = {}

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


def simulate():
    map_name = maps[combo_map.get()]
    map_img = imageio.imread(map_name)
    map_img = np.array(map_img)[:, :, 0]

    plt.figure()
    plt.imshow(map_img, cmap="Accent")
    plt.title(map_name)
    plt.show()


button_sim = ttk.Button(root, text="Simulate", command=simulate)
button_sim.pack()
root.mainloop()
