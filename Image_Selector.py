import threading
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import subprocess
import os
import depth_estimation
import fuse_depth_maps
import point_cloud_generation


def select_image(side):
    if side == "left":
        file_path = filedialog.askopenfilename(initialdir="./Middlebury",
                                               filetypes=[("Images", "*Left*.jpg *Left*.jpeg *Left*.png")])
    else:
        file_path = filedialog.askopenfilename(initialdir="./Middlebury",
                                               filetypes=[("Images", "*Right*.jpg *Right*.jpeg *Right*.png")])

    if file_path:
        if side == "left":
            left_label.config(text=os.path.basename(file_path))
            left_img = Image.open(file_path)
            left_img.thumbnail((300, 300))
            left_img = ImageTk.PhotoImage(left_img)
            left_panel.config(image=left_img)
            left_panel.image = left_img
            selected_images["left"] = file_path
        else:
            right_label.config(text=os.path.basename(file_path))
            right_img = Image.open(file_path)
            right_img.thumbnail((300, 300))
            right_img = ImageTk.PhotoImage(right_img)
            right_panel.config(image=right_img)
            right_panel.image = right_img
            selected_images["right"] = file_path
def show_loading():
    global loading_window
    loading_window = tk.Toplevel(root)
    loading_window.title("Processing...")
    loading_window.geometry("200x100")
    loading_label = tk.Label(loading_window, text="Processing...", font=("Arial", 12))
    loading_label.pack(pady=20)
    loading_window.update()

def hide_loading():
    loading_window.destroy()

def run_executable():
    if selected_images["left"] and selected_images["right"]:
        show_loading()  # Afișează fereastra de loading
        thread = threading.Thread(target=execute_processing)
        thread.start()
    else:
        print("Selectează ambele imagini înainte de a rula executabilul!")

def execute_processing():
    run_depth_processing([selected_images["left"], selected_images["right"]])


def run_depth_processing(paths):
    try:
        depth_estimation.run(paths)
        fuse_depth_maps.run()
    finally:
        root.after(1000, hide_loading)
    point_cloud_generation.run(paths)


# Interfața grafică
root = tk.Tk()
root.title("Stereo Image Selector")

selected_images = {"left": None, "right": None}

frame = tk.Frame(root)
frame.pack(pady=10)

left_button = tk.Button(frame, text="Select Left Image", command=lambda: select_image("left"))
left_button.grid(row=0, column=0, padx=10)

right_button = tk.Button(frame, text="Select Right Image", command=lambda: select_image("right"))
right_button.grid(row=0, column=1, padx=10)

left_label = tk.Label(frame, text="No Left Image Selected")
left_label.grid(row=1, column=0)

right_label = tk.Label(frame, text="No Right Image Selected")
right_label.grid(row=1, column=1)

left_panel = tk.Label(frame)
left_panel.grid(row=2, column=0)

right_panel = tk.Label(frame)
right_panel.grid(row=2, column=1)

run_button = tk.Button(root, text="Run Executable", command=run_executable)
run_button.pack(pady=10)

root.mainloop()
