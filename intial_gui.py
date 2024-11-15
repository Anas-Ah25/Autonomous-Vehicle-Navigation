import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import subprocess
from Phase_1_CSAI import *
from PIL import Image, ImageTk  # Pillow is used to display images in Tkinter

class PathfindingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pathfinding Algorithm GUI")
        self.root.geometry("600x600")

        # Initialize algorithm and grid variables
        self.algorithm = None
        self.grid = None

        # UI Elements
        self.create_widgets()

    def create_widgets(self):
        # Title label
        self.title_label = tk.Label(self.root, text="Choose Algorithm", font=("Arial", 16))
        self.title_label.pack(pady=10)

        # Algorithm selection dropdown
        self.algorithm_var = tk.StringVar()
        self.algorithm_dropdown = ttk.Combobox(self.root, textvariable=self.algorithm_var, state="readonly")
        self.algorithm_dropdown['values'] = ['BFS', 'DFS', 'A*', 'IDS', 'Simulated Annealing']
        self.algorithm_dropdown.pack(pady=10)

        # Start button to run the selected algorithm
        self.start_button = tk.Button(self.root, text="Start Algorithm", command=self.run_algorithm)
        self.start_button.pack(pady=10)

        # Exploration video button
        self.exploration_button = tk.Button(self.root, text="Show Exploration Video", command=self.show_exploration_video, state=tk.DISABLED)
        self.exploration_button.pack(pady=5)

        # Path video button
        self.path_button = tk.Button(self.root, text="Show Path Video", command=self.show_path_video, state=tk.DISABLED)
        self.path_button.pack(pady=5)

        # Path image display
        self.path_image_label = tk.Label(self.root, text="Optimal Path Image will appear here", font=("Arial", 10))
        self.path_image_label.pack(pady=20)

    def run_algorithm(self):
        algorithm_name = self.algorithm_var.get()
        if algorithm_name == "":
            messagebox.showerror("Error", "Please select an algorithm.")
            return

        # Set up grid and run the algorithm based on the selection
        self.grid = algorithm(25, 25, (0, 0), (24, 24))  # Modify if you want a custom grid size
        if algorithm_name == 'BFS':
            path, AlGname, all_movements = self.grid.bfs()
        elif algorithm_name == 'DFS':
            path, AlGname, all_movements = self.grid.dfs()
        elif algorithm_name == 'A*':
            path, AlGname, all_movements = self.grid.astar()
        elif algorithm_name == 'IDS':
            path, AlGname, final_moves = self.grid.iterative_deepening_search()
            self.grid.images_frames(AlGname, path, final_moves)
        elif algorithm_name == 'Simulated Annealing':
            path, AlGname, all_movements = self.grid.simulated_annealing()

        # Frames for the video
        self.grid.images_frames(AlGname, path, all_movements)

        # Enable the buttons for exploration and path videos
        self.exploration_button.config(state=tk.NORMAL)
        self.path_button.config(state=tk.NORMAL)

        # Generate path image and display
        self.display_path_image(path, AlGname)

        # Generate the videos
        self.grid.videoFrom_images(AlGname)
        self.grid.videoFrom_movements(AlGname)

    def display_path_image(self, path, algorithm_name):
        # Assuming the path image is saved with the algorithm name
        image_path = f"images_of_{algorithm_name}_path/final_path.png"

        if os.path.exists(image_path):
            # Open image using Pillow and display it in Tkinter
            img = Image.open(image_path)
            img = img.resize((400, 400), Image.Resampling.LANCZOS)  # Resize for better display
            img_tk = ImageTk.PhotoImage(img)
            self.path_image_label.config(image=img_tk, text="")
            self.path_image_label.image = img_tk
        else:
            messagebox.showerror("Error", "Path image not found.")

    def show_exploration_video(self):
        # Get the selected algorithm name and construct the correct video path
        algorithm_name = self.algorithm_var.get()
        video_path = f"{algorithm_name}_exploration.mp4"  # Use double underscore as per your file naming convention
        if os.path.exists(video_path):
            self.play_video(video_path)
        else:
            messagebox.showerror("Error", f"Exploration video for {algorithm_name} not found.")

    def show_path_video(self):
        # Get the selected algorithm name and construct the correct path video path
        algorithm_name = self.algorithm_var.get()
        video_path = f"{algorithm_name}_path_formation.mp4"  # Use double underscore as per your file naming convention
        if os.path.exists(video_path):
            self.play_video(video_path)
        else:
            messagebox.showerror("Error", f"Path video for {algorithm_name} not found.")

    def play_video(self, video_path):
        # Use a subprocess to play the video using the default system player
        try:
            if os.name == 'nt':  # For Windows
                subprocess.run(["start", video_path], shell=True)
            elif os.name == 'posix':  # For Linux/MacOS
                subprocess.run(["xdg-open", video_path])  # Linux
                # subprocess.run(["open", video_path])  # For MacOS, uncomment if needed
        except Exception as e:
            messagebox.showerror("Error", f"Failed to play video: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = PathfindingApp(root)
    root.mainloop()
