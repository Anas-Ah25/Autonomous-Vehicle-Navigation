import tkinter as tk
from tkinter import messagebox, filedialog
from tkinter import ttk
import os

# Import the algorithm class (ensure it is accessible)
from Phase_1_CSAI import *
class PathfindingGUI(tk.Tk):
    def __init__(self, grid):
        super().__init__()

        self.grid = grid
        self.title("Pathfinding Algorithm GUI")
        self.geometry("600x400")

        # Disable the icon issue
        self.iconbitmap(default="")  # Disable icon setting (optional)

        # Add GUI elements
        self.create_widgets()

    def create_widgets(self):
        # Dropdown for algorithm selection
        self.algo_label = tk.Label(self, text="Select Algorithm:")
        self.algo_label.pack(pady=10)

        self.algorithm_options = ['BFS', 'DFS', 'A*', 'IDS', 'Simulated Annealing']
        self.selected_algo = tk.StringVar(self)
        self.selected_algo.set(self.algorithm_options[0])  # default value
        self.algorithm_menu = ttk.Combobox(self, textvariable=self.selected_algo, values=self.algorithm_options)
        self.algorithm_menu.pack(pady=10)

        # Button to run the selected algorithm
        self.run_button = tk.Button(self, text="Run Algorithm", command=self.run_algorithm)
        self.run_button.pack(pady=20)

        # Button to generate exploration video
        self.explore_video_button = tk.Button(self, text="Generate Exploration Video", command=self.generate_explore_video, state=tk.DISABLED)
        self.explore_video_button.pack(pady=10)

        # Button to generate best path video
        self.best_path_video_button = tk.Button(self, text="Generate Best Path Video", command=self.generate_best_path_video, state=tk.DISABLED)
        self.best_path_video_button.pack(pady=10)

        # Button to view the optimal path image
        self.show_path_button = tk.Button(self, text="Show Optimal Path Image", command=self.show_optimal_path_image, state=tk.DISABLED)
        self.show_path_button.pack(pady=10)

        # Status label
        self.status_label = tk.Label(self, text="Status: Ready", font=("Arial", 12))
        self.status_label.pack(pady=20)

    def run_algorithm(self):
        # Get the selected algorithm
        algorithm_name = self.selected_algo.get()

        # Get start and goal points
        start = (0, 0)
        goal = (24, 24)

        # Run the algorithm and get the results
        if algorithm_name == "BFS":
            path, alg_name, all_moves = self.grid.bfs()
        elif algorithm_name == 'DFS':
            path, alg_name, all_moves = self.grid.dfs()
        elif algorithm_name == 'A*':
            path, alg_name, all_moves = self.grid.astar()
        elif algorithm_name == 'IDS':
            path, alg_name, final_moves = self.grid.iterative_deepening_search()
            self.grid.images_frames(alg_name, path, final_moves)
            self.grid.videoFrom_images(alg_name)
            self.grid.visualize_path(path, alg_name)
            self.status_label.config(text="Status: IDS Algorithm Completed!")
            self.enable_video_buttons()
            return  # IDS doesn't need more steps here
        elif algorithm_name == 'Simulated Annealing':
            path, alg_name, all_moves = self.grid.simulated_annealing()

        # Generate frames for videos
        self.grid.images_frames(alg_name, path, all_moves)

        # Generate the path formation video
        self.grid.videoFrom_images(alg_name)
        self.grid.visualize_path(path, alg_name)

        # Update status and enable buttons for videos and image
        self.status_label.config(text="Status: Algorithm Completed!")
        self.enable_video_buttons()

    def enable_video_buttons(self):
        """Enable video generation buttons after algorithm run"""
        self.explore_video_button.config(state=tk.NORMAL)
        self.best_path_video_button.config(state=tk.NORMAL)
        self.show_path_button.config(state=tk.NORMAL)

    def generate_explore_video(self):
        """Generate the exploration video"""
        algorithm_name = self.selected_algo.get()
        self.grid.videoFrom_movements(algorithm_name)
        messagebox.showinfo("Video Created", f"Exploration video for {algorithm_name} created successfully!")

    def generate_best_path_video(self):
        """Generate the best path video"""
        algorithm_name = self.selected_algo.get()
        self.grid.videoFrom_images(algorithm_name)
        messagebox.showinfo("Video Created", f"Best path video for {algorithm_name} created successfully!")

    def show_optimal_path_image(self):
        """Show the optimal path image"""
        algorithm_name = self.selected_algo.get()
        path_image = f"images_of_{algorithm_name}_path/final_path.png"
        if os.path.exists(path_image):
            img = tk.PhotoImage(file=path_image)
            img_label = tk.Label(self, image=img)
            img_label.image = img  # Keep a reference!
            img_label.pack(pady=10)
        else:
            messagebox.showerror("Error", "Path image not found! Please run the algorithm first.")

if __name__ == "__main__":
    width = 25
    height = 25
    start = (0, 0)
    goal = (24, 24)

    # Create the grid object (this is where the algorithm class is used)
    grid = algorithm(width, height, start, goal)

    # Create and run the GUI
    app = PathfindingGUI(grid)
    app.mainloop()
