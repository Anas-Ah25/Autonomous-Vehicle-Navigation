import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
import time
import tkinter as tk
from PIL import Image, ImageTk
import threading
import cv2

# ---------------- Node and Grid Classes ----------------
class Node:
    def __init__(self, x, y, goal, g=0):
        self.x = x
        self.y = y
        self.h = self.heuristic(goal)  # heuristic
        self.g = g  # cost
        self.f = self.g + self.h  # total cost (A*)

    def heuristic(self, goal):
        return abs(self.x - goal[0]) + abs(self.y - goal[1])


class Grid:
    def __init__(self, width, height, start, goal):
        self.width = width
        self.height = height
        self.grid = np.zeros((width, height))
        self.Moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        self.start = start
        self.goal = goal
        self.pattern()

    def pattern(self):  # This is grid pattern

        # Outer boundaries of the grid
        self.grid[0, :] = 1
        self.grid[:, 0] = 1
        self.grid[-1, :] = 1
        self.grid[:, -1] = 1

        # Horizontal and vertical paths resembling city roads
        for i in range(3, self.width - 3, 3):
            self.grid[i, 1:self.width - 1] = 1

        for j in range(3, self.height - 3, 3):
            self.grid[1:self.height - 1, j] = 1

        # Add random obstacles within the grid
        np.random.seed(0)
        obstacles = np.random.randint(1, self.width - 1, size=(int(self.width * self.height * 0.15), 2))
        for (x, y) in obstacles:
            if (x, y) != self.start and (x, y) != self.goal:
                self.grid[x, y] = 0  # Block these cells

        # Central area with complex paths
        self.grid[10:15, 5:20] = 1
        self.grid[5:10, 10:15] = 1
        self.grid[15:20, 10:15] = 1
        self.grid[10:15, 10:15] = 0

        # Additional paths to increase the grid complexity
        self.grid[5, 5:10] = 1
        self.grid[10:15, 3] = 1
        self.grid[15, 15:20] = 1
        self.grid[20, 5:10] = 1
        self.grid[20, 15:25] = 1
        self.grid[5:7, 24] = 0

    def get_node(self, x, y):
        return Node(x, y, self.goal)
    
# ----------------------------------------------------------------------------

# +++++++++++++++++++++++++++++ Algorithm Class ++++++++++++++++++++++++++++++
class algorithm(Grid):
    def __init__(self, width, height, start, goal):
        super().__init__(width, height, start, goal)
        self.path = []
        self.images_List = []  # Added to store image paths
# ****************************************************************************************************
# ***************************************** Visulizations ********************************************* 
    # Visualization methods
    def visualize_path(self, path, AlGname):
        #  ---------- The final path image -------------
        plt.imshow(self.grid, cmap="gray_r")
        plt.scatter(self.start[1], self.start[0], color="green")
        plt.scatter(self.goal[1], self.goal[0], color="red")
        path = np.array(path)
        for i in range(1, path.shape[0] - 1):
            plt.scatter(path[i, 1], path[i, 0], color="blue")
        plt.title(f"Final Path - {AlGname}")
        plt.xticks([])
        plt.yticks([])
        plt.savefig(f"Final Path - {AlGname}.png")
        plt.close()

        # ---------- The explored nodes image -------------
        plt.imshow(self.grid, cmap="gray_r")
        plt.scatter(self.start[1], self.start[0], color="green")
        plt.scatter(self.goal[1], self.goal[0], color="red")
        all_moves = np.array(self.all_moves)
        plt.scatter(all_moves[:, 1], all_moves[:, 0], color="lightblue", alpha=0.5)
        plt.title(f"Explored Nodes - {AlGname}")
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(f"{AlGname}_final_result.png")
        plt.close()

    def images_frames(self, algoritmName, path, all_moves):
        images_List = []
        WholeMoves = []

        # ----------------------------------- Frames of the final path -----------------------------
        current_path = []
        for i in range(len(path)):
            current_path.append(path[i])
            plt.figure(figsize=(6, 6))
            plt.imshow(self.grid, cmap="gray_r")
            plt.scatter(self.start[1], self.start[0], color="green")
            plt.scatter(self.goal[1], self.goal[0], color="red")
            # plot current path
            if len(current_path) > 1:
                current = np.array(current_path)
                plt.scatter(current[1:, 1], current[1:, 0], color="blue", label="Path")
            plt.title(f"Path Formation {algoritmName} - Step {i}")
            plt.xticks([])
            plt.yticks([])
            # each image as frame, saved in directory
            os.makedirs(f"images_of_{algoritmName}_path", exist_ok=True)
            image_path = f"images_of_{algoritmName}_path/path_{i}.png"
            plt.savefig(image_path)
            images_List.append(image_path)
            plt.close()

        # ===================================================================================
        # ------------------------------ whole algorithm moves trace --------------------------
        '''The main logic here is to take the (all_moves) which is the nodes that the algoritm covered
            then plot each move, this plot is like a frame, we make the video from playing all these frames
            after each other, so i created a directory for each type of frames saved during the code running
            this directory has the frames, where we will take the images inside it and pass it to 'videoFrom_movements' 
            and 'videoFrom_images', this will happen after (return images_List, WholeMoves) of the images_frames function we are in now
            those are the lists with  paths for the frames we talked about
        '''
        # Whole algorithm moves trace
        explored_nodes = []
        for i, move in enumerate(all_moves):
            explored_nodes.append(move)
            plt.figure(figsize=(6, 6))
            plt.imshow(self.grid, cmap="gray_r")
            plt.scatter(self.start[1], self.start[0], color="green")
            plt.scatter(self.goal[1], self.goal[0], color="red")
            explored = np.array(explored_nodes)
            plt.scatter(explored[:, 1], explored[:, 0], color="grey", alpha=0.5)
            plt.title(f"Exploration of {algoritmName} - Step {i}")
            plt.xticks([])
            plt.yticks([])
            os.makedirs(f"images_of_{algoritmName}_exploration", exist_ok=True)
            AllMoves_img_path = f"images_of_{algoritmName}_exploration/explore_{i}.png"
            plt.savefig(AllMoves_img_path)
            WholeMoves.append(AllMoves_img_path)
            plt.close()

        self.images_List = images_List
        self.wholeMoves = WholeMoves
        self.all_moves = all_moves

        return images_List, WholeMoves
    # ===================================================================================================
    # -------------------------------------- Video making -----------------------------------------------
    # ===================================================================================================
    def videoFrom_images(self, algoName, fps=5):
        output_video_path = f"{algoName}_path_formation.mp4"
        with imageio.get_writer(output_video_path, fps=fps) as video_writer:
            for image_path in self.images_List:
                video_writer.append_data(imageio.imread(image_path))
        print("Path formation video created successfully!")

    def videoFrom_movements(self, algoName, fps=5):
        output_video_path = f"{algoName}_exploration.mp4"
        with imageio.get_writer(output_video_path, fps=fps) as video_writer:
            for image_path in self.wholeMoves:
                video_writer.append_data(imageio.imread(image_path))
        print("Exploration process video created successfully!")
    # ****************************************************************************************************
    # ++++++++++++++++++++++++++++++++++++++++++++++++++ Algorithms itself ++++++++++++++++++++++++++++++

    def bfs(self):
        queue = [self.start]
        visited = set()
        parent = {}
        parent[self.start] = None
        all_moves = []
        memory_max = 1 # memory size counter
        while queue:
            memory_max = max(memory_max,len(queue)) # mmory size
            current = queue.pop(0)
            all_moves.append(current)
            if current == self.goal:
                break
            visited.add(current)
            for i, j in self.Moves:
                new_x, new_y = current[0] + i, current[1] + j
                if (0 <= new_x < self.width and 0 <= new_y < self.height and
                        self.grid[new_x, new_y] == 1 and (new_x, new_y) not in visited):
                    queue.append((new_x, new_y))
                    visited.add((new_x, new_y))
                    parent[(new_x, new_y)] = current

        path = []
        current = self.goal
        while current:
            path.append(current)
            current = parent[current]
        return path[::-1], 'BFS', all_moves,memory_max
    # --------------- DFS ---------------
    def dfs(self):
        stack = [self.start]
        visited = set()
        parent = {}
        parent[self.start] = None
        all_moves = []
        memory_max = 1 # memory size, maximum number of elements in the stack
        while stack:
            memory_max = max(memory_max, len(stack))
            current = stack.pop()
            all_moves.append(current)
            if current == self.goal:
                break
            visited.add(current)
            for i, j in self.Moves:
                new_x, new_y = current[0] + i, current[1] + j
                if (0 <= new_x < self.width and 0 <= new_y < self.height and
                        self.grid[new_x, new_y] == 1 and (new_x, new_y) not in visited):
                    stack.append((new_x, new_y))
                    visited.add((new_x, new_y))
                    parent[(new_x, new_y)] = current
        path = []
        current = self.goal
        while current:
            path.append(current)
            current = parent[current]

        return path[::-1], 'DFS', all_moves, memory_max


# ####################################################################
# -------------------------------- GUI -------------------------------
# ####################################################################


class AlgorithmSelectionGUI:
    def __init__(self, root, algorithms):
        self.root = root
        self.algorithms = algorithms
        self.selected_algorithm = None

        self.root.title("Select Algorithm")

        self.selection_window = tk.Frame(self.root)
        self.selection_window.pack(fill="both", expand=True)

        self.title_label = tk.Label(self.selection_window, text="Select Algorithm", font=("Arial", 24))
        self.title_label.pack(pady=20)

        self.algorithm_var = tk.StringVar(value=self.algorithms[0])

        for algo in self.algorithms:
            rb = tk.Radiobutton(self.selection_window, text=algo, variable=self.algorithm_var, value=algo)
            rb.pack(anchor='w', padx=20)

        self.start_button = tk.Button(self.selection_window, text="Start Algorithm", command=self.start_algorithm)
        self.start_button.pack(pady=20)

    def start_algorithm(self):
        self.selected_algorithm = self.algorithm_var.get()
        self.root.quit()

class LoadingScreenGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Loading")
        self.loading_label = tk.Label(self.root, text="Loading, please wait...", font=("Arial", 18))
        self.loading_label.pack(pady=20)

class ResultGUI:
    def __init__(self, root, outputs, algorithm_name, run_time, memory_max, analytics_data):
        self.root = root
        self.root.title("Algorithm Visualization")
        self.outputs = outputs
        self.algorithm_name = algorithm_name
        self.run_time = run_time
        self.memory_max = memory_max
        self.analytics_data = analytics_data
        self.try_again = False  # Flag to indicate if user wants to try another algorithm

        # Create pages
        self.result_page = tk.Frame(self.root)
        self.analytics_page = tk.Frame(self.root)

        # Set up pages
        self.setup_result_page()
        self.setup_analytics_page()

        self.show_result_page()

    def setup_result_page(self):
        self.video_label = tk.Label(self.result_page)
        self.video_label.pack()

        self.control_frame = tk.Frame(self.result_page)
        self.control_frame.pack()

        self.play_exploration_button = tk.Button(self.control_frame, text="Play Exploration Video", command=self.play_exploration_video)
        self.play_exploration_button.pack(side=tk.LEFT, padx=5)

        self.play_path_button = tk.Button(self.control_frame, text="Play Path Video", command=self.play_path_video)
        self.play_path_button.pack(side=tk.LEFT, padx=5)

        self.view_final_path_button = tk.Button(self.control_frame, text="View Final Path Image", command=self.view_final_path_image)
        self.view_final_path_button.pack(side=tk.LEFT, padx=5)

        self.view_explored_nodes_button = tk.Button(self.control_frame, text="View Explored Nodes Image", command=self.view_explored_nodes_image)
        self.view_explored_nodes_button.pack(side=tk.LEFT, padx=5)

        self.try_another_button = tk.Button(self.control_frame, text="Try Another Algorithm", command=self.try_another_algorithm)
        self.try_another_button.pack(side=tk.LEFT, padx=5)

        self.analytics_button = tk.Button(self.control_frame, text="Analytics", command=self.show_analytics_page)
        self.analytics_button.pack(side=tk.LEFT, padx=5)

    def setup_analytics_page(self):
        self.analytics_label = tk.Label(self.analytics_page, text="Analytics", font=("Arial", 18))
        self.analytics_label.pack(pady=20)
        self.analytics_text = tk.Text(self.analytics_page, width=60, height=20)
        self.analytics_text.pack()
        self.back_button = tk.Button(self.analytics_page, text="Back", command=self.show_result_page)
        self.back_button.pack(pady=10)

    def show_result_page(self):
        self.analytics_page.pack_forget()
        self.result_page.pack(fill="both", expand=True)

        self.exploration_video_path = self.outputs['exploration_video']
        self.path_video_path = self.outputs['path_video']
        self.final_path_image_path = self.outputs['final_image']
        self.explored_nodes_image_path = self.outputs['explored_nodes_image']

        self.play_exploration_video()

    def play_exploration_video(self):
        self.play_video(self.exploration_video_path)

    def play_path_video(self):
        self.play_video(self.path_video_path)

    def play_video(self, video_path):
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(video_path)
        self.update_video_frame()

    def update_video_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = frame.resize((600, 600), Image.LANCZOS)
            self.photo = ImageTk.PhotoImage(frame)
            self.video_label.config(image=self.photo)
            self.video_label.image = self.photo  # Keep a reference
            self.root.after(33, self.update_video_frame)
        else:
            self.cap.release()
            self.cap = None

    def view_final_path_image(self):
        image = Image.open(self.final_path_image_path)
        image = image.resize((600, 600), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(image)
        self.video_label.config(image=self.photo)
        self.video_label.image = self.photo  # Keep a reference

    def view_explored_nodes_image(self):
        image = Image.open(self.explored_nodes_image_path)
        image = image.resize((600, 600), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(image)
        self.video_label.config(image=self.photo)
        self.video_label.image = self.photo  # Keep a reference

    def show_analytics_page(self):
        self.result_page.pack_forget()
        self.analytics_page.pack(fill="both", expand=True)
        # Update analytics text
        self.analytics_text.delete('1.0', tk.END)
        self.analytics_text.insert(tk.END, "Algorithms Tried and Execution Times:\n")
        for algo, data in self.analytics_data.items():
            self.analytics_text.insert(tk.END, f"{algo}:\n")
            self.analytics_text.insert(tk.END, f"    Execution Time: {data['run_time']:.2f} seconds\n")
            self.analytics_text.insert(tk.END, f"    Max Data Structure Size: {data['memory_max']} nodes\n")

    def try_another_algorithm(self):
        # Set flag and close the GUI
        self.try_again = True
        self.root.quit()
# ########################################################################

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++  Run algorithm  ++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def execute_algorithm(algorithmName, width=25, height=25, start=(0, 0), goal=(24, 24)):
    print(f"Starting {algorithmName} algorithm...")
    start_time = time.time()
    grid = algorithm(width, height, start, goal)
    if algorithmName == "BFS":
        path, AlGname, all_movements, memory_max = grid.bfs()
    elif algorithmName == 'DFS':
        path, AlGname, all_movements, memory_max = grid.dfs()
    else:
        raise ValueError("Unknown algorithm name")
    grid.images_frames(AlGname, path, all_movements)
    grid.videoFrom_images(AlGname)
    grid.videoFrom_movements(AlGname)
    grid.visualize_path(path, AlGname)
    end_time = time.time()
    run_time = end_time - start_time
    print(f"{algorithmName} algorithm completed in {run_time:.2f} seconds.")
    return {
        'exploration_video': f"{AlGname}_exploration.mp4",
        'path_video': f"{AlGname}_path_formation.mp4",
        'final_image': f"Final Path - {AlGname}.png",
        'explored_nodes_image': f"{AlGname}_final_result.png"
    }, run_time, memory_max



# ----------------------- Main run -----------------------

if __name__ == "__main__":
    algorithms = ["BFS", "DFS"]
    analytics_data = {}  # Dictionary to store analytics data

    while True:
        # Algorithm selection GUI
        root_selection = tk.Tk()
        algorithm_selection_gui = AlgorithmSelectionGUI(root_selection, algorithms)
        root_selection.mainloop()
        selected_algorithm = algorithm_selection_gui.selected_algorithm
        root_selection.destroy()

        if selected_algorithm is None:
            break  # Exit if no algorithm was selected

        # Loading screen
        root_loading = tk.Tk()
        loading_gui = LoadingScreenGUI(root_loading)
        root_loading.update()

        # Execute the algorithm
        outputs, run_time, memory_max = execute_algorithm(selected_algorithm)
        root_loading.destroy()  # Close the loading screen

        analytics_data[selected_algorithm] = {
            'run_time': run_time,
            'memory_max': memory_max
        }

        # Result GUI
        root_result = tk.Tk()
        result_gui = ResultGUI(root_result, outputs, selected_algorithm, run_time, memory_max, analytics_data)
        root_result.mainloop()
        root_result.destroy()

        if not result_gui.try_again:
            break  # Exit if the user does not want to try another algorithm
