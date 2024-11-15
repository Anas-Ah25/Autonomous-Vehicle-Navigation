# testapp.py

import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
from heapq import heappop, heappush
import time

# ---------------- Node and Grid Classes ----------------
class Node:
    def __init__(self, x, y, goal, g=0):
        self.x = x
        self.y = y
        self.h = self.heuristic(goal)  # heuristic
        self.g = g  # cost
        self.f = self.g + self.h  # total cost (A*)
    
    def heuristic(self, goal):
        # Manhattan distance
        return abs(self.x - goal[0]) + abs(self.y - goal[1])
    
    def __lt__(self, other):
        return self.f < other.f

class Grid:
    def __init__(self, width, height, start, goal):
        self.width = width
        self.height = height
        self.grid = np.ones((width, height))
        self.Moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        self.start = start
        self.goal = goal
        self.pattern()

    def pattern(self):  # Complex pattern to create the grid map
        # Outer boundaries of the grid
        self.grid[0, :] = 0
        self.grid[:, 0] = 0
        self.grid[-1, :] = 0
        self.grid[:, -1] = 0

        # Horizontal and vertical paths resembling city roads
        for i in range(3, self.width - 3, 3):
            self.grid[i, 1:self.height - 1] = 1

        for j in range(3, self.height - 3, 3):
            self.grid[1:self.width - 1, j] = 1

        # Add random obstacles within the grid to create dead-ends and complexity
        np.random.seed(0)  # Seed for reproducibility
        num_obstacles = int(self.width * self.height * 0.15)
        obstacles = np.random.randint(1, self.width - 1, size=(num_obstacles, 2))
        for (x, y) in obstacles:
            if (x, y) != self.start and (x, y) != self.goal:
                self.grid[x, y] = 0  # Block these cells

        # Central area with complex paths
        self.grid[10:15, 5:20] = 1
        self.grid[5:10, 10:15] = 1
        self.grid[15:20, 10:15] = 1
        self.grid[10:15, 10:15] = 0  # Add a central blocked area

        # Additional paths to increase the grid complexity
        self.grid[5, 5:10] = 1
        self.grid[10:15, 3] = 1
        self.grid[15, 15:20] = 1
        self.grid[20, 5:10] = 1
        self.grid[20, 15:25] = 1
        self.grid[5:7, 24] = 0

    def get_node(self, x, y):
        return Node(x, y, self.goal)

# +++++++++++++++++++++++++++++ Algorithm Class ++++++++++++++++++++++++++++++++++++++++++++++++++++++
class algorithm(Grid):
    def __init__(self, width, height, start, goal):
        super().__init__(width, height, start, goal)
        self.path = []
        self.images_List = []  # To store image paths

    # ***************************************** Visualizations ***************************************************  
    def visualize_path(self, path, AlGname):  # Image of the result
        # ---------- The final path image -------------
        plt.figure(figsize=(6, 6))
        plt.imshow(self.grid, cmap="gray_r")
        plt.scatter(self.start[1], self.start[0], color="green", label="Start")
        plt.scatter(self.goal[1], self.goal[0], color="red", label="Goal")
        path = np.array(path)
        if len(path) > 1:
            plt.plot(path[:, 1], path[:, 0], color="blue", label="Path")
        plt.legend()
        plt.title(f"Final Path - {AlGname}")
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(f"Final Path - {AlGname}.png")
        plt.close()

        # ---------- The explored nodes image -------------
        plt.figure(figsize=(6, 6))
        plt.imshow(self.grid, cmap="gray_r")
        plt.scatter(self.start[1], self.start[0], color="green", label="Start")
        plt.scatter(self.goal[1], self.goal[0], color="red", label="Goal")
        all_moves = np.array(self.all_moves)
        plt.scatter(all_moves[:, 1], all_moves[:, 0], color="lightblue", alpha=0.5, label="Explored Nodes")
        plt.legend(loc="upper right")
        plt.title(f"Explored Nodes - {AlGname}")
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(f"{AlGname}_final_result.png")
        plt.close()

    def images_frames(self, algoritmName, path, all_moves):  # Make the frames of the final path and algorithm moves
        images_List = []  # For path formation
        WholeMoves = []  # For exploration process

        # ----------------------------------- Frames of the final path -----------------------------
        current_path = []
        for i in range(len(path)):
            current_path.append(path[i])
            plt.figure(figsize=(6, 6))
            plt.imshow(self.grid, cmap="gray_r")
            plt.scatter(self.start[1], self.start[0], color="green")
            plt.scatter(self.goal[1], self.goal[0], color="red")

            # Plot current path
            if len(current_path) > 1:
                current = np.array(current_path)
                plt.plot(current[:, 1], current[:, 0], color="blue")

            plt.title(f"Path Formation {algoritmName} - Step {i}")
            plt.xticks([])
            plt.yticks([])

            path_dir = f"images_of_{algoritmName}_path"
            os.makedirs(path_dir, exist_ok=True)
            image_path = os.path.join(path_dir, f"path_{i}.png")
            plt.savefig(image_path)
            images_List.append(image_path)
            plt.close()

        # ===================================================================================
        # ------------------------------ Whole algorithm moves trace --------------------------
        explored_nodes = []
        for i, move in enumerate(all_moves):
            # --------------- Plot the move ----------------
            explored_nodes.append(move)  # Add the move to the list of explored nodes
            plt.figure(figsize=(6, 6))
            plt.imshow(self.grid, cmap="gray_r")
            plt.scatter(self.start[1], self.start[0], color="green")
            plt.scatter(self.goal[1], self.goal[0], color="red")
            # The explored nodes plot
            explored = np.array(explored_nodes)
            plt.scatter(explored[:, 1], explored[:, 0], color="grey", alpha=0.5, label="Explored Nodes")
            plt.title(f"Exploration of {algoritmName} in the Space - Step {i}")
            plt.xticks([])
            plt.yticks([])
            plt.legend(loc="upper right")

            # --------------- Saving frames ----------------
            exploration_dir = f"images_of_{algoritmName}_exploration"
            os.makedirs(exploration_dir, exist_ok=True)  # Directory for frames of all algorithm moves
            AllMoves_img_path = os.path.join(exploration_dir, f"explore_{i}.png")  # The frame path
            plt.savefig(AllMoves_img_path)
            WholeMoves.append(AllMoves_img_path)
            plt.close()
            # ----------------------------------------------

        self.images_List = images_List  # Final path frames
        self.wholeMoves = WholeMoves  # Whole algorithm moves frames
        self.all_moves = all_moves

        return images_List, WholeMoves

    # -------------------------------------- Video Making -----------------------------------------------
    def videoFrom_images(self, algoName, fps=5):  # Make the video of the final path
        output_video_path = f"{algoName}_path_formation.mp4"
        with imageio.get_writer(output_video_path, fps=fps) as video_writer:
            for image_path in self.images_List:
                video_writer.append_data(imageio.imread(image_path))
        print("Path formation video created successfully!")

    def videoFrom_movements(self, algoName, fps=5):  # Make the video of the whole algorithm moves
        output_video_path = f"{algoName}_exploration.mp4"
        with imageio.get_writer(output_video_path, fps=fps) as video_writer:
            for image_path in self.wholeMoves:
                video_writer.append_data(imageio.imread(image_path))
        print("Exploration process video created successfully!")

    # -------------------------------------- Algorithms Implementation -----------------------------------------------
    def bfs(self):
        queue = [self.start]
        visited = set()
        parent = {}
        parent[self.start] = None
        all_moves = []

        while queue:
            current = queue.pop(0)
            all_moves.append(current)  # Add the move made by the algorithm in the search
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
            current = parent.get(current, None)
        path = path[::-1]

        return path, 'BFS', all_moves  # Return the path and all moves made by the algorithm

    def dfs(self):
        stack = [self.start]
        visited = set()
        parent = {}
        parent[self.start] = None
        all_moves = []

        while stack:
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
            current = parent.get(current, None)
        path = path[::-1]

        return path, 'DFS', all_moves  # Return the path and all moves made by the algorithm

    

# -------------------------------------- Execute Algorithm Function -----------------------------------------------
def execute_algorithm(algorithmName, width=25, height=25, start=(1, 1), goal=(23, 23)):
    """
    Executes the specified algorithm and generates visualizations.

    Parameters:
    - algorithmName: str, one of "BFS", "DFS", "A*"
    - width: int, width of the grid
    - height: int, height of the grid
    - start: tuple, starting coordinates
    - goal: tuple, goal coordinates

    Returns:
    - dict containing paths to the generated exploration video, path formation video, and final image.
    """
    print(f"Starting {algorithmName} algorithm... (execute_algorithm)")
    start_time = time.time()

    grid = algorithm(width, height, start, goal)

    if algorithmName == "BFS":
        path, AlGname, all_movements = grid.bfs()
    elif algorithmName == 'DFS':
        path, AlGname, all_movements = grid.dfs()
    elif algorithmName == 'A*':
        path, AlGname, all_movements = grid.astar()
    else:
        raise ValueError("Unknown algorithm")

    # Generate frames for visualization
    grid.images_frames(AlGname, path, all_movements)

    # Generate videos
    grid.videoFrom_images(AlGname)
    grid.videoFrom_movements(AlGname)

    # Generate final path image
    grid.visualize_path(path, AlGname)  # Image of result

    end_time = time.time()
    print(f"{algorithmName} algorithm completed in {end_time - start_time:.2f} seconds.")
    print("Done")

    # Return file paths
    return {
        'exploration_video': f"{AlGname}_exploration.mp4",
        'path_video': f"{AlGname}_path_formation.mp4",
        'final_image': f"Final Path - {AlGname}.png"
    }

# -------------------------------------- Main Function for Testing -----------------------------------------------
