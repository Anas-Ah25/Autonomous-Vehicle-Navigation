import numpy as np
import matplotlib.pyplot as plt
from random import choice, random
from math import exp
from itertools import count
import os
import imageio
from heapq import heappop, heappush

# ---------------- Node and Grid Classes ----------------
class Node:
    def __init__(self, x, y, goal,g=0):
        self.x = x
        self.y = y
        self.h = self.heuristic(goal)  # heuristic
        self.g = g  # cost
        self.f = self.g + self.h  # total cost (astar)

    def heuristic(self, goal):
        return abs(self.x - goal[0]) + abs(self.y - goal[1])
    

class Grid:
    def __init__(self,width,height,start,goal):
        self.width = width
        self.height = height
        self.grid = np.zeros((width, height))
        self.Moves = [(0, 1),(0, -1),(1, 0),(-1, 0)]
        self.start = start
        self.goal = goal
        self.pattern()

    def pattern(self): # complex pattern edited by gpt
 
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
        
        # Add random obstacles within the grid to create dead-ends and complexity
        np.random.seed(0)  # Seed for reproducibility
        obstacles = np.random.randint(1, self.width - 1, size=(int(self.width * self.height * 0.15), 2))
        for (x, y) in obstacles:
            if (x, y) != self.start and (x, y) != self.goal:  # Avoid blocking start or goal
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

    def get_node(self,x,y):
        return Node(x,y,self.goal)
# ------------------------------------------------------------------


# ***************************************** Algorithm ***************************************************
class algorithm(Grid):
    def __init__(self, width, height, start, goal):
        super().__init__(width, height, start, goal)
        self.path = []
        self.images_List = []  # Added to store image paths
# ****************************************************************************************************
# ***************************************** Visulizations ***************************************************  
    
    def visualize_path(self, path, AlGname):
        plt.close()  # Close any open plots
        plt.figure(figsize=(6, 6))  # Create a new figure

    # Display the grid and add start/goal markers
        plt.imshow(self.grid, cmap="gray_r")
        plt.scatter(self.start[1], self.start[0], color="green", label="Start")
        plt.scatter(self.goal[1], self.goal[0], color="red", label="Goal")

    # Plot the path
        if path:
            path = np.array(path)
            plt.scatter(path[:, 1], path[:, 0], color="blue", label="Path")

        plt.legend()
        plt.title(f"Path Formation {AlGname}")
        plt.xticks([])  # Hide the x-ticks
        plt.yticks([])  # Hide the y-ticks
    
    # Saving the image
        os.makedirs(f"images_of_{AlGname}_path", exist_ok=True)
        image_path = f"images_of_{AlGname}_path/final_path.png"
        plt.savefig(image_path)

    # Show the image
        plt.show()  # Ensure the image is displayed on screen
        print(f"Image of the path has been saved at {image_path}")
        plt.close()  # Close the plot to avoid overlapping with other figures

        

    def images_frames(self, algoritmName, path, all_moves): # make the frames of the final path, and other for algirtms moves
        images_List = []  # for path formation
        WholeMoves = []  # for exploration process
        
        # ----------------------------------- Frames of the final path -----------------------------
        current_path = []
        for i in range(len(path)):
            current_path.append(path[i])
            plt.figure(figsize=(6, 6))
            plt.imshow(self.grid, cmap="gray_r")
            plt.scatter(self.start[1], self.start[0], color="green", label="Start")
            plt.scatter(self.goal[1], self.goal[0], color="red", label="Goal")
            
            # Plot current path
            if len(current_path) > 1:
                current = np.array(current_path)
                plt.scatter(current[1:,1], current[1:,0], color="blue", label="Path")
            
            plt.legend()
            plt.title(f"Path Formation {algoritmName} Last Itteration - Step {i}")
            plt.xticks([])
            plt.yticks([])
            
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
        explored_nodes = []
        for i, move in enumerate(all_moves):
            #  --------------- plot the move ----------------
            explored_nodes.append(move) # add the move to the list of explored nodes
            plt.figure(figsize=(6, 6))
            plt.imshow(self.grid, cmap="gray_r")
            plt.scatter(self.start[1], self.start[0], color="green")
            plt.scatter(self.goal[1], self.goal[0], color="red")
            # the explored nodes plot
            explored = np.array(explored_nodes) # 
            plt.scatter(explored[:,1], explored[:,0], color="grey", alpha=0.5, label="Explored")
            plt.legend(loc="upper right")
            plt.title(f"Exploration of {algoritmName} in the space - Step {i}")
            plt.xticks([])
            plt.yticks([])
            # --------------------------------------------------------
            # --------------- saving frames ----------------
            os.makedirs(f"images_of_{algoritmName}_exploration", exist_ok=True)  #  directory for frames of all algoritm moves
            AllMoves_img_path = f"images_of_{algoritmName}_exploration/explore_{i}.png" # the frame path
            plt.savefig(AllMoves_img_path)
            WholeMoves.append(AllMoves_img_path)
            plt.close()
            # ----------------------------------------------
        self.images_List = images_List # final path frames
        self.wholeMoves = WholeMoves # whole algoritm moves frames
        self.all_moves = all_moves
        
        return images_List, WholeMoves

    def videoFrom_images(self, algoName, fps=5):  # make the video of the final path
        output_video_path = f"{algoName}_path_formation.mp4"
        with imageio.get_writer(output_video_path, fps=fps) as video_writer:
            for image_path in self.images_List:
                video_writer.append_data(imageio.imread(image_path))
        print("Path formation video created successfully!")

    def videoFrom_movements(self, algoName, fps=5): #  make the video of the whole algoritm moves
        output_video_path = f"{algoName}_exploration.mp4"
        with imageio.get_writer(output_video_path, fps=fps) as video_writer:
            for image_path in self.wholeMoves:
                video_writer.append_data(imageio.imread(image_path))
        print("Exploration process video created successfully!")
   
   
    # ===============================================================================================================
    # ============================== Algorithms ====================================================================
    def bfs(self):
        queue = [self.start]
        visited = set()
        parent = {}
        parent[self.start] = None
        all_moves = []

        while queue:
            current = queue.pop(0)
            all_moves.append(current) # add the move made by the algorithm in the search
            if current == self.goal:
                break
            visited.add(current)
            for i, j in self.Moves:
                new_x, new_y = current[0] + i, current[1] + j
                if 0 <= new_x < self.width and 0 <= new_y < self.height and self.grid[new_x, new_y] == 1 and (new_x, new_y) not in visited:
                    queue.append((new_x, new_y))
                    visited.add((new_x, new_y))
                    parent[(new_x, new_y)] = current

        path = []
        current = self.goal
        while current:
            path.append(current)
            current = parent[current]
        return path[::-1], 'BFS',all_moves
    # -------------------- Depth First Search --------------------
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
                if 0 <= new_x < self.width and 0 <= new_y < self.height and self.grid[new_x, new_y] == 1 and (new_x, new_y) not in visited:
                    stack.append((new_x, new_y))
                    visited.add((new_x, new_y))
                    parent[(new_x, new_y)] = current
        path = []
        while current:
            path.append(current)
            current = parent[current]

        return path[::-1], all_moves  # Return the path and all moves made by the algorithm

    # -------------------- A* Search --------------------


    # ----------------------------------------------------
    # -------------------- Greedy Best First Search --------------------



    #===============================================================================================================
    #===============================================================================================================
    # -------------------- IDS --------------------
    def depth_limited_search(self, current, limit, parent, visited, all_moves):
        if current == self.goal:
            return True
        if limit <= 0:
            return False

        visited.add(current)
        for dx, dy in self.Moves:
            neighbor = (current[0] + dx, current[1] + dy)
            if (
                0 <= neighbor[0] < self.width and
                0 <= neighbor[1] < self.height and
                self.grid[neighbor[0], neighbor[1]] == 1 and
                neighbor not in visited
            ):
                parent[neighbor] = current
                all_moves.append(neighbor)
                if self.depth_limited_search(neighbor, limit - 1, parent, visited, all_moves):
                    return True

        return False
    def iterative_deepening_search(self):
        depth = 0
        parent = {self.start: None}
        all_moves = []  # Store moves for visualization
        final_moves = []  # To store moves of the last iteration

        while True:
            visited = set()
            iteration_moves = []  # Track moves for the current depth iteration

            # Perform depth-limited search
            if self.depth_limited_search(self.start, depth, parent, visited, iteration_moves):
                final_moves = iteration_moves  # Store the moves of the last successful iteration
                break

            depth += 1
            all_moves.append(iteration_moves)  # Track moves across all iterations

    # Reconstruct the final path
        path = []
        current = self.goal
        while current:
            path.append(current)
            current = parent[current]

        # Visualize the last iteration's steps using the existing function
        self.images_frames("IDS", path, final_moves)  # Frames of the last iteration's path and moves
        self.videoFrom_images("IDS")  # Video for the last iteration's path
        self.visualize_path(path, "IDS")  # Image of the final path

        return path[::-1], "IDS", final_moves  # Return the path and moves from the last iteration


    #---------------------Simulated Annealing------------------------------
    def simulated_annealing(self, max_iterations=10000):
        def schedule(t):
            return max(0.01, 1 - 0.001 * t)

        current_state = self.start
        path = [current_state]
        all_moves = []

        for t in range(max_iterations):  # Limit the number of iterations
            T = schedule(t)
            if current_state == self.goal or T == 0:
                return path, "Simulated Annealing", all_moves

            # Get valid neighbors
            neighbors = [
            (current_state[0] + dx, current_state[1] + dy)
            for dx, dy in self.Moves
            if 0 <= current_state[0] + dx < self.width
            and 0 <= current_state[1] + dy < self.height
            and self.grid[current_state[0] + dx, current_state[1] + dy] == 1
        ]

            if not neighbors:
                break  # No valid moves, stop the search

            next_state = choice(neighbors)
            all_moves.append(next_state)

            # Calculate heuristic values (Manhattan distance)
            current_heuristic = abs(current_state[0] - self.goal[0]) + abs(current_state[1] - self.goal[1])
            next_heuristic = abs(next_state[0] - self.goal[0]) + abs(next_state[1] - self.goal[1])
            delta = current_heuristic - next_heuristic

        # Accept or reject move
            if delta > 0 or random() < exp(delta / T):
                current_state = next_state
                path.append(current_state)

        print("Maximum iterations reached!")
        return path, "Simulated Annealing", all_moves

        
    



if __name__ == "__main__":
    width = 25 
    height = 25
    start = (0,0)
    goal = (24,24)
    algorithmName = input('select number of algorithm: 1: BFS, 2: DFS, 3: A*,4: IDS,5: Simulated Annealing ')
    grid = algorithm(width, height, start, goal)

    if algorithmName == "1":
        path, AlGname, all_movments = grid.bfs()
    elif algorithmName == '2':
        path, AlGname,all_movments = grid.dfs()
    elif algorithmName == '3':
        path, AlGname,all_movments = grid.astar()
    elif algorithmName == '4':
        path, AlGname, final_moves = grid.iterative_deepening_search()
        grid.images_frames(AlGname, path, final_moves)  # Frames of the final path and moves
        grid.videoFrom_images(AlGname)  # Video for the path
        grid.visualize_path(path, AlGname)
        print(path)
        print("Done")
    elif algorithmName == '5':
        path, AlGname,all_movments = grid.simulated_annealing(max_iterations=5000)


    if algorithmName != '4':
        # frames that make video
        grid.images_frames(AlGname, path,all_movments)
        # video
        grid.videoFrom_images(AlGname)
        grid.visualize_path(path,AlGname,) # image of result
        print(path)
        print("Done")
