import numpy as np
import matplotlib.pyplot as plt
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
    
    def visualize_path(self,path,AlGname): # image of the result
        self.grid[self.start[0], self.start[1]] = 2  # start
        self.grid[self.goal[0], self.goal[1]] = 3  # goal

        for (x, y) in path:
            self.grid[x, y] = 4  # path

        plt.figure(figsize=(6,6))
        plt.imshow(self.grid,cmap="gray_r")
        plt.scatter(self.start[1],self.start[0],color="green", label="Start")
        plt.scatter(self.goal[1],self.goal[0],color="red",label="Goal")
        path = np.array(path)
        for i in range(1, path.shape[0] - 1):
            plt.scatter(path[i,1],path[i,0],color="blue")
        plt.legend()
        plt.title(f"Map - path using {AlGname}")
        plt.axis('off')
        # make no x or y axis appear 
        plt.xticks([])
        plt.yticks([])
        plt.savefig(f"{AlGname}_image_result.png")
        plt.show()
        plt.pause(10)
        plt.close()

    def images_frames(self, algoritmName, path,all_moves): # make the frames of the final path, and other for algirtms moves
        images_List = []
        WholeMoves = []
        # ----------------------------------- Frames of the final path -----------------------------
        for i in range(len(path)): # plot the grid with the point we moved with
            plt.figure(figsize=(6, 6))
            plt.imshow(self.grid, cmap="gray_r")
            plt.scatter(self.start[1], self.start[0], color="green", label="Start")
            plt.scatter(self.goal[1], self.goal[0], color="red", label="Goal")
            # path till now 
            for j in range(1,len(path)-1):  
                plt.scatter(path[j][1], path[j][0], color="blue")
            plt.legend()
            # make no x or y axis appear 
            plt.xticks([])
            plt.yticks([])
            plt.title(f"Map {algoritmName}-Step {i}")
            # make a directory to save the images with each time to avoid mess 
            os.makedirs(f"images_of {algoritmName}", exist_ok=True) #  save the directory with the name of algorithm
            image_path = f"images_of {algoritmName}/image_{i}.png" # save the image with the name of the algorithm, from directory we made below
            plt.savefig(image_path)
            images_List.append(image_path) # list with saved images
            plt.close() # we dont want to show them

        self.images_List = images_List  # store the list in the class
        # ===================================================================================
        # ------------------------------ whole algoirtm moves, trace how the algoritm move  ------------------------------
        plt.figure(figsize=(6, 6))
        plt.imshow(self.grid, cmap="gray_r")
        plt.scatter(self.start[1], self.start[0], color="green", label="Start")
        plt.scatter(self.goal[1], self.goal[0], color="red", label="Goal")
        
        # plot
        for move in all_moves[:i + 1]:  
            plt.scatter(move[1], move[0], color="blue")
        plt.legend()
        plt.xticks([])
        plt.yticks([])
        plt.title(f"Map {algoritmName} - Whole Moves Step {i}")

        os.makedirs(f"images_of {algoritmName}_path", exist_ok=True) #  directory for frames of all algoritm moves
        image_wholeMoves_path = f"images_of {algoritmName}_path/image_wholeMoves_{i}.png" 
        WholeMoves.append(image_wholeMoves_path)
        self.wholeMoves = WholeMoves  # Store the list in the class
        # ---------------------------------------------------------------------------------------------


        return images_List,WholeMoves


    def videoFrom_images(self, algoName, fps=5):  # make the video of the final path
        output_video_path = f"{algoName}_video.mp4"
        with imageio.get_writer(output_video_path, fps=fps) as video_writer:
            for image_path in self.images_List:
                video_writer.append_data(imageio.imread(image_path))
        print("Final Path video created successfully!")

    def videoFrom_movements(self, algoName, fps=5): #  make the video of the whole algoritm moves, how it worked
        output_video_path = f"{algoName}_WholeMoves_video.mp4"
        with imageio.get_writer(output_video_path, fps=fps) as video_writer:
            for image_path in self.wholeMoves:
                video_writer.append_data(imageio.imread(image_path))
        print("Algoritm movments video created successfully!")
# ****************************************************************************************************
   
   
   
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



if __name__ == "__main__":
    width = 25 
    height = 25
    start = (0,0)
    goal = (24,24)
    algorithmName = input('select number of algorithm: 1: BFS, 2: DFS, 3: A*')
    grid = algorithm(width, height, start, goal)

    if algorithmName == "1":
        path, AlGname, all_movments = grid.bfs()
    elif algorithmName == '2':
        path, AlGname,all_movments = grid.dfs()
    elif algorithmName == '3':
        path, AlGname,all_movments = grid.astar()


    # frames that make video
    grid.images_frames(AlGname, path,all_movments )
    # video
    grid.videoFrom_images(AlGname)
    grid.visualize_path(path,AlGname,) # image of result
    print(path)
    print("Done")
