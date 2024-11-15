import numpy as np
import matplotlib.pyplot as plt
import os
import imageio

class Node:
    def __init__(self, x, y, goal):
        self.x = x
        self.y = y
        self.h = self.calculate_heuristic(goal)  # heuristic
        self.g = 1  # cost
        self.f = self.g + self.h  # total cost (astar)

    def calculate_heuristic(self, goal):
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

    def pattern(self):
        self.grid[0,:] = 1  # row 10
        self.grid[1:5,9] = 1  # Vertical line from 1 to row 6
        self.grid[5,:] = 1   # Horizontal path in row 5
        self.grid[6, 2:6] = 1 # Horizontal path in row 6
        self.grid[7:9,4] = 1 
        self.grid[9,:] = 1 
        self.grid[:,2] = 1 
        self.grid[:,4] = 1 

    def get_node(self,x,y):
        return Node(x,y,self.goal)

class algorithm(Grid):
    def __init__(self, width, height, start, goal):
        super().__init__(width, height, start, goal)
        self.path = []
        self.images_List = []  # Added to store image paths
# ****************************************************************************************************
# ***************************************** Visulizations ***************************************************  
    def visualize_path(self,path,AlGname):
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
        plt.title("Map")
        plt.axis('off')
        plt.savefig(f"{AlGname}_image_result.png")
        plt.show()
        plt.pause(10)
        plt.close()

    def images_frames(self, algoritmName, path):
        images_List = []
        
        for i in range(len(path)): # plot the grid with the point we moved with
            plt.figure(figsize=(6, 6))
            plt.imshow(self.grid, cmap="gray_r")
            plt.scatter(self.start[1], self.start[0], color="green", label="Start")
            plt.scatter(self.goal[1], self.goal[0], color="red", label="Goal")
            # path till now 
            for j in range(i + 1):  
                plt.scatter(path[j][1], path[j][0], color="blue")
            plt.legend()
            # make a directory to save the images with each time to avoid mess 
            os.makedirs(f"images_of {algoritmName}", exist_ok=True) #  save the directory with the name of algorithm
            image_path = f"images_of {algoritmName}/image_{i}.png" # save the image with the name of the algorithm, from directory we made below
            plt.savefig(image_path)
            images_List.append(image_path) # list with saved images
            plt.close() # we dont want to show them
        self.images_List = images_List  # Store the list in the class
        return images_List

    def videoFrom_images(self, algoName, fps=2):  # make video from frames of movment
        output_video_path = f"{algoName}_video.mp4"  # algoritm name with file name
        with imageio.get_writer(output_video_path, fps=fps) as video_writer:
            for image_path in self.images_List:
                video_writer.append_data(imageio.imread(image_path))  # list from the images frames making function
        print("Video created successfully!")
# ****************************************************************************************************
   
   
   
    # ===============================================================================================================
    # ============================== Algorithms ====================================================================
    def bfs(self):
        queue = [self.start]
        visited = set()
        parent = {}
        parent[self.start] = None

        while queue:
            current = queue.pop(0)
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
        return path[::-1], 'bfs'
    #===============================================================================================================
    #===============================================================================================================



if __name__ == "__main__":
    width = 10 
    height = 10
    start = (0,9)
    goal = (9,0)
    algorithmName = input('select number of algorithm: 1: bfs, 2: dfs, 3: astar')
    grid = algorithm(width, height, start, goal)

    if algorithmName == "1":
        path, AlGname = grid.bfs()
    elif algorithmName == '2':
        path, AlGname = grid.dfs()

    # Generate and save the frames
    grid.images_frames(AlGname, path)
    # Create the video from the frames
    grid.videoFrom_images(AlGname)
    
    grid.visualize_path(path, AlGname)
    print(path)
    print("Done")
