import numpy as np
import matplotlib.pyplot as plt
import os
import imageio

class Node:
    def __init__(self, x, y, goal):
        self.x = x
        self.y = y
        self.h = self.calculate_heuristic(goal)  # heuristic (manhattan distance)
        self.g = 1  # cost to move to next node
        self.f = self.g + self.h  # total cost for A* algorithm

    def calculate_heuristic(self, goal):
        return abs(self.x - goal[0]) + abs(self.y - goal[1])

class Grid:
    def __init__(self, width, height, start, goal):
        self.width = width
        self.height = height
        self.grid = np.zeros((width, height))
        self.Moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # possible moves: right, left, down, up
        self.start = start
        self.goal = goal
        self.pattern()

    def pattern(self):
        # Create a maze-like pattern for the game
        # Walls and obstacles (1 represents walkable path)
        self.grid[0,:] = 1  # Top wall
        self.grid[1:5,9] = 1  # Right vertical corridor
        self.grid[5,:] = 1   # Middle horizontal corridor
        self.grid[6, 2:6] = 1  # Lower horizontal path
        self.grid[7:9,4] = 1  # Lower vertical path
        self.grid[9,:] = 1  # Bottom wall
        self.grid[:,2] = 1  # Left vertical corridor
        self.grid[:,4] = 1  # Middle vertical corridor

    def get_node(self,x,y):
        return Node(x,y,self.goal)

class algorithm(Grid):
    def __init__(self, width, height, start, goal):
        super().__init__(width, height, start, goal)
        self.path = []
        self.images_List = []  # Store image paths for animation

    def setup_plot(self):
        """Setup common plotting parameters for consistent visualization"""
        plt.figure(figsize=(8, 8))
        plt.imshow(self.grid, cmap='binary')  # Using binary colormap for better contrast
        plt.axis('off')  # Remove axes
        plt.grid(False)  # Remove grid
        
        # Add game-like markers for start and goal
        plt.scatter(self.start[1], self.start[0], color='limegreen', s=200, marker='*', 
                   label='Start', edgecolor='white', linewidth=2)
        plt.scatter(self.goal[1], self.goal[0], color='red', s=200, marker='H', 
                   label='Goal', edgecolor='white', linewidth=2)

    def visualize_path(self,path,AlGname):
        # Visualize final path
        self.setup_plot()
        path = np.array(path)
        
        # Plot path with a nicer visual style
        if len(path) > 1:
            plt.plot(path[1:-1,1], path[1:-1,0], 'o-', color='dodgerblue', 
                    linewidth=3, markersize=10, markeredgecolor='white',
                    markeredgewidth=2, label='Path')
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
        plt.tight_layout()
        plt.savefig(f"{AlGname}_image_result.png", bbox_inches='tight', dpi=150)
        plt.show()
        plt.pause(10)
        plt.close()

    def images_frames(self, algoritmName, path):
        images_List = []
        
        for i in range(len(path)):
            self.setup_plot()
            
            # Plot path up to current position
            current_path = path[:i+1]
            if len(current_path) > 1:
                path_array = np.array(current_path)
                plt.plot(path_array[:,1], path_array[:,0], 'o-', color='dodgerblue', 
                        linewidth=3, markersize=10, markeredgecolor='white',
                        markeredgewidth=2)
            
            # Highlight current position
            if i > 0 and i < len(path)-1:
                plt.scatter(path[i][1], path[i][0], color='yellow', s=150, 
                          marker='o', edgecolor='black', linewidth=2, 
                          label='Current Position' if i == 1 else "")
            
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
            plt.tight_layout()
            
            # Save frame
            os.makedirs(f"images_of_{algoritmName}", exist_ok=True)
            image_path = f"images_of_{algoritmName}/image_{i:03d}.png"
            plt.savefig(image_path, bbox_inches='tight', dpi=150)
            images_List.append(image_path)
            
            plt.close()
        
        self.images_List = images_List
        return images_List

    def videoFrom_images(self, algoName, fps=5):  # Increased FPS for smoother animation
        output_video_path = f"{algoName}_video.mp4"
        with imageio.get_writer(output_video_path, fps=fps) as video_writer:
            for image_path in self.images_List:
                video_writer.append_data(imageio.imread(image_path))
        print("Video created successfully!")

    def bfs(self):
        # Breadth First Search implementation
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
                if (0 <= new_x < self.width and 0 <= new_y < self.height and 
                    self.grid[new_x, new_y] == 1 and (new_x, new_y) not in visited):
                    queue.append((new_x, new_y))
                    visited.add((new_x, new_y))
                    parent[(new_x, new_y)] = current

        # Reconstruct path
        path = []
        current = self.goal
        while current:
            path.append(current)
            current = parent.get(current)
        return path[::-1], 'bfs'

if __name__ == "__main__":
    width = 25
    height = 25
    start = (0,9)
    goal = (9,0)
    algorithmName = input('Select number of algorithm: 1: BFS, 2: DFS, 3: A* >>> ')
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
    print(f"Path found: {path}")
    print("Done")
