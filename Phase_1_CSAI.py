import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
from heapq import heappop, heappush
import time
import tkinter as tk
from PIL import Image, ImageTk
import subprocess

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

    def pattern(self): # complex pattern edited by gpt, this is what make the map ways
 
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


# +++++++++++++++++++++++++++++ Algorithm Class ++++++++++++++++++++++++++++++++++++++++++++++++++++++
class algorithm(Grid):
    def __init__(self, width, height, start, goal):
        super().__init__(width, height, start, goal)
        self.path = []
        self.images_List = []  # Added to store image paths
# ****************************************************************************************************
# ***************************************** Visulizations ***************************************************  
    def visualize_path(self, path, AlGname): # image of the result
        #  ---------- The final path image -------------
        plt.imshow(self.grid, cmap="gray_r")
        plt.scatter(self.start[1], self.start[0], color="green")
        plt.scatter(self.goal[1], self.goal[0], color="red")
        path = np.array(path)
        for i in range(1, path.shape[0] - 1):
          plt.scatter(path[i,1], path[i,0], color="blue")
        # plt.legend()
        plt.title(f"Final Path - {AlGname}")
        plt.xticks([])
        plt.yticks([])
        plt.savefig(f"Final Path - {AlGname}.png")
        
        # ---------- The explored nodes image -------------
        plt.imshow(self.grid, cmap="gray_r")
        plt.scatter(self.start[1], self.start[0], color="green")
        plt.scatter(self.goal[1], self.goal[0], color="red")
        all_moves = np.array(self.all_moves)
        plt.scatter(all_moves[:,1], all_moves[:,0], color="lightblue", alpha=0.5)
        # plt.legend(loc="upper right")
        plt.title(f"Explored Nodes - {AlGname}")
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(f"{AlGname}_final_result.png")
        plt.close()
        
        
           
      

    def images_frames(self, algoritmName, path, all_moves): # make the frames of the final path, and other for algirtms moves
        images_List = []  # for path formation
        WholeMoves = []  # for exploration process
        
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
                plt.scatter(current[1:,1], current[1:,0], color="blue", label="Path")
            
            # plt.legend()
            plt.title(f"Path Formation {algoritmName} - Step {i}")
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
            plt.scatter(explored[:,1], explored[:,0],color="grey",alpha=0.5)
            # plt.legend(loc="upper right")
            plt.title(f"Exploration of {algoritmName} in the space - Step {i}")
            plt.xticks([])
            plt.yticks([])
            # --------------------------------------------------------
            # --------------- saving frames ----------------
            os.makedirs(f"images_of_{algoritmName}_exploration",exist_ok=True)  #  directory for frames of all algoritm moves
            AllMoves_img_path = f"images_of_{algoritmName}_exploration/explore_{i}.png" # the frame path
            plt.savefig(AllMoves_img_path)
            WholeMoves.append(AllMoves_img_path)
            plt.close()
            # ----------------------------------------------
        self.images_List = images_List # final path frames
        self.wholeMoves = WholeMoves # whole algoritm moves frames
        self.all_moves = all_moves
        
        return images_List, WholeMoves
    

    # ===================================================================================================
    # -------------------------------------- Video making -----------------------------------------------
    # ===================================================================================================
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
# ****************************************************************************************************
   
   
   
    # ===============================================================================================================
    # ============================== Algorithms itself ====================================================================
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
        return path[::-1], 'BFS',all_moves # return the path and all moves made by the algorithm
    # --------------------------------------------------------------------------------------------------------
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

        return path[::-1],'DFS', all_moves  # return the path and all moves made by the algorithm

    # -------------------- A* Search --------------------


    # ----------------------------------------------------------------------------
    # -------------------- Greedy Best First Search --------------------

    # ----------------------------------------------------------------------------
    # -------------------- UCS Algorithm --------------------
    # ----------------------------------------------------------------------------


    #===============================================================================================================
    #===============================================================================================================
# -------------------------------- GUI --------------------------------
# GUI Class
class ResultGUI:
    def __init__(self, outputs):
        self.outputs = outputs
        self.root = tk.Tk()
        self.root.title("Algorithm Visualization")

        # Create variables to keep track of images
        self.exploration_images = []
        self.path_images = []
        self.current_index = 0
        self.current_images = []
        self.image_type = 'exploration'  # 'exploration' or 'path'

        # Load images
        self.load_images()

        # Create UI elements
        self.create_widgets()

    def load_images(self):
        # Load exploration images
        exploration_dir = self.outputs.get('exploration_images_dir')
        if exploration_dir and os.path.isdir(exploration_dir):
            exploration_files = sorted(os.listdir(exploration_dir))
            for file in exploration_files:
                if file.endswith('.png'):
                    img_path = os.path.join(exploration_dir, file)
                    image = Image.open(img_path)
                    self.exploration_images.append(image)

        # Load path images
        path_dir = self.outputs.get('path_images_dir')
        if path_dir and os.path.isdir(path_dir):
            path_files = sorted(os.listdir(path_dir))
            for file in path_files:
                if file.endswith('.png'):
                    img_path = os.path.join(path_dir, file)
                    image = Image.open(img_path)
                    self.path_images.append(image)

        # Set current images to exploration images by default
        self.current_images = self.exploration_images

    def create_widgets(self):
        # Create a frame for controls
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.TOP)

        # Create radio buttons to select image type
        self.image_type_var = tk.StringVar(value='exploration')
        self.exploration_radio = tk.Radiobutton(control_frame, text="Exploration Images", variable=self.image_type_var, value='exploration', command=self.change_image_type)
        self.exploration_radio.pack(side=tk.LEFT)

        self.path_radio = tk.Radiobutton(control_frame, text="Path Images", variable=self.image_type_var, value='path', command=self.change_image_type)
        self.path_radio.pack(side=tk.LEFT)

        # Create buttons to navigate images
        self.prev_button = tk.Button(control_frame, text="<< Previous", command=self.prev_image)
        self.prev_button.pack(side=tk.LEFT)

        self.next_button = tk.Button(control_frame, text="Next >>", command=self.next_image)
        self.next_button.pack(side=tk.LEFT)

        # Create buttons to open videos and final image
        self.open_exploration_video_button = tk.Button(control_frame, text="Open Exploration Video", command=self.open_exploration_video)
        self.open_exploration_video_button.pack(side=tk.LEFT)

        self.open_path_video_button = tk.Button(control_frame, text="Open Path Video", command=self.open_path_video)
        self.open_path_video_button.pack(side=tk.LEFT)

        self.open_final_image_button = tk.Button(control_frame, text="Open Final Image", command=self.open_final_image)
        self.open_final_image_button.pack(side=tk.LEFT)

        # Create a canvas to display images
        self.canvas = tk.Canvas(self.root, width=600, height=600)
        self.canvas.pack()

        # Display the first image
        self.display_image()

    def change_image_type(self):
        self.image_type = self.image_type_var.get()
        if self.image_type == 'exploration':
            self.current_images = self.exploration_images
        elif self.image_type == 'path':
            self.current_images = self.path_images
        self.current_index = 0
        self.display_image()

    def display_image(self):
        # Clear the canvas
        self.canvas.delete("all")
        # Display the current image on the canvas
        if self.current_images:
            image = self.current_images[self.current_index]
            # Resize image to fit the canvas if necessary
            image = image.resize((600, 600), Image.ANTIALIAS)
            self.photo = ImageTk.PhotoImage(image)
            self.canvas.create_image(300, 300, image=self.photo)
        else:
            # No images to display
            self.canvas.create_text(300, 300, text="No images to display")

    def prev_image(self):
        if self.current_images:
            self.current_index = (self.current_index - 1) % len(self.current_images)
            self.display_image()

    def next_image(self):
        if self.current_images:
            self.current_index = (self.current_index + 1) % len(self.current_images)
            self.display_image()

    def open_exploration_video(self):
        video_path = self.outputs.get('exploration_video')
        if video_path and os.path.isfile(video_path):
            if os.name == 'nt':
                os.startfile(video_path)
            elif os.name == 'posix':
                subprocess.call(('xdg-open', video_path))

    def open_path_video(self):
        video_path = self.outputs.get('path_video')
        if video_path and os.path.isfile(video_path):
            if os.name == 'nt':
                os.startfile(video_path)
            elif os.name == 'posix':
                subprocess.call(('xdg-open', video_path))

    def open_final_image(self):
        image_path = self.outputs.get('final_image')
        if image_path and os.path.isfile(image_path):
            if os.name == 'nt':
                os.startfile(image_path)
            elif os.name == 'posix':
                subprocess.call(('xdg-open', image_path))

    def run(self):
        self.root.mainloop()



# -------- run the app ------------

def execute_algorithm(algorithmName, width=25, height=25, start=(0, 0), goal=(24, 24)):

    print(f"Starting {algorithmName} algorithm... (algo function)")
    start_time = time.time()
    grid = algorithm(width, height, start, goal)
    if algorithmName == "BFS":
        path, AlGname, all_movements = grid.bfs()
    elif algorithmName == 'DFS':
        path, AlGname, all_movements = grid.dfs()
    # elif algorithmName == 'A*':
   

    # frames that make video
    grid.images_frames(AlGname, path, all_movements)
    # video
    grid.videoFrom_images(AlGname)
    grid.videoFrom_movements(AlGname)
    grid.visualize_path(path, AlGname)  # image of result

    end_time = time.time()
    print(f"{algorithmName} algorithm completed in {end_time - start_time:.2f} seconds.")
    print("Done")
    return {
        'exploration_video': f"{AlGname}_exploration.mp4",
        'path_video': f"{AlGname}_path_formation.mp4",
        'final_image': f"Final Path - {AlGname}.png"
    }






# -------- run the app ------------

if __name__ == "__main__":
    # Example usage
    width = 25
    height = 25
    start = (0, 0)
    goal = (24, 24)
    output = algorithmName = input("Enter the name of the algorithm (BFS, DFS, A*): ")
    execute_algorithm(algorithmName,width,height,start,goal)
    gui = ResultGUI(output)
    gui.run()
