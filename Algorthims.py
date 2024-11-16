from Node_Grid import *

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
    
    #----------------------------------------IDS-------------------------------------

    def depth_limited_search(self, current, limit, parent, visited, all_moves):
    # Track the number of visited nodes
        visited_nodes_count = len(visited)  # Store the count of visited nodes

    # Monitor memory usage of key data structures
        current_memory_usage = sys.getsizeof(visited) + sys.getsizeof(parent) + sys.getsizeof(all_moves)
        self.max_memory_usage = max(self.max_memory_usage, current_memory_usage)

        if current == self.goal:
            return True

        if limit <= 0:
            return False

        visited.add(current)  # Add current node to the visited set
        visited_nodes_count += 1  # Increment the count of visited nodes

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

            # Call the recursive DFS and track visited nodes
                if self.depth_limited_search(neighbor, limit - 1, parent, visited, all_moves):
                    return True

        return False

    def depth_limited_search(self, current, limit, parent, visited, all_moves):
    # Track the number of visited nodes
        visited_nodes_count = len(visited)  # Store the count of visited nodes

    # Update the maximum visited nodes count if the current count is greater
        self.max_visited_nodes = max(self.max_visited_nodes, visited_nodes_count)

        if current == self.goal:
            return True

        if limit <= 0:
            return False

        visited.add(current)  # Add current node to the visited set
        visited_nodes_count += 1  # Increment the count of visited nodes

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

            # Call the recursive DFS and track visited nodes
                if self.depth_limited_search(neighbor, limit - 1, parent, visited, all_moves):
                    return True

        return False

    def iterative_deepening_search(self):
        depth = 0
        parent = {self.start: None}
        all_moves = []  # Store moves for visualization
        final_moves = []  # To store moves of the last iteration
        total_visited_nodes = 0  # To track the total number of visited nodes
        self.max_visited_nodes = 0  # Initialize max visited nodes

        while True:
            visited = set()
            iteration_moves = []  # Track moves for the current depth iteration

        # Perform depth-limited search
            if self.depth_limited_search(self.start, depth, parent, visited, iteration_moves):
                final_moves = iteration_moves  # Store the moves of the last successful iteration
                break

        # Count the number of visited nodes for this iteration
            visited_nodes_this_iteration = len(visited)
            total_visited_nodes += visited_nodes_this_iteration  # Increment total visited nodes count

        # Update the maximum visited nodes after each depth iteration
            self.max_visited_nodes = max(self.max_visited_nodes, visited_nodes_this_iteration)

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

    # Print or log the total visited nodes during the search process
        print(f"Total number of visited nodes: {total_visited_nodes}")
        print(f"Maximum number of visited nodes during IDS: {self.max_visited_nodes}")

        return path[::-1], "IDS", final_moves, self.max_visited_nodes
    
    
    # 2-Mohamed--->UCS
     #------------------------UCS----------------------  
    def ucs(self):
        from heapq import heappop, heappush  

        priority_queue = [] 
        heappush(priority_queue, (0, self.start))
        visited = set()
        parent = {self.start: None}
        costs = {self.start: 0} 
        all_moves = []  
        memory_max = 1  

        while priority_queue:
            memory_max = max(memory_max, len(priority_queue))  
            current_cost, current = heappop(priority_queue)
            all_moves.append(current)

            if current == self.goal:
                break 

            visited.add(current)

            
            for dx, dy in self.Moves:
                neighbor = (current[0] + dx, current[1] + dy)

                if (0 <= neighbor[0] < self.width and
                        0 <= neighbor[1] < self.height and
                        self.grid[neighbor[0], neighbor[1]] == 1 and
                        neighbor not in visited):
                    new_cost = current_cost + 1 

                    if neighbor not in costs or new_cost < costs[neighbor]:
                        costs[neighbor] = new_cost
                        heappush(priority_queue, (new_cost, neighbor))
                        parent[neighbor] = current


        path = []
        current = self.goal
        while current:
            path.append(current)
            current = parent.get(current, None)

        return path[::-1], "UCS", all_moves, memory_max  

  #------------------------Hill Climbing Annealing-----------------------


# 1-Mohamed--->hill climbing
    def hill_climbing(self):
        current = self.start
        path = [current]
        all_moves = [] 
        memory_max = 1  

        while current != self.goal:
            best_next = None
            best_heuristic = float('inf')  

            for move in self.Moves:
                new_x, new_y = current[0] + move[0], current[1] + move[1]
                if 0 <= new_x < self.width and 0 <= new_y < self.height and self.grid[new_x, new_y] == 1:
                    heuristic = abs(new_x - self.goal[0]) + abs(new_y - self.goal[1])  # Manhattan distance
                    if heuristic < best_heuristic:
                        best_heuristic = heuristic
                        best_next = (new_x, new_y)

            if best_next is None or best_heuristic >= abs(current[0] - self.goal[0]) + abs(current[1] - self.goal[1]):
                print("Stuck at a local maximum or goal not reachable.")
                break

            current = best_next
            path.append(current)
            all_moves.append(current)
            memory_max = max(memory_max, len(all_moves))  

        return path, "Hill Climbing", all_moves, memory_max

 

    #------------------------Simulated Annealing-----------------------
    def simulated_annealing(self, max_iterations=10000):
        def schedule(t):
            return max(0.01, 1 - 0.001 * t)

        current_state = self.start
        path = [current_state]
        all_moves = []

        max_memory = 0  # To track the maximum memory usage
        for t in range(max_iterations):  # Limit the number of iterations
            T = schedule(t)
            if current_state == self.goal or T == 0:
                return path, "Simulated Annealing", all_moves, max_memory

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

        # Track the maximum memory usage (size of all_moves)
            max_memory = max(max_memory, len(all_moves))

        # Calculate heuristic values (Manhattan distance)
            current_heuristic = abs(current_state[0] - self.goal[0]) + abs(current_state[1] - self.goal[1])
            next_heuristic = abs(next_state[0] - self.goal[0]) + abs(next_state[1] - self.goal[1])
            delta = current_heuristic - next_heuristic

        # Accept or reject move
            if delta > 0 or random() < exp(delta / T):
                current_state = next_state
                path.append(current_state)

        print(f"Maximum memory used (max nodes in all iterations): {max_memory}")
        print("Maximum iterations reached!")
        return path, "Simulated Annealing", all_moves, max_memory
