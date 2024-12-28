from node_grid import *
from libraries import *


# Calculate max memory usage for all algorthims except IDS
def get_memory_size(all_moves, path):
    size = sys.getsizeof(all_moves) + sys.getsizeof(path)
    for move in all_moves:
        size += sys.getsizeof(move)
    for node in path:
        size += sys.getsizeof(node)
    return size / 1024  # Convert to KB

# Calculate max memory usage for  IDS
def get_memory_size_ids(*args): # Variable number of inputs
        total_size = sum(sys.getsizeof(arg) for arg in args)
        return total_size / 1024 # Convert to KB

# Class algorithm containg all algorithms functions 
class algorithm(Grid):
    def __init__(self, width, height, start, goal):
        super().__init__(width, height, start, goal)
        self.path = []
        self.images_List = []  # Added to store image paths

    '''==============================================================================================='''
    '''######################################### Visualization #########################################'''
    '''==============================================================================================='''

    
    def visualize_path(self, path, AlGname): # Create static image for final path and explorations processes
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
        plt.savefig(f"{AlGname}_Final_Path.png")
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
        plt.savefig(f"{AlGname}_explored_map.png")
        plt.close()


    def images_frames(self, algoritmName, path, all_moves):# Create frames for video making for both processes and store it in directories
        images_List = [] # store the frames of final path
        WholeMoves = [] # store the frames of exploration path

        # -------------- Frames of the final path -------------
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
        # ------------------------------ whole algorithm for video making --------------------------
        '''The main logic here is to take the (all_moves) which is the nodes that the algoritm covered
            then plot each move, this plot is like a frame, we make the video from playing all these frames
            after each other, so we created a directory for each type of frames saved during the code running
            this directory has the frames, where we will take the images inside it and pass it to 'videoFrom_movements' 
            and 'videoFrom_images', this will happen after (return images_List, WholeMoves) of the images_frames function we are in now
            those are the lists with  paths for the frames we talked about
        '''
        # -------------- Frames of the exploration path -------------

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
    
    # -------------------------------------- Video making -----------------------------------------------
    
    def videoFrom_images(self, algoName, fps=5): # Create path formation video
        output_video_path = f"{algoName}_path_formation.mp4"
        with imageio.get_writer(output_video_path, fps=fps) as video_writer:
            for image_path in self.images_List:
                video_writer.append_data(imageio.imread(image_path))
        print("Path formation video created successfully!")
 
    def videoFrom_movements(self, algoName, fps=5): # Create video for nodes exploration
        output_video_path = f"{algoName}_exploration.mp4"
        with imageio.get_writer(output_video_path, fps=fps) as video_writer:
            for image_path in self.wholeMoves:
                video_writer.append_data(imageio.imread(image_path))
        print("Exploration process video created successfully!")


    '''==============================================================================================='''
    '''######################################### Search algorithms #########################################'''
    '''==============================================================================================='''


    '''==========================================Breath First Search===================================='''

    def bfs(self):  
            start_time = time.time() # for calculating the total working time 
            queue = [self.start]  
            visited = set()  
            parent = {}  
            parent[self.start] = None
            all_moves = []  # all moves made by the algorithm
            total_search_cost = 0  #  total search cost
            final_path_cost = 0  #  final path cost
            memory_max = sys.getsizeof(queue) # memory

            while queue:
                memory_max = max(memory_max, len(queue))
                current = queue.pop(0)  
                all_moves.append(current)  

                # add the cost of the current node to the total search cost
                total_search_cost += self.cost_map[current[0], current[1]]

                if current == self.goal:
                    break
                visited.add(current) 

                # neighbors
                for i, j in self.Moves:
                    new_x, new_y = current[0] + i, current[1] + j
                    if (0 <= new_x < self.width and 0 <= new_y < self.height and
                        self.grid[new_x, new_y] == 1 and (new_x, new_y) not in visited):
                        queue.append((new_x, new_y))  # add neighbor to the queue
                        visited.add((new_x, new_y))  # mark neighbor as visited
                        parent[(new_x, new_y)] = current  # set parent for path reconstruction
            self.all_moves = all_moves 

            # ----------- Reconstruct the path from the parent ------------
            path = []
            current = self.goal
            while current:
                path.append(current)
                # calculate the final path cost using the cost_map from grid 
                final_path_cost += self.cost_map[current[0], current[1]]
                current = parent[current]  # Move to the parent node

            path.reverse()  # start to end
            end_time = time.time()  
            working_time = end_time - start_time  # total time to run
            memory_max = get_memory_size(all_moves, path)
   
            return path, 'BFS', all_moves,memory_max, total_search_cost, final_path_cost, working_time

    '''==========================================Depth First Search===================================='''

    def dfs(self):
        stack = [self.start]
        visited = set()
        parent = {}
        parent[self.start] = None
        all_moves = []
        memory_max = sys.getsizeof(stack) 
        start_time = time.time()  
        total_search_cost = 0  
        final_path_cost = 0  

        while stack:
            memory_max = max(memory_max, len(stack))
            current = stack.pop()
            all_moves.append(current)
            # Add the cost of the current node to the total search cost
            total_search_cost += self.cost_map[current[0], current[1]]
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

        # Reverse the path to get it from start to goal
        path = []
        current = self.goal
        self.all_moves = all_moves  
        while current:
            path.append(current)
            final_path_cost += self.cost_map[current[0], current[1]]
            current = parent[current]

        end_time = time.time() 
        working_time = end_time - start_time # Total working time
        path.reverse() # Reverse the path to get it from start to end
        memory_max = get_memory_size(all_moves, path) # Max memory usage

    
        return path, 'DFS', all_moves,memory_max, total_search_cost, final_path_cost, working_time 

    '''========================================== IDS =============================================='''  

    def depth_limited_search(self, current, limit, parent, visited, all_moves):
        if current == self.goal:
            return True

        if limit <= 0:
            return False

        visited.add(current)  # Add current node to the visited set

        for dx, dy in self.Moves: # get all possible and valid and not visited neighbors
            neighbor = (current[0] + dx, current[1] + dy)
            if (
                0 <= neighbor[0] < self.width and
                0 <= neighbor[1] < self.height and
                self.grid[neighbor[0], neighbor[1]] == 1 and
                neighbor not in visited
            ):
                parent[neighbor] = current
                all_moves.append(neighbor)

                # Call the recursive depth-limited search
                if self.depth_limited_search(neighbor, limit - 1, parent, visited, all_moves):
                    return True

        return False

    def iterative_deepening_search(self):
        memory_max = 0  
        depth = 0
        parent = {self.start: None}
        all_moves = []  
        total_visited_nodes = 0  
        start_time = time.time()  

        while True:
            # Reset the visited and itteration_moves for this itteration 
            visited = set()
            iteration_moves = []  

            
            if self.depth_limited_search(self.start, depth, parent, visited, iteration_moves):
                break # Goal is found
            # else
            memory_max = max(memory_max, get_memory_size_ids(visited, parent, iteration_moves))

            depth += 1
            all_moves.extend(iteration_moves)  # Append moves from the performed itteration 

        path = []
        current = self.goal
        while current:
            path.append(current)
            current = parent[current]

        path.reverse()  

        total_search_cost = sum(self.cost_map[x[0], x[1]] for x in all_moves)
        final_path_cost = sum(self.cost_map[x[0], x[1]] for x in path)
        end_time = time.time()  
        working_time = end_time - start_time  

        return path, 'IDS', all_moves, memory_max, total_search_cost, final_path_cost, working_time

    '''========================================== UCS =============================================='''

    def ucs(self):
        priority_queue = []
        heappush(priority_queue, (0, self.start))  
        visited = set()
        parent = {self.start: None}  
        costs = {self.start: 0}  
        all_moves = []  
        total_search_cost = 0  
        memory_max = 0  
        start_time = time.time() 

        while priority_queue:
            # Track the maximum memory usage of the priority queue
            memory_max = max(memory_max, sys.getsizeof(priority_queue))

            current_cost, current = heappop(priority_queue) 
            all_moves.append(current)  
            total_search_cost += self.cost_map[current[0], current[1]]
            if current == self.goal:
                break
            visited.add(current)

            # Explore neighbors
            for dx, dy in self.Moves:
                neighbor = (current[0] + dx, current[1] + dy)

                if (0 <= neighbor[0] < self.width and
                        0 <= neighbor[1] < self.height and
                        self.grid[neighbor[0], neighbor[1]] == 1 and
                        neighbor not in visited):
                    new_cost = current_cost + self.cost_map[neighbor[0], neighbor[1]]  # Cost to neighbor

                    # Update neighbor if not visited or found with a lower cost
                    if neighbor not in costs or new_cost < costs[neighbor]:
                        costs[neighbor] = new_cost
                        heappush(priority_queue, (new_cost, neighbor))  # Push neighbor with updated cost
                        parent[neighbor] = current

        # Reconstruct the path
        path = []
        current = self.goal
        final_path_cost = 0  
        while current:
            path.append(current)
            final_path_cost += self.cost_map[current[0], current[1]]  
            current = parent.get(current, None) 

        path.reverse()  
        end_time = time.time() 
        working_time = end_time - start_time  
        memory_max = memory_max / 1024

        return path, 'UCS', all_moves, memory_max, total_search_cost, final_path_cost, working_time

    '''========================================== Greeedy Best First Search =============================================='''

    def greedy_algorithm(self):
        current = self.start
        path = [current]
        all_moves = []
        memory_max = 1
        total_search_cost = 0  
        final_path_cost = 0  
        start_time = time.time()  

        while current != self.goal:
            best_next = None
            best_heuristic = float('inf')
            
            for move in self.Moves:
                new_x, new_y = current[0] + move[0], current[1] + move[1]
                if 0 <= new_x < self.width and 0 <= new_y < self.height and self.grid[new_x, new_y] == 1:
                    heuristic = abs(new_x - self.goal[0]) + abs(new_y - self.goal[1]) 
                    if heuristic < best_heuristic:
                        best_heuristic = heuristic
                        best_next = (new_x, new_y)
                        
            if best_next is None:
                print("Stuck at a local maximum or goal not reachable.")
                break
            
            current = best_next
            path.append(current)
            all_moves.append(current)
            total_search_cost += self.cost_map[current[0], current[1]]  # Add cost of the current node to total search cost
            memory_max = max(memory_max, len(all_moves))
        
        # Calculate the final path cost
        for node in path:
            final_path_cost += self.cost_map[node[0], node[1]]

        end_time = time.time()  
        working_time = end_time - start_time 
        memory_max = get_memory_size(all_moves, path)

        return path, "Greedy", all_moves, memory_max, total_search_cost, final_path_cost, working_time

    '''========================================== A* Search===================================='''

    def a_star(self):
        start_node = Node(self.start[0], self.start[1], self.goal, g=0)  
        frontier = []  # Priority queue 
        heapq.heappush(frontier, (start_node.f,start_node)) # start node with priority of  f, which is the summation of the cost and heuristic
        reached = {self.start: start_node.g}  # dictionary to track the best cost to reach each node
        parent = {self.start: None}  # parent dictionary for path reconstruction
        all_moves = []  
        start_time = time.time()  
        memory_max = sys.getsizeof(frontier)  

        while frontier:
            memory_max = max(memory_max, len(frontier))
            _, current_node = heapq.heappop(frontier)  # pop
            current = (current_node.x, current_node.y)  # get position
            all_moves.append(current)  
            if current == self.goal:
                break

            # ------ neighbors ------
            for i, j in self.Moves:
                new_x, new_y = current[0] + i, current[1] + j
                if 0 <= new_x < self.width and 0 <= new_y < self.height and self.grid[new_x, new_y] == 1:
                    # Calculate the g and f values for the neighbor
                    new_g = current_node.g + self.cost_map[new_x, new_y] # rather than making other cost map combining both
                    neighbor_node = Node(new_x, new_y, self.goal, g=new_g)
                    neighbor = (neighbor_node.x, neighbor_node.y)

                    # If the neighbor is not in reached or has a better cost, update it
                    if neighbor not in reached or new_g < reached[neighbor]:
                        reached[neighbor] = new_g
                        parent[neighbor] = current
                        heapq.heappush(frontier, (neighbor_node.f, neighbor_node))  # Push neighbor with f value

        # Reconstruct the final path
        path = []
        current = self.goal
        final_path_cost = 0  
        while current:
            path.append(current)
            final_path_cost += self.cost_map[current[0], current[1]]  # Add the node's cost to the final path cost
            current = parent[current]

        path.reverse()  # Reverse the path to get it from start to goal
        total_search_cost = sum(self.cost_map[x, y] for x, y in all_moves)  # Calculate the total search cost
        end_time = time.time()  # End timing
        working_time = end_time - start_time  # Calculate the execution time

        # Store all explored nodes
        self.all_moves = all_moves 
        memory_max = get_memory_size(all_moves, path)


        return path, 'Astar', all_moves,memory_max, total_search_cost, final_path_cost, working_time 
    

    '''========================================== Hill Climbing =============================================='''
    
    def hill_climbing(self):
            current = self.start
            path = [current]
            all_moves = [] 
            memory_max = 1  
            total_search_cost = 0  
            final_path_cost = 0  
            start_time = time.time() 

            while current != self.goal:
                best_next = None
                best_heuristic = float('inf')  

                for move in self.Moves:
                    new_x, new_y = current[0] + move[0], current[1] + move[1]
                    if 0 <= new_x < self.width and 0 <= new_y < self.height and self.grid[new_x, new_y] == 1:
                        heuristic = abs(new_x - self.goal[0]) + abs(new_y - self.goal[1])  
                        if heuristic < best_heuristic:
                            best_heuristic = heuristic
                            best_next = (new_x, new_y)

                if best_next is None or best_heuristic >= abs(current[0] - self.goal[0]) + abs(current[1] - self.goal[1]):
                    print("Stuck at a local maximum or goal not reachable.")
                    break

                current = best_next
                path.append(current)
                all_moves.append(current)
                total_search_cost += self.cost_map[current[0], current[1]]  # Increment the total cost with the current node
                memory_max = max(memory_max, len(all_moves))  

            for node in path:  # Calculate the final path cost
                final_path_cost += self.cost_map[node[0], node[1]]

            end_time = time.time()  
            working_time = end_time - start_time
            memory_max = get_memory_size(all_moves, path)

            return path, "Hill Climbing", all_moves, memory_max, total_search_cost, final_path_cost, working_time

    '''========================================== Simulated Annealing =============================================='''

    def simulated_annealing(self, max_iterations=10000):
            def schedule(t):
                return max(0.01, 1 - 0.001 * t)

            current_state = self.start
            path = [current_state]
            all_moves = []
            total_search_cost = 0  
            final_path_cost = 0 
            start_time = time.time() 

            for t in range(max_iterations):
                T = schedule(t)
                if current_state == self.goal or T == 0:
                    break
                neighbors = [  # All possible and valid moves 1 -> valid, 0 -> invalid 
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


                # Calculate heuristic values using Manhattan distance
                current_heuristic = abs(current_state[0] - self.goal[0]) + abs(current_state[1] - self.goal[1])
                next_heuristic = abs(next_state[0] - self.goal[0]) + abs(next_state[1] - self.goal[1])
                delta = current_heuristic - next_heuristic 

                # Add cost of current state to the total search cost
                total_search_cost += self.cost_map[current_state[0], current_state[1]]

                # Accept or reject this move according to the Tempreture
                if delta > 0 or random.random() < exp(delta / T):
                    current_state = next_state
                    path.append(current_state)

            # Calculate the final path cost
            for node in path:
                final_path_cost += self.cost_map[node[0], node[1]]

            end_time = time.time()  
            working_time = end_time - start_time  
            memory_max = get_memory_size(all_moves, path)  

            return path, "Simulated Annealing", all_moves, memory_max, total_search_cost, final_path_cost, working_time


    '''========================================== Genetic =============================================='''  

    def genetic_algorithm(self, pop_size=100, ngen=1000, pmut=0.1):
        start_time = time.time()

        # Helper function to check if a position is valid
        def is_valid(pos):
            x, y = pos
            return (0 <= x < self.width and 0 <= y < self.height and
                    self.grid[x, y] == 1)

        # Generate a random valid path from start towards goal
        def generate_random_path():
            path = [self.start]
            visited = set(path)
            while path[-1] != self.goal:
                current = path[-1]
                neighbors = []
                for move in self.Moves:
                    new_pos = (current[0] + move[0], current[1] + move[1])
                    if is_valid(new_pos) and new_pos not in visited:
                        neighbors.append(new_pos)
                if not neighbors:
                    break  # Dead end reached
                next_pos = random.choice(neighbors)
                path.append(next_pos)
                visited.add(next_pos)
            return path

        # Fitness function to evaluate paths
        def fitness_fn(path):
            if path[-1] != self.goal:
                return float('-inf')  # Invalid path (did not reach goal)
            else:
                return -len(path)  # Shorter paths have higher fitness

        # Selection operator: Tournament selection
        def selection(population, fitnesses):
            selected = []
            for _ in range(pop_size):
                i, j = random.sample(range(len(population)), 2)
                selected.append(population[i] if fitnesses[i] > fitnesses[j] else population[j])
            return selected

        # Crossover operator: Combine paths at a common node
        def crossover(parent1, parent2):
            set1 = set(parent1)
            set2 = set(parent2)
            common = list(set1 & set2)
            if len(common) <= 1:
                # No common node other than start; cannot crossover
                return parent1[:]
            common_node = random.choice(common[1:])  # Exclude start node
            idx1 = parent1.index(common_node)
            idx2 = parent2.index(common_node)
            child = parent1[:idx1] + parent2[idx2:]
            # Remove cycles
            seen = set()
            new_child = []
            for pos in child:
                if pos in seen:
                    break
                new_child.append(pos)
                seen.add(pos)
            return new_child

        # Mutation operator: Randomly alter part of the path
        def mutate(path):
            if random.random() < pmut and len(path) > 2:
                idx = random.randint(1, len(path) - 2)  # Exclude start and end
                new_subpath = generate_random_path_from(path[idx - 1])
                return path[:idx] + new_subpath
            return path

        # Generate a random valid path starting from a given position
        def generate_random_path_from(position):
            path = []
            current = position
            visited = set()
            while current != self.goal:
                neighbors = []
                for move in self.Moves:
                    new_pos = (current[0] + move[0], current[1] + move[1])
                    if is_valid(new_pos) and new_pos not in visited:
                        neighbors.append(new_pos)
                if not neighbors:
                    break  # Dead end
                next_pos = random.choice(neighbors)
                path.append(next_pos)
                visited.add(next_pos)
                current = next_pos
            return path

            # Initialize population
        population = [generate_random_path() for _ in range(pop_size)]
        memory_max = pop_size
        all_moves = set()

        # Initialize best_path and best_fitness
        fitnesses = [fitness_fn(individual) for individual in population]
        best_fitness = float('-inf')
        best_path = population[0]  # Initialize with the first individual

        # Find the initial best path
        for i, fitness in enumerate(fitnesses):
            if fitness > best_fitness:
                best_fitness = fitness
                best_path = population[i]

        for generation in range(ngen):
            # Evaluate fitness
            fitnesses = [fitness_fn(individual) for individual in population]

            # Update best path
            for i, fitness in enumerate(fitnesses):
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_path = population[i]
                    if best_path[-1] == self.goal:
                        break  # Found a valid path to goal

            if best_path[-1] == self.goal:
                break  # Terminate if goal is reached

            # Selection
            selected_population = selection(population, fitnesses)

            # Create next generation
            next_population = []
            for i in range(0, pop_size, 2):
                parent1 = selected_population[i]
                parent2 = selected_population[(i + 1) % pop_size]
                # Crossover
                child1 = crossover(parent1, parent2)
                child2 = crossover(parent2, parent1)
                # Mutation
                child1 = mutate(child1)
                child2 = mutate(child2)
                next_population.extend([child1, child2])

            population = next_population[:pop_size]

            # Update all_moves
            for individual in population:
                all_moves.update(individual)

        # Calculate execution time
        time_taken = time.time() - start_time

        # If no valid path found, return failure
        if best_path is None or best_path[-1] != self.goal:
            total_search_cost = sum(self.cost_map[x][y] for x, y in all_moves)
            final_path_cost = float('inf')  # Indicate no valid path found
            return [], 'Genetic Algorithm', list(all_moves), memory_max, total_search_cost, final_path_cost, time_taken

        # Collect all moves from the best path
        all_moves.update(best_path)

        # Calculate total_search_cost and final_path_cost
        total_search_cost = sum(self.cost_map[x][y] for x, y in all_moves if self.cost_map[x][y] != np.inf)
        final_path_cost = sum(self.cost_map[x][y] for x, y in best_path if self.cost_map[x][y] != np.inf)
        memory_max = get_memory_size(all_moves,best_path)

        return best_path, 'Genetic Algorithm', list(all_moves), memory_max, total_search_cost, final_path_cost, time_taken
    


    '''==============================================================================================='''
    '''######################################### Compare algorithms #########################################'''
    '''==============================================================================================='''
 

    def compare_algorithms(self):
        
        # Initialize the DataFrame 
        df = pd.DataFrame(columns=['Algorithm', 'Path Length', 'Time Taken', 'Memory Used', 'Total Cost', 'Final Path Cost'])

        # Directory to store algorithm images
        image_root = "algorithm_images"
        os.makedirs(image_root, exist_ok=True)

        # List to store directory names for zipping
        directory_names = []

        # Helper function to handle algorithm outputs
        def handle_algorithm_output(path, alg_name, moves, memory_max, search_cost, final_path_cost, time_taken):
            # Add results to DataFrame
            df.loc[len(df)] = {
                'Algorithm': alg_name,
                'Path Length': len(path),
                'Time Taken': time_taken,
                'Memory Used': memory_max,
                'Total Cost': search_cost,
                'Final Path Cost': final_path_cost,
            }

            # Create a directory for this algorithm's images
            algorithm_dir = os.path.join(image_root, alg_name)
            os.makedirs(algorithm_dir, exist_ok=True)
            directory_names.append(algorithm_dir)

            # Save the outputted image from visualize_path
            self.visualize_path(path, alg_name)  # Use the provided visualize_path function
            path_image_path = f"{alg_name}_Final_Path.png"
            shutil.move(path_image_path, os.path.join(algorithm_dir, path_image_path))
            # explored image
            explored_image_path = f"{alg_name}_explored_map.png"
            shutil.move(explored_image_path, os.path.join(algorithm_dir, explored_image_path))

        # Process each algorithm manually
        try:
            # BFS
            bfs_path, bfs_name, bfs_moves, bfs_memory_max, bfs_search_cost, bfs_final_path_cost, bfs_time_taken = self.bfs()
            handle_algorithm_output(bfs_path, bfs_name, bfs_moves, bfs_memory_max, bfs_search_cost, bfs_final_path_cost, bfs_time_taken)
        except Exception as e:
            print(f"Error running BFS: {e}")

        try:
            # DFS
            dfs_path, dfs_name, dfs_moves, dfs_memory_max, dfs_search_cost, dfs_final_path_cost, dfs_time_taken = self.dfs()
            handle_algorithm_output(dfs_path, dfs_name, dfs_moves, dfs_memory_max, dfs_search_cost, dfs_final_path_cost, dfs_time_taken)
        except Exception as e:
            print(f"Error running DFS: {e}")

        try:
            # UCS
            ucs_path, ucs_name, ucs_moves, ucs_memory_max, ucs_search_cost, ucs_final_path_cost, ucs_time_taken = self.ucs()
            handle_algorithm_output(ucs_path, ucs_name, ucs_moves, ucs_memory_max, ucs_search_cost, ucs_final_path_cost, ucs_time_taken)
        except Exception as e:
            print(f"Error running UCS: {e}")

        try:
            # IDS
            ids_path, ids_name, ids_moves, ids_memory_max, ids_search_cost, ids_final_path_cost, ids_time_taken = self.iterative_deepening_search()
            handle_algorithm_output(ids_path, ids_name, ids_moves, ids_memory_max, ids_search_cost, ids_final_path_cost, ids_time_taken)
        except Exception as e:
            print(f"Error running IDS: {e}")

        try:
            # Greedy
            greedy_path, greedy_name, greedy_moves, greedy_memory_max, greedy_search_cost, greedy_final_path_cost, greedy_time_taken = self.greedy_algorithm()
            handle_algorithm_output(greedy_path, greedy_name, greedy_moves, greedy_memory_max, greedy_search_cost, greedy_final_path_cost, greedy_time_taken)
        except Exception as e:
            print(f"Error running Greedy: {e}")

        try:
            # A* Search
            Astar_path, Astar_name, Astar_moves, Astar_memory_max, Astar_search_cost, Astar_final_path_cost, Astar_time_taken = self.a_star()
            handle_algorithm_output(Astar_path, Astar_name, Astar_moves, Astar_memory_max, Astar_search_cost, Astar_final_path_cost, Astar_time_taken)
        except Exception as e:
            print(f"Error running A*: {e}")

        try:
            # Hill Climbing
            hill_climbing_path, hill_climbing_name, hill_climbing_moves, hill_climbing_memory_max, hill_climbing_search_cost, hill_climbing_final_path_cost, hill_climbing_time_taken = self.hill_climbing()
            handle_algorithm_output(hill_climbing_path, hill_climbing_name, hill_climbing_moves, hill_climbing_memory_max, hill_climbing_search_cost, hill_climbing_final_path_cost, hill_climbing_time_taken)
        except Exception as e:
            print(f"Error running Hill Climbing: {e}")

        try:
            # Genetic Algorithm
            genetic_path, genetic_name, genetic_moves, genetic_memory_max,  genetic_search_cost, genetic_final_path_cost,genetic_time_taken= self.genetic_algorithm(pop_size=100, ngen=1000, pmut=0.1)
            handle_algorithm_output(genetic_path, genetic_name, genetic_moves, genetic_memory_max, genetic_search_cost, genetic_final_path_cost, genetic_time_taken)
        except Exception as e:
            print(f"Error running Genetic Algorithm: {e}")

        try:
            # Simulated Annealing
            simulated_annealing_path, simulated_annealing_name, simulated_annealing_moves, simulated_annealing_memory_max, simulated_annealing_search_cost, simulated_annealing_final_path_cost, simulated_annealing_time_taken = self.simulated_annealing()
            handle_algorithm_output(simulated_annealing_path, simulated_annealing_name, simulated_annealing_moves, simulated_annealing_memory_max, simulated_annealing_search_cost, simulated_annealing_final_path_cost, simulated_annealing_time_taken)
        except Exception as e:
            print(f"Error running Simulated Annealing: {e}")

        # Create a zip file with all directories
        zip_file_path = "algorithm_results.zip"
        with zipfile.ZipFile(zip_file_path, "w") as zf:
            for directory in directory_names:
                for root, _, files in os.walk(directory):
                    for file in files:
                        zf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), image_root))

        # Cleanup: Remove individual directories after zipping
        shutil.rmtree(image_root)
        # make memory and time colums to 2 decimal places
        df['Memory Used'] = df['Memory Used'].apply(lambda x: f"{x:.2f}")
        df['Time Taken'] = df['Time Taken'].apply(lambda x: f"{x:.5f}")

        # Return the DataFrame and zip file path
        return df, zip_file_path


'''#####################################################################################################'''
'''########################################## Run all algorithms ##########################################'''
'''#####################################################################################################'''


def main():
    # Define grid dimensions, start, and goal positions
    width, height = 25,25
    start = (0,0)
    goal = (24,24)
    algo = algorithm(width, height, start, goal)
    results_df, zip_file_path = algo.compare_algorithms()
    from tabulate import tabulate
    print(tabulate(results_df.values,headers=results_df.columns,tablefmt='pretty'))
    # print(results_df)
    print(f"All images and results have been saved to: {zip_file_path}")



if __name__ == "__main__":
    main()
