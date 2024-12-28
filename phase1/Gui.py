from libraries import *
from algorthims import *

'''========================================== Algorithm Selection MAIN Window ===================================='''

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

'''========================================== Loading Window ===================================='''

class LoadingScreenGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Loading")
        self.loading_label = tk.Label(self.root, text="Loading, please wait...", font=("Arial", 18))
        self.loading_label.pack(pady=20)

'''==========================================Result of Executed Algorithm Window ===================================='''

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
            frame = frame.resize((600, 600), Image.LANCZOS) # LANCZOS -> maintain image quality when DownSampling
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


'''==============================================================================================='''
'''######################################### Excute the chosen algorithm #########################################'''
'''==============================================================================================='''


def execute_algorithm(algorithmName, width=25, height=25, start=(0, 0), goal=(24, 24)):
    print(f"Starting {algorithmName} algorithm...")
    start_time = time.time()
    grid = algorithm(width, height, start, goal)

    if algorithmName == "BFS":
        path, AlGname, all_moves,memory_max, total_search_cost, final_path_cost, working_time,= grid.bfs()

    elif algorithmName == "DFS":
        path, AlGname, all_moves, memory_max,_,_,_ = grid.dfs()

    elif algorithmName == "IDS":
        path, AlGname, all_moves, memory_max,_,_,_ = grid.iterative_deepening_search()

    elif algorithmName == "UCS":  
        path, AlGname, all_moves, memory_max,_,_,_ = grid.ucs()

    elif algorithmName == "Greedy Best First Search":
        path, AlGname, all_moves, memory_max,_,_,_ = grid.greedy_algorithm()

    elif algorithmName == "A*":
        path, AlGname, all_moves, memory_max,_,_,_ = grid.a_star()

    elif algorithmName == "Hill Climbing":
        path, AlGname, all_moves, memory_max,_,_,_ = grid.hill_climbing()

    elif algorithmName == "Simulated Annealing":
        path, AlGname, all_moves, memory_max,_,_,_ = grid.simulated_annealing(max_iterations=5000)

    elif algorithmName == "Genetic Algorithm":
        path, AlGname, all_moves, memory_max,_,_,_ = grid.genetic_algorithm()

    else:
        raise ValueError("Unknown algorithm name")
    
    grid.images_frames(AlGname, path, all_moves)
    grid.videoFrom_images(AlGname)
    grid.videoFrom_movements(AlGname)
    grid.visualize_path(path, AlGname)
    end_time = time.time()
    run_time = end_time - start_time
    print(f"{algorithmName} algorithm completed in {run_time:.2f} seconds.")
    return {
        'exploration_video': f"{AlGname}_exploration.mp4",
        'path_video': f"{AlGname}_path_formation.mp4",
        'final_image': f"{AlGname}_Final_Path.png",
        'explored_nodes_image': f"{AlGname}_explored_map.png"
    }, run_time, memory_max

