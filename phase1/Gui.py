from libraries import *
from algorthims import *

# Define a constant for the window size
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 370

'''========================================== Algorithm Selection MAIN Window ===================================='''

class AlgorithmSelectionGUI:
    def __init__(self, root, algorithms):
        self.root = root
        self.algorithms = algorithms

        self.root.geometry("800x500")
        self.root.title("Select Algorithm ")

        self.bg_image = Image.open("background.jpeg").resize((800, 500), Image.LANCZOS)
        self.bg_photo = ImageTk.PhotoImage(self.bg_image)

        self.canvas = tk.Canvas(self.root, width=800, height=500, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.create_image(0, 0, image=self.bg_photo, anchor="nw")

        self.title_label = tk.Label(self.root, text="Select Algorithm", font=("Arial", 22, "bold"), 
                                    fg="white", bg="#000050")  # Semi-transparent black
        self.title_label.place(relx=0.1, rely=0.1, anchor="w")  # Move title to the left

        self.algorithm_var = tk.StringVar(value=self.algorithms[0])
        self.buttons = []
        for i, algo in enumerate(self.algorithms):
            rb = tk.Radiobutton(self.root, text=algo, variable=self.algorithm_var, value=algo, 
                                font=("Arial", 12, "bold"),
                                fg="white", bg="#1E2A38",  # Dark blue background
                                selectcolor="#0099FF",  # Neon blue when selected
                                activebackground="#00CCFF", activeforeground="white",
                                indicatoron=0, width=25, height=2, relief="flat", cursor="hand2",
                                command=self.update_selection)
            rb.place(relx=0.1, rely=0.2 + i * 0.08, anchor="w")  # Move buttons to the left
            self.buttons.append(rb)

        self.start_button = tk.Button(self.root, text="Start Algorithm", command=self.start_algorithm,
                                      font=("Arial", 12, "bold"), bg="#0099FF", fg="white", 
                                      activebackground="#00CCFF", relief="flat", bd=2, 
                                      padx=10, pady=5, cursor="hand2")
        self.start_button.place(relx=0.1, rely=0.95, anchor="w")  # Move start button to the left

        self.bg_photo_ref = self.bg_photo  

    def update_selection(self):
        """Update button colors based on selection"""
        for button in self.buttons:
            if button["value"] == self.algorithm_var.get():
                button.configure(bg="#00CCFF", fg="black")  
            else:
                button.configure(bg="#1E2A38", fg="white")   

    def start_algorithm(self):
        self.selected_algorithm = self.algorithm_var.get()
        self.root.quit()

'''========================================== Loading Window ===================================='''

class LoadingScreenGUI:
    def __init__(self, root):
        self.root = root
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.title("Loading")
        self.root.configure(bg="black")
        
        self.loading_label = tk.Label(self.root, text="Loading, please wait...", font=("Arial", 18), fg="white", bg="black")
        self.loading_label.pack(pady=20)
        

'''==========================================Result of Executed Algorithm Window ===================================='''

class ResultGUI:
    def __init__(self, root, outputs, algorithm_name, run_time, memory_max, analytics_data):
        self.root = root
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.title("Algorithm Visualization")
        self.root.configure(bg="#2C3E50")
        
        self.outputs = outputs
        self.algorithm_name = algorithm_name
        self.run_time = run_time
        self.memory_max = memory_max
        self.analytics_data = analytics_data
        self.try_again = False
        
        self.result_page = tk.Frame(self.root, bg="#2C3E50")
        self.analytics_page = tk.Frame(self.root, bg="#34495E")
        
        self.setup_result_page()
        self.setup_analytics_page()
        
        self.show_result_page()
    
    def setup_result_page(self):
        self.video_label = tk.Label(self.result_page, bg="#2C3E50")
        self.video_label.pack()

        self.control_frame = tk.Frame(self.result_page, bg="#2C3E50")
        self.control_frame.pack()

        self.play_exploration_button = ttk.Button(self.control_frame, text="Play Exploration Video", command=self.play_exploration_video)
        self.play_exploration_button.pack(side=tk.LEFT, padx=5)

        self.play_path_button = ttk.Button(self.control_frame, text="Play Path Video", command=self.play_path_video)
        self.play_path_button.pack(side=tk.LEFT, padx=5)

        self.view_final_path_button = ttk.Button(self.control_frame, text="View Final Path Image", command=self.view_final_path_image)
        self.view_final_path_button.pack(side=tk.LEFT, padx=5)

        self.view_explored_nodes_button = ttk.Button(self.control_frame, text="View Explored Nodes Image", command=self.view_explored_nodes_image)
        self.view_explored_nodes_button.pack(side=tk.LEFT, padx=5)

        self.try_another_button = ttk.Button(self.control_frame, text="Try Another Algorithm", command=self.try_another_algorithm)
        self.try_another_button.pack(side=tk.LEFT, padx=5)

        self.analytics_button = ttk.Button(self.control_frame, text="Analytics", command=self.show_analytics_page)
        self.analytics_button.pack(side=tk.LEFT, padx=5)
    
    def setup_analytics_page(self):
        self.analytics_label = tk.Label(self.analytics_page, text="Analytics", font=("Arial", 18), fg="white", bg="#34495E")
        self.analytics_label.pack(pady=20)
        
        self.analytics_text = tk.Text(self.analytics_page, width=80, height=25, bg="#ECF0F1")
        self.analytics_text.pack()
        
        self.back_button = ttk.Button(self.analytics_page, text="Back", command=self.show_result_page)
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
    
    def view_final_path_image(self):
        self.display_image(self.final_path_image_path)
    
    def view_explored_nodes_image(self):
        self.display_image(self.explored_nodes_image_path)
    
    def display_image(self, image_path):
        image = Image.open(image_path)
        image = image.resize((420, 320), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(image)
        self.video_label.config(image=self.photo)
        self.video_label.image = self.photo
    
    def update_video_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (420, 320))
            frame = Image.fromarray(frame)
            self.photo = ImageTk.PhotoImage(frame)
            self.video_label.config(image=self.photo)
            self.video_label.image = self.photo
            self.root.after(33, self.update_video_frame)
        else:
            self.cap.release()
            self.cap = None
    
    def show_analytics_page(self):
        self.result_page.pack_forget()
        self.analytics_page.pack(fill="both", expand=True)
        
        self.analytics_text.delete('1.0', tk.END)
        self.analytics_text.insert(tk.END, "Algorithms Tried and Execution Times:\n")
        for algo, data in self.analytics_data.items():
            self.analytics_text.insert(tk.END, f"{algo}:\n")
            self.analytics_text.insert(tk.END, f"    Execution Time: {data['run_time']:.2f} seconds\n")
            self.analytics_text.insert(tk.END, f"    Max Data Structure Size: {data['memory_max']} nodes\n")
    
    def try_another_algorithm(self):
        self.try_again = True
        self.root.quit()


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
        'exploration_video': os.path.join(grid.output_dir, f"{AlGname}_exploration.mp4"),
        'path_video': os.path.join(grid.output_dir, f"{AlGname}_path_formation.mp4"),
        'final_image': os.path.join(grid.output_dir, f"{AlGname}_Final_Path.png"),
        'explored_nodes_image': os.path.join(grid.output_dir, f"{AlGname}_explored_map.png")
    }, run_time, memory_max
