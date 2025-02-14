from libraries import *
from algorthims import *

# Adjust window and component sizes for better proportions
WINDOW_WIDTH = 650  # Slightly wider for better button layout
WINDOW_HEIGHT = 500
VIDEO_WIDTH = 380
VIDEO_HEIGHT = 320
BUTTON_WIDTH = 13  # Slightly wider for text
BUTTON_HEIGHT = 1

'''========================================== Algorithm Selection MAIN Window ===================================='''

class AlgorithmSelectionGUI:
    def __init__(self, root, algorithms):
        self.root = root
        self.algorithms = algorithms
        self.selected_algorithm = None

        # Configure window
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.resizable(False, False)
        self.root.title("Select Algorithm")

        # Background setup
        self.bg_image = Image.open(r"phase1\static\background.jpeg").resize((WINDOW_WIDTH, WINDOW_HEIGHT), Image.LANCZOS)
        self.bg_photo = ImageTk.PhotoImage(self.bg_image)

        self.canvas = tk.Canvas(self.root, width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.create_image(0, 0, image=self.bg_photo, anchor="nw")

        # Title with left alignment
        self.title_label = tk.Label(self.root, text="Select Algorithm", 
                                  font=("Arial", 18, "bold"),
                                  fg="white", bg="#000050")
        self.title_label.place(relx=0.1, rely=0.1, anchor="w")

        # Modify radio buttons to look like regular buttons
        self.algorithm_var = tk.StringVar(value=self.algorithms[0])
        self.buttons = []
        for i, algo in enumerate(self.algorithms):
            btn = tk.Button(self.root, 
                          text=algo,
                          font=("Arial", 11),
                          fg="white", 
                          bg="#1E2A38",
                          activebackground="#00CCFF",
                          width=25,
                          height=1,
                          command=lambda x=algo: self.select_algorithm(x))
            btn.place(relx=0.1, rely=0.2 + i * 0.07, anchor="w")
            self.buttons.append(btn)

        # Start button at bottom left
        self.start_button = tk.Button(self.root, text="Start Algorithm",
                                    command=self.start_algorithm,
                                    font=("Arial", 11),
                                    bg="#0099FF", fg="white",
                                    width=15, height=1)
        self.start_button.place(relx=0.1, rely=0.9, anchor="w")

    def select_algorithm(self, selected):
        self.algorithm_var.set(selected)
        for button in self.buttons:
            if button['text'] == selected:
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
        self.root.resizable(False, False)
        self.root.configure(bg="#000050")
        
        self.loading_label = tk.Label(self.root, text="Loading, please wait...",
                                    font=("Arial", 16), fg="white", bg="#000050")
        self.loading_label.place(relx=0.5, rely=0.5, anchor="center")

'''==========================================Result of Executed Algorithm Window ===================================='''

class ResultGUI:
    def __init__(self, root, outputs, algorithm_name, run_time, memory_max, analytics_data):
        self.root = root
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.resizable(False, False)
        self.root.title("Algorithm Visualization")
        self.root.configure(bg="#2C3E50")
        
        # Store paths with proper directory
        self.outputs = {
            'exploration_video': os.path.join('algorithm_outputs', outputs['exploration_video']),
            'path_video': os.path.join('algorithm_outputs', outputs['path_video']),
            'final_image': os.path.join('algorithm_outputs', outputs['final_image']),
            'explored_nodes_image': os.path.join('algorithm_outputs', outputs['explored_nodes_image'])
        }
        self.algorithm_name = algorithm_name
        self.run_time = run_time
        self.memory_max = memory_max
        self.analytics_data = analytics_data
        self.try_again = False  # Flag to indicate if user wants to try another algorithm

        # Create pages
        self.result_page = tk.Frame(self.root, width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
        self.analytics_page = tk.Frame(self.root, width=WINDOW_WIDTH, height=WINDOW_HEIGHT)

        # Set up pages
        self.setup_result_page()
        self.setup_analytics_page()

        self.show_result_page()

    def setup_result_page(self):
        # Main container with padding
        main_container = tk.Frame(self.result_page, bg="#2C3E50")
        main_container.pack(expand=True, fill="both", padx=20, pady=10)
        
        # Algorithm name label at top
        algo_label = tk.Label(main_container, 
                            text=self.algorithm_name,
                            font=("Arial", 14, "bold"),
                            fg="white", bg="#2C3E50")
        algo_label.pack(pady=(0, 10))
        
        # Video display frame
        video_frame = tk.Frame(main_container, bg="#2C3E50")
        video_frame.pack(pady=10)
        
        self.video_label = tk.Label(video_frame, 
                                  bg="#1E2A38",
                                  width=VIDEO_WIDTH,
                                  height=VIDEO_HEIGHT)
        self.video_label.pack()

        # Custom button creation function
        def create_result_button(parent, text, command):
            return tk.Button(parent,
                             text=text,
                             command=command,
                             font=("Arial", 9),
                             fg="white",
                             bg="#1E2A38",
                             activebackground="#00CCFF",
                             activeforeground="white",
                             bd=0,           # Remove border
                             relief="flat",
                             padx=8,
                             pady=1,        # Less vertical padding
                             width=BUTTON_WIDTH,
                             height=1,
                             cursor="hand2")

        # Button frames with spacing
        button_frame_top = tk.Frame(main_container, bg="#2C3E50")
        button_frame_top.pack(pady=3)
        button_frame_bottom = tk.Frame(main_container, bg="#2C3E50")
        button_frame_bottom.pack(pady=3)

        # Top row buttons
        buttons_top = [
            ("👁 Exploration", self.play_exploration_video),
            ("▶ Path Video", self.play_path_video),
            ("🎯 Final Path", self.view_final_path_image)
        ]

        # Bottom row buttons
        buttons_bottom = [
            ("🔍 Explored", self.view_explored_nodes_image),
            ("📊 Stats", self.show_analytics_page),
            ("↺ Try Again", self.try_another_algorithm)
        ]

        # Create buttons with consistent spacing
        for text, command in buttons_top:
            btn = create_result_button(button_frame_top, text, command)
            btn.pack(side=tk.LEFT, padx=4)

        for text, command in buttons_bottom:
            btn = create_result_button(button_frame_bottom, text, command)
            btn.pack(side=tk.LEFT, padx=4)

    def setup_analytics_page(self):
        # Main analytics container with better styling
        analytics_container = tk.Frame(self.analytics_page, bg="#34495E")
        analytics_container.pack(expand=True, fill="both", padx=30, pady=20)
        
        # Header with stats
        header_frame = tk.Frame(analytics_container, bg="#34495E")
        header_frame.pack(fill="x", pady=(0, 15))
        
        tk.Label(header_frame,
                text=f"Analytics Dashboard",
                font=("Arial", 16, "bold"),
                fg="white", bg="#34495E").pack()
        
        # Current algorithm stats
        stats_frame = tk.Frame(analytics_container, bg="#34495E")
        stats_frame.pack(fill="x", pady=(0, 10))
        
        current_stats = tk.Label(stats_frame,
                               text=f"Current Algorithm: {self.algorithm_name}\n"
                                    f"Runtime: {self.run_time:.2f}s\n"
                                    f"Memory Usage: {self.memory_max} nodes",
                               font=("Arial", 11),
                               fg="white", bg="#34495E",
                               justify=tk.LEFT)
        current_stats.pack(anchor="w")
        
        # Scrollable results area with better formatting
        text_frame = tk.Frame(analytics_container)
        text_frame.pack(expand=True, fill="both", pady=10)
        
        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.analytics_text = tk.Text(text_frame,
                                    width=45,
                                    height=12,
                                    yscrollcommand=scrollbar.set,
                                    font=("Consolas", 11),
                                    bg="#ECF0F1",
                                    padx=10,
                                    pady=10)
        self.analytics_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.analytics_text.yview)
        
        # Back button with styling
        self.back_button = tk.Button(analytics_container,
                                   text="← Back",
                                   command=self.show_result_page,
                                   font=("Arial", 9),
                                   fg="white",
                                   bg="#1E2A38",
                                   activebackground="#00CCFF",
                                   activeforeground="white",
                                   bd=0,
                                   relief="flat",
                                   padx=8,
                                   pady=1,      # Less vertical padding
                                   width=BUTTON_WIDTH,
                                   height=1,
                                   cursor="hand2")
        self.back_button.pack(pady=15)

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
        try:
            image = Image.open(image_path)
            image = image.resize((VIDEO_WIDTH, VIDEO_HEIGHT), Image.LANCZOS)
            self.photo = ImageTk.PhotoImage(image)
            self.video_label.config(image=self.photo)
            self.video_label.image = self.photo
        except Exception as e:
            print(f"Error loading image: {e}")
            self.video_label.config(text="Error loading image")

    def update_video_frame(self):
        try:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
                frame = Image.fromarray(frame)
                self.photo = ImageTk.PhotoImage(frame)
                self.video_label.config(image=self.photo)
                self.video_label.image = self.photo
                self.root.after(33, self.update_video_frame)
            else:
                self.cap.release()
                self.cap = None
        except Exception as e:
            print(f"Error updating video frame: {e}")
            self.video_label.config(text="Error playing video")

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
