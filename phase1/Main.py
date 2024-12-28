from Node_Grid import *
from Algorthims import *
from Gui import *
from Libraries import *

if __name__ == "__main__":
    algorithms = ["BFS", "DFS", "UCS", "IDS", "Greedy Best First Search", "A*", "Hill Climbing", "Simulated Annealing", "Genetic Algorithm"]
    analytics_data = {}  # Dictionary to store analytics data for all algorithms

    while True:
        # Algorithm selection GUI
        root_selection = tk.Tk()
        algorithm_selection_gui = AlgorithmSelectionGUI(root_selection, algorithms)
        root_selection.mainloop()
        selected_algorithm = algorithm_selection_gui.selected_algorithm
        root_selection.destroy()

        if selected_algorithm is None:
            break  # Exit if no algorithm was selected

        # Loading screen
        root_loading = tk.Tk()
        loading_gui = LoadingScreenGUI(root_loading)
        root_loading.update()

        # Execute the algorithm
        outputs, run_time, memory_max = execute_algorithm(selected_algorithm)
        root_loading.destroy()  # Close the loading screen

        # append the analytics for the excuted algorithm 
        analytics_data[selected_algorithm] = {
            'run_time': run_time,
            'memory_max': memory_max
        }

        # Result GUI 
        root_result = tk.Tk()
        result_gui = ResultGUI(root_result, outputs, selected_algorithm, run_time, memory_max, analytics_data)
        root_result.mainloop()
        root_result.destroy()

        if not result_gui.try_again:
            break  # Exit if the user does not want to try another algorithm
