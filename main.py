from simulations import *
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# TODO
# a) histogram and sequential graph of utility of agents
# b) brute-force upper bound
# c) variaty in data: indifference, cheating (biased)
# d) swap. Top trade cycle
# e) Efficiency: Pareto-optimality

if __name__ == "__main__":
    iter_num = 100
    room_nums = [2, 5, 7, 10, 12, 15]
    noise_list = [2, 5, 20]
    (n_rows, n_cols) = (2, 3)
    solver_list = ["serial_dictatorship", "match_by_characteristics", "random_match"]
    
    if n_rows * n_cols != len(room_nums):
        raise Exception("Subplot arrangement have different size than the room number list")
    
    results = {s: [] for s in solver_list}
    order_res = {s: {r: np.zeros((3 * r, )) for r in room_nums} for s in solver_list}
    hist_res = {s: {r: [] for r in room_nums} for s in solver_list}
    pbar = tqdm(total=iter_num*len(room_nums)*len(solver_list), desc="Matching rooms")
    for n_rooms in room_nums:
        n_agents = 3 * n_rooms
        res = {s: [] for s in solver_list}
        for iter in range(iter_num):
            inst = geo_3dsr.get_instance(rooms=n_rooms, noise=5, quiet=True, asymmetric_noise=True)

            for s in solver_list:
                solver: solver_base = eval(s)(inst=inst)
                solver.solve()
                ut = solver.evaluate()
                agent_ut = solver.evaluate_agentwise()
                order_res[s][n_rooms] = order_res[s][n_rooms] * iter / (iter + 1) + agent_ut / (iter + 1)
                hist_res[s][n_rooms].extend(agent_ut)
                res[s].append(ut)
                
                pbar.update()
                
        for k, v in map(lambda k: (k, sum(res[k])/iter_num), res):
            results[k].append(v/n_agents)
    
    pbar.close()
    
    # Average utilitarian welfare
    for k in solver_list:
        plt.plot(room_nums, results[k], label=k)
    plt.xlabel("Room Numbers")
    plt.ylabel("Average Utilitarian Welfare")
    plt.legend()
    plt.show()
    
    # Individual utilitarian welfare
    s = "serial_dictatorship"
    for idx, nr in enumerate(room_nums):
        ax = plt.subplot(n_rows, n_cols, idx+1)
        ax.plot(range(3*nr), order_res[s][nr])
        plt.title(f"n_rooms = {nr}")
    plt.suptitle(f"Squential Individual Utilitarian Welfare under {s}")
    plt.show()
    
    # Utility histogram
    for s in solver_list:
        for idx, nr in enumerate(room_nums):
            ax = plt.subplot(n_rows, n_cols, idx+1)
            ax.hist(hist_res[s][nr])
            plt.title(f"n_rooms = {nr}")
        plt.suptitle(f"Individual Utilitarian Welfare Distribution under {s}")
        plt.show()