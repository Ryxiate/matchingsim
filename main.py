from simulations import *
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("error")

# TODO
# - swap: top trading cycle

if __name__ == "__main__":
    iter_num = 100
    # room_nums = [2, 5, 10, 15, 20, 30]
    room_nums = [2, 5, 15, 20, 30, 50]
    noise_list = [3, 10, 20]
    init_noise = 3
    (n_rows, n_cols) = (2, 3)
    solver_list = ["serial_dictatorship", "random_serial_dictatorship", "match_by_characteristics", "random_match"]
    
    # Plotting options
    save_not_show = True
    dpi = 300
    
    if n_rows * n_cols != len(room_nums):
        raise Exception("Subplot arrangement have different size than the room number list")
    
    results = {s: [] for s in solver_list}
    order_res = {s: {r: np.zeros((3 * r, )) for r in room_nums} for s in solver_list}
    hist_res = {s: {r: [] for r in room_nums} for s in solver_list}
    pbar = tqdm(total=iter_num*len(room_nums)*len(solver_list), desc="Matching rooms (Step 1/3)")
    for n_rooms in room_nums:
        n_agents = 3 * n_rooms
        res = {s: [] for s in solver_list}
        for iter in range(iter_num):
            inst = geo_3dsr.get_instance(rooms=n_rooms, noise=init_noise, quiet=True, asymmetric_noise=True)

            for s in solver_list:
                solver: solver_base = eval(s)(inst=inst)
                solver.solve()
                ut = solver.evaluate()
                agent_ut = solver.evaluate_agentwise()
                order_res[s][n_rooms] = order_res[s][n_rooms] * iter / (iter + 1) + agent_ut / (iter + 1)       # incremental update to average
                hist_res[s][n_rooms].extend(agent_ut)
                res[s].append(ut)
                
                pbar.update()
                
        for k, v in map(lambda k: (k, sum(res[k])/iter_num), res):
            results[k].append(v/n_agents)
    
    pbar.close()
    
    plt.figure(figsize=(8, 5))
    # Average utilitarian welfare
    for k in solver_list:
        plt.plot(room_nums, results[k], label=k)
    plt.xlabel("Room Numbers")
    plt.ylabel("Average Utilitarian Welfare")
    plt.legend()
    if save_not_show:
        plt.savefig("figure/default.png", dpi=dpi, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
    
    plt.figure(figsize=(16, 9))
    # Individual utilitarian welfare
    s = "serial_dictatorship"
    for idx, nr in enumerate(room_nums):
        ax = plt.subplot(n_rows, n_cols, idx+1)
        ax.plot(range(3*nr), order_res[s][nr])
        plt.title(f"n_rooms = {nr}", fontsize=14)
        plt.xlabel("Room Numbers")
    # plt.suptitle(f"Sequential Individual Utilitarian Welfare under {s}", fontsize=16)
    if save_not_show:
        plt.savefig("figure/serial.png", dpi=dpi, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
    
    # Utility histogram
    for s in solver_list:
        plt.figure(figsize=(16, 9))
        for idx, nr in enumerate(room_nums):
            ax = plt.subplot(n_rows, n_cols, idx+1)
            ax.hist(hist_res[s][nr])
            plt.title(f"n_rooms = {nr}", fontsize=14)
            plt.xlabel("Room Numbers")
        plt.suptitle(f"Individual Utilitarian Welfare Distribution under {s}", fontsize=16)
        if save_not_show:
            plt.savefig(f"figure/{s}_hist.png", dpi=dpi, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
    
    # Under different noise
    # This is crap and inefficient. Fix this
    results = {n: {s: [] for s in solver_list} for n in noise_list}
    pbar = tqdm(total=iter_num*len(noise_list)*len(room_nums)*len(solver_list), desc="Matching rooms under different noise (Step 2/3)")
    for noise in noise_list:
        for n_rooms in room_nums:
            n_agents = 3 * n_rooms
            res = {s: [] for s in solver_list}
            for iter in range(iter_num):
                inst = geo_3dsr.get_instance(rooms=n_rooms, noise=noise, quiet=True, asymmetric_noise=True)

                for s in solver_list:
                    solver: solver_base = eval(s)(inst=inst)
                    solver.solve()
                    ut = solver.evaluate()
                    
                    res[s].append(ut)
                    pbar.update()
                    
            for k, v in map(lambda k: (k, sum(res[k])/iter_num), res):
                results[noise][k].append(v/n_agents)
    pbar.close()
    
    plt.figure(figsize=(16, 9))
    for idx, noise in enumerate(noise_list):
        ax = plt.subplot(1, len(noise_list), idx+1)
        for s in solver_list:
            ax.plot(room_nums, results[noise][s], label=s)
        plt.title(f"noise = {noise}", fontsize=14)
        plt.xlabel("Room Numbers")
    # plt.suptitle("Average Utilitarian Welfare under Different Noise", fontsize=16)
    plt.legend()
    if save_not_show:
        plt.savefig("figure/noise.png", dpi=dpi, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
    
    # Under different data settings
    results = {i: {s: [] for s in solver_list} for i in range(4)}
    pbar = tqdm(total=4*iter_num*len(room_nums)*len(solver_list), desc="Matching rooms under different scenarios (Step 3/3)")
    for i in range(4):
        for n_rooms in room_nums:
            n_agents = 3 * n_rooms
            res = {s: [] for s in solver_list}
            for iter in range(iter_num):
                inst = geo_3dsr.get_instance(rooms=n_rooms, noise=init_noise, quiet=True, asymmetric_noise=True)
                if i % 2:
                    inst.factor_bias()
                if i // 2:
                    inst.utilities_dropout()

                for s in solver_list:
                    solver: solver_base = eval(s)(inst=inst)
                    solver.solve()
                    ut = solver.evaluate()

                    res[s].append(ut)
                    
                    pbar.update()
                
            for k, v in map(lambda k: (k, sum(res[k])/iter_num), res):
                results[i][k].append(v/n_agents)
    pbar.close()
    
    plt.figure(figsize=(16, 9))
    for idx, name in enumerate(["vanilla", "with one-directional bias", "with indifference", "with bias and indifference"]):
        ax = plt.subplot(2, 2, idx+1)
        for s in solver_list:
            ax.plot(room_nums, results[idx][s], label=s)
        plt.title(name, fontsize=14)
        plt.xlabel("Room Numbers")
    # plt.suptitle("Average Utilitarian Welfare under Different Data Scenarios", fontsize=16)
    plt.legend()
    if save_not_show:
        plt.savefig("figure/scenarios.png", dpi=dpi, bbox_inches="tight")
        plt.close()
    else:
        plt.show()