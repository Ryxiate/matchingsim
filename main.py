from simulations import *
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import warnings
import json

warnings.filterwarnings("error")

# TODO
# - swap: top trading cycle

if __name__ == "__main__":
    # Load configurations
    with open("./config.json", "r") as f:
        config = json.load(f)

    iter_num = config["SimulationOptions"]["iter_num"]
    room_nums = config["SimulationOptions"]["room_nums"]
    noise_list = config["SimulationOptions"]["noise_list"]
    init_noise = config["SimulationOptions"]["init_noise"]
    solver_list = config["SimulationOptions"]["solver_list"]
    steps = config["SimulationOptions"]["steps"]
    
    rooms_n_rows = config["GraphOptions"]["rooms_n_rows"]
    rooms_n_cols = config["GraphOptions"]["rooms_n_cols"]
    noise_n_rows = config["GraphOptions"]["noise_n_rows"]
    noise_n_cols = config["GraphOptions"]["noise_n_cols"]
    save_not_show = config["GraphOptions"]["save_not_show"]
    dpi = config["GraphOptions"]["dpi"]
    use_title = config["GraphOptions"]["use_title"]
    
    if rooms_n_rows * rooms_n_cols != len(room_nums):
        raise Exception("Subplot arrangement have different size than the room number list")
    if noise_n_rows * noise_n_cols != len(noise_list):
        raise Exception("Subplot arrangement have different size than the noise list")
    
    # Step 1
    # Default simulation
    if 1 in steps:
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
                    
            for k, v in map(lambda k: (k, mean(res[k])), res):
                results[k].append(v)
        
        pbar.close()
        
        plt.figure(figsize=(8, 5))
        # Average utilitarian welfare
        for k in solver_list:
            plt.plot(room_nums, results[k], label=k)
        plt.xlabel("Room Numbers")
        plt.ylabel("Average Utilitarian Welfare")
        plt.legend()
        if use_title:
            plt.title("Average Utilitarian Welfare by Different Algorithms")
        if save_not_show:
            plt.savefig("figure/default.png", dpi=dpi, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
        
        plt.figure(figsize=(16, 9))
        # Individual utilitarian welfare
        s = "serial_dictatorship"
        if s in solver_list:
            for idx, nr in enumerate(room_nums):
                ax = plt.subplot(rooms_n_rows, rooms_n_cols, idx+1)
                ax.plot(range(3*nr), order_res[s][nr])
                plt.title(f"n_rooms = {nr}", fontsize=12)
                plt.xlabel("Room Numbers")
                plt.ylabel("Individual Utilities")
            if use_title:
                plt.suptitle(f"Sequential Individual Utilitarian Welfare by {s}", fontsize=16)
            if save_not_show:
                plt.savefig("figure/serial.png", dpi=dpi, bbox_inches="tight")
                plt.close()
            else:
                plt.show()
        
        # Utility histogram
        for s in solver_list:
            plt.figure(figsize=(16, 9))
            for idx, nr in enumerate(room_nums):
                ax = plt.subplot(rooms_n_rows, rooms_n_cols, idx+1)
                ax.hist(hist_res[s][nr])
                plt.title(f"n_rooms = {nr}", fontsize=12)
                plt.xlabel("Room Numbers")
                plt.ylabel("Individual Utilities")
            if use_title:
                plt.suptitle(f"Individual Utilitarian Welfare Distribution by {s}", fontsize=16)
            if save_not_show:
                plt.savefig(f"figure/{s}_hist.png", dpi=dpi, bbox_inches="tight")
                plt.close()
            else:
                plt.show()
    
    # Step 2
    # Under different noise
    # This is crap and inefficient. Fix this
    if 2 in steps:
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
                        
                for k, v in map(lambda k: (k, mean(res[k])), res):
                    results[noise][k].append(v)
        pbar.close()
        
        plt.figure(figsize=(16, 9))
        for idx, noise in enumerate(noise_list):
            ax = plt.subplot(noise_n_rows, noise_n_cols, idx+1)
            for s in solver_list:
                ax.plot(room_nums, results[noise][s], label=s)
            plt.title(f"noise = {noise}", fontsize=12)
            plt.xlabel("Room Numbers")
            plt.legend()
        if use_title:
            plt.suptitle("Average Utilitarian Welfare under Different Noise", fontsize=16)
        if save_not_show:
            plt.savefig("figure/noise.png", dpi=dpi, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
    
    # Step 3
    # Under different data settings
    if 3 in steps:
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
                    
                for k, v in map(lambda k: (k, mean(res[k])), res):
                    results[i][k].append(v)
        pbar.close()
        
        plt.figure(figsize=(16, 9))
        for idx, name in enumerate(["vanilla", "with one-directional bias", "with indifference", "with bias and indifference"]):
            ax = plt.subplot(2, 2, idx+1)
            for s in solver_list:
                ax.plot(room_nums, results[idx][s], label=s)
            plt.title(name, fontsize=12)
            plt.xlabel("Room Numbers")
            plt.legend()
        if use_title:
            plt.suptitle("Average Utilitarian Welfare under Different Data Scenarios", fontsize=16)
        if save_not_show:
            plt.savefig("figure/scenarios.png", dpi=dpi, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
            
    # Step 4
    # Swapping and top trading graphs