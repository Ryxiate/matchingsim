from simulations import *
from tqdm import tqdm
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import warnings
import json

warnings.filterwarnings("error")

# TODO
# - swap: top trading cycle

if __name__ == "__main__":
    steps_num = 4
    # Load configurations
    with open("./config.json", "r") as f:
        config = json.load(f)

    steps = config["SimulationOptions"]["steps"]
    iter_num = config["SimulationOptions"]["iter_num"]
    room_nums = config["SimulationOptions"]["room_nums"]
    noise_list = config["SimulationOptions"]["noise_list"]
    init_noise = config["SimulationOptions"]["init_noise"]
    solver_list = check_solver_validity(config["SimulationOptions"]["solver_list"])
    eval_techniques = check_evaluator_validity(config["SimulationOptions"]["eval_list"])
    
    rooms_n_rows = config["GraphOptions"]["rooms_n_rows"]
    rooms_n_cols = config["GraphOptions"]["rooms_n_cols"]
    noise_n_rows = config["GraphOptions"]["noise_n_rows"]
    noise_n_cols = config["GraphOptions"]["noise_n_cols"]
    save_not_show = config["GraphOptions"]["save_not_show"]
    dpi = config["GraphOptions"]["dpi"]
    use_title = config["GraphOptions"]["use_title"]
    small_figure_ratio = tuple(config["GraphOptions"]["small_figure_ratio"].values())
    large_figure_ratio = tuple(config["GraphOptions"]["large_figure_ratio"].values())
    
    if rooms_n_rows * rooms_n_cols != len(room_nums):
        raise Exception("Subplot arrangements have different size than the room number list")
    if noise_n_rows * noise_n_cols != len(noise_list):
        raise Exception("Subplot arrangements have different size than the noise list")
    
    # Step 1
    # Default simulation
    step = 1
    if step in steps:
        results = {s: [] for s in solver_list}
        order_res = {s: {r: np.zeros((3 * r, )) for r in room_nums} for s in solver_list}
        hist_res = {s: {r: [] for r in room_nums} for s in solver_list}
        pbar = tqdm(total=iter_num*len(room_nums)*len(solver_list), desc=f"Matching rooms (Step {step}/{steps_num})")
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
        
        plt.figure(figsize=small_figure_ratio)
        # Average utilitarian welfare
        for k in solver_list:
            plt.plot(room_nums, results[k], label=k)
        plt.xlabel("Room Numbers")
        plt.ylabel("Average Utilitarian Welfare")
        plt.legend()
        if use_title:
            plt.title("Average Utilitarian Welfare by Different Algorithms")
        plt.tight_layout(pad=0.2)
        if save_not_show:
            plt.savefig("figure/default.png", dpi=dpi, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
        
        plt.figure(figsize=large_figure_ratio)
        # Individual utilitarian welfare
        ss = ["serial_dictatorship", "SD_by_rooms"]
        ss_valid = [s for s in ss if s in solver_list]
        ss_cnt = len(ss_valid)
        for s in ss_valid:
            for idx, nr in enumerate(room_nums):
                ax = plt.subplot(rooms_n_rows, rooms_n_cols, idx+1)
                ax.plot(range(3*nr), order_res[s][nr], label=s)
                plt.title(f"n_rooms = {nr}", fontsize=12)
                plt.xlabel("Room Numbers")
                plt.ylabel("Individual Utilities")
                if ss_cnt > 1:
                    plt.legend()
        
        if ss_cnt:
            if use_title:
                plt.suptitle(f"Sequential Individual Utilitarian Welfare by {s}", fontsize=16)
            plt.tight_layout(pad=0.2)
            if save_not_show:
                plt.savefig("figure/serial.png", dpi=dpi, bbox_inches="tight")
                plt.close()
            else:
                plt.show()
        
        # Utility histogram
        for s in solver_list:
            plt.figure(figsize=large_figure_ratio)
            for idx, nr in enumerate(room_nums):
                ax = plt.subplot(rooms_n_rows, rooms_n_cols, idx+1)
                ax.hist(hist_res[s][nr])
                plt.title(f"n_rooms = {nr}", fontsize=12)
                plt.xlabel("Room Numbers")
                plt.ylabel("Individual Utilities")
            if use_title:
                plt.suptitle(f"Individual Utilitarian Welfare Distribution by {s}", fontsize=16)
            plt.tight_layout(pad=0.2)
            if save_not_show:
                plt.savefig(f"figure/{s}_hist.png", dpi=dpi, bbox_inches="tight")
                plt.close()
            else:
                plt.show()
    
    # Step 2
    # Under different noise
    # This is crap and inefficient. Fix this
    step = 2
    if step in steps:
        results = {n: {s: [] for s in solver_list} for n in noise_list}
        pbar = tqdm(total=iter_num*len(noise_list)*len(room_nums)*len(solver_list), desc=f"Matching rooms under different noise (Step {step}/{steps_num})")
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
        
        plt.figure(figsize=large_figure_ratio)
        for idx, noise in enumerate(noise_list):
            ax = plt.subplot(noise_n_rows, noise_n_cols, idx+1)
            for s in solver_list:
                ax.plot(room_nums, results[noise][s], label=s)
            plt.title(f"noise = {noise}", fontsize=12)
            plt.xlabel("Room Numbers")
            plt.legend()
        if use_title:
            plt.suptitle("Average Utilitarian Welfare under Different Noise", fontsize=16)
        plt.tight_layout(pad=0.2)
        if save_not_show:
            plt.savefig("figure/noise.png", dpi=dpi, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
    
    # Step 3
    # Under different data settings
    step = 3
    if step in steps:
        results = {i: {s: [] for s in solver_list} for i in range(4)}
        pbar = tqdm(total=4*iter_num*len(room_nums)*len(solver_list), desc=f"Matching rooms under different scenarios (Step {step}/{steps_num})")
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
        
        plt.figure(figsize=large_figure_ratio)
        for idx, name in enumerate(["vanilla", "with one-directional bias", "with indifference", "with bias and indifference"]):
            ax = plt.subplot(2, 2, idx+1)
            for s in solver_list:
                ax.plot(room_nums, results[idx][s], label=s)
            plt.title(name, fontsize=12)
            plt.xlabel("Room Numbers")
            plt.legend()
        if use_title:
            plt.suptitle("Average Utilitarian Welfare under Different Data Scenarios", fontsize=16)
        plt.tight_layout(pad=0.2)
        if save_not_show:
            plt.savefig("figure/scenarios.png", dpi=dpi, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
            
    # Step 4
    # Swapping and top trading cycles
    step = 4
    if step in steps and len(eval_techniques):
        results = {t: {s: [] for s in solver_list} for t in eval_techniques}
        pbar = tqdm(total=len(eval_techniques)*iter_num*len(room_nums)*len(solver_list), 
                    desc=f"Evaluating solver performance with chosen techniques (Step {step}/{steps_num})")
        for n_rooms in room_nums:
            n_agents = 3 * n_rooms
            res = {t: {s: [] for s in solver_list} for t in eval_techniques}
            for iter in range(iter_num):
                inst = geo_3dsr.get_instance(rooms=n_rooms, noise=init_noise, quiet=True, asymmetric_noise=True)

                for s in solver_list:
                    solver: solver_base = eval(s)(inst=inst)
                    solver.solve()
                    origin_ut = solver.evaluate()
                    
                    for t in eval_techniques:
                        evaluator = eval(t).from_solver(solver=solver)
                        res[t][s].append(evaluator.evaluate())
                        pbar.update()
                        
            for (t, s) in product(eval_techniques, solver_list):
                results[t][s].append(mean(res[t][s]))
        pbar.close()
        
        eval_title_corr = {"preference_swapper": ("Proportional Utilitarian Welfare Improvements After Swapping", "Proportional Utility Improvement", "swap"),
                      "ttc_evaluator": ("Minimum Alpha for No Top Trading Cycles", "Minimum Alpha", "ttc")}
        for (t, (title, y_title, tn)) in eval_title_corr.items():
            plt.figure(figsize=small_figure_ratio)
            for s in solver_list:
                plt.plot(room_nums, results[t][s], label=s)
            plt.xlabel("Room Numbers")
            plt.ylabel(y_title)
            plt.legend()
            if use_title:
                plt.title(title, fontsize=16)
            plt.tight_layout(pad=0.2)
            if save_not_show:
                plt.savefig(f"figure/{tn}.png", dpi=dpi, bbox_inches="tight")
                plt.close()
            else:
                plt.show()