from .solvers import solver_base
from .utils import *

from functools import reduce
import numpy as np


class eval_base(object):
    def __init__(self, solution: dict[int, set[int]], utilities: np.ndarray):
        if not equal(utilities.shape):
            raise Exception("Utilities should be a square matrix.")
        if not utilities.shape[0] == len(solution) * 3:
            raise Exception("Length of utilities and solution does not match.")
        self._solution = solution.copy()
        self._utilities = utilities.copy()
        
        self.n_rooms = len(solution)
        self.n_agents = self._utilities.shape[0]
        
    @classmethod
    def from_solver(cls, solver: solver_base):
        return cls(solution=solver.solution, utilities=solver.utilities)
        
    def _calculate_utilities(self, solution: dict = None, return_mean: bool = True):
        if solution is None:
            solution = self._solution
            
        if any([len(room) != 3 for room in solution.values()]):
            raise Exception("Solution empty or incomplete.")
        room_utilities = {}
        for r in solution.keys():
            (i, j, k) = solution[r]
            room_utilities[r] = self._utilities[[i, j, k]][:, [i, j, k]].sum()
            
        if return_mean:
            return mean(room_utilities.values())
        return room_utilities

    @property
    def solution(self):
        return self._solution.copy()
        
    def evaluate(self, verbose: int = 0):
        raise NotImplementedError(f"Evaluate method not supported for class {type(self).__name__}")
    

class preference_swapper(eval_base):
    '''Greedily swap the agents based on their preferences to one another.
    
    Each step, will choose the swapping scheme that gives greatest improvements on total utilitarian welfare.
    If all swapping schemes are inferior to the original solution, then no swap will be carried out.
    '''
    def __init__(self, solution: dict[int, set[int]], utilities: np.ndarray, preferences: set[tuple]):
        super().__init__(solution=solution, utilities=utilities)
        self.preferences = preferences.copy()
        
    @classmethod
    def from_solver(cls, solver: solver_base):
        return cls(solution=solver.solution, utilities=solver.utilities, preferences=solver.preferences)
        
    def _swap(self, inplace: bool = True, verbose: int = 0) -> dict[int, set]:
        # Really inefficient implementation but I can't find any other way. Help!
        swap_cnt = 0
        new_solution = self.solution
        for agents in self.preferences:
            # Find the rooms these two agents are in
            room_assign: dict[int, tuple[int, set[int], bool]] = {a: (room_num, room - {a}, False if tuple(room - {a}) in self.preferences else True) 
                                                                for (room_num, room) in self.solution.items() for a in agents if a in room}
            room_nums = list(map(lambda x: x[0], room_assign.values()))
            if len(agents) == 2:
                # For a 2-person preference group
                if not equal(map(lambda x: x[0], room_assign.values())):
                    # if the two are not assigned to the same room
                    swap_options = [{rn: {rm} | set(agents), other(room_nums, rn): 
                                    {other(roommates, rm)} | room_assign[other(agents, a)][1]} 
                                    for (a, (rn, roommates, status)) in room_assign.items() if status for rm in roommates]
                    if len(swap_options):
                        best_option: dict[int, set] = sorted(swap_options, key=lambda t: self._calculate_utilities(t), reverse=True)[0]
                        if self._calculate_utilities(best_option) >= self._calculate_utilities({rn: new_solution[rn] for (rn, _, _) in room_assign.values()}):
                            # If the best swapped solution is better than the original
                            for (rn, room) in best_option.items():
                                new_solution[rn] = room
                            swap_cnt += 1
                    else:
                        # Pass if all roommates already prefer each other (can't move) 
                        continue      
                    
            elif len(agents) == 3:
                # For a 3-person preference group
                equal_status = {a: equal(map(lambda x: room_assign[x][0], other(agents, a))) for a in agents}
                if all(equal_status.values()):
                    # Already in the same room
                    continue
                elif any(equal_status.values()):
                    # Two are in the same room
                    # Assign the three agents and other 3 in the same room
                    new_assign = list(zip(set(room_nums), [set(agents), set([a for v in room_assign.values() for a in v[1] if a not in agents])]))
                    if self._calculate_utilities({rn: room for (rn, room) in new_assign}) >= self._calculate_utilities({rn: new_solution[rn] for rn in set(room_nums)}):
                        # If the best swapped solution is better than the original
                        for (rn, room) in new_assign:
                            new_solution[rn] = room  
                        swap_cnt += 1   
                else:
                    # All are in different rooms
                    swap_options = []
                    for (rn, roommates, status) in room_assign.values():
                        if status:
                            roommates = list(roommates)
                            swap_options.append({rn: set(agents)} | {o_rn: self._solution[o_rn] - set(agents) | {o_ag} 
                                                                     for (o_rn, o_ag) in zip(other(room_nums, rn), roommates)})
                            swap_options.append({rn: set(agents)} | {o_rn: self._solution[o_rn] - set(agents) | {o_ag} 
                                                                     for (o_rn, o_ag) in zip(other(room_nums, rn), reversed(roommates))})
                    if len(swap_options):
                        best_option: dict[int, set] = sorted(swap_options, key=lambda t: self._calculate_utilities(t), reverse=True)[0]
                        if self._calculate_utilities(best_option) >= self._calculate_utilities({rn: new_solution[rn] for (rn, _, _) in room_assign.values()}):
                            # If the best swapped solution is better than the original
                            for (rn, room) in best_option.items():
                                new_solution[rn] = room
                            swap_cnt += 1
                    else:
                        # Pass if all roommates already prefer each other (can't move) 
                        continue
            else:
                raise OverflowError(f"Expected preference group of 2 or 3. Got {len(agents)} ({agents})")
        if inplace:
            self._solution = new_solution
        if verbose:
            print(f"Total swap count:", swap_cnt)
        return new_solution
    
    def evaluate(self, verbose: int = 0):
        '''Return the proportional utilities increase after swap
        
        :param verbose: if 1, print the total number of swaps
        :return: the proportional utilities increase
        '''
        original_ut = self._calculate_utilities()
        return self._calculate_utilities(self._swap(verbose=verbose)) / original_ut


class ttc_evaluator(eval_base):
    '''Build the top trading graph concerning a specific solution
    
    Find the minimum alpha for the graph to be acyclic.
    '''
    def __init__(self, solution: dict[int, set[int]], utilities: np.ndarray):
        super().__init__(solution=solution, utilities=utilities)
        self._visited = {a: False for a in range(self.n_agents)}
        self._ancestor = {a: -1 for a in range(self.n_agents)}
        self._instack = {a: False for a in range(self.n_agents)}
    
    def _build_trading_graph(self):
        '''O(|V|^2) in building the weighted directed graph'''
        
        self._ttg = {a: [] for a in range(self.n_agents)}
        ator = {a: rn for (rn, room) in self.solution.items() for a in room}
        graph = {a: [] for a in range(self.n_agents)}
        
        for agent in range(self.n_agents):
            agent_rn = ator[agent]
            for oth in range(self.n_agents):
                if ator[oth] != agent_rn:
                    a_rms = tuple(self.solution[ator[agent]]-{agent})
                    o_rms = tuple(self.solution[ator[oth]]-{oth})
                    o_ut = self._utilities[agent, a_rms[0]] + self._utilities[agent, a_rms[1]]
                    if o_ut:    # Avoid division by 0 (considered as alpha = 1)
                        alpha = (self._utilities[agent, o_rms[0]] + self._utilities[agent, o_rms[1]]) / (self._utilities[agent, a_rms[0]] + self._utilities[agent, a_rms[1]])
                        if alpha > 1:
                            graph[agent].append((oth, alpha))
        self._ttg = graph
        return graph
    
    def _explore(self, v: int):
        self._visited[v] = True
        self._instack[v] = True     # Push v in stack
        for (u, _) in self._ttg[v]:
            if self._instack[u]:    # A backedge has been found
                self._ancestor[u] = v
                return u
            if not self._visited[u]:
                self._ancestor[u] = v
                res = self._explore(u)
                if res is not None: return res  # Propagate back the node in the cycle
        self._instack[v] = False    # Pop v out of the stack
                
    def _find_cycle(self):
        for v in range(self.n_agents):
            res = self._explore(v)
            if res is not None:
                cycle = [res]
                u = res
                while True:
                    u = self._ancestor[u]
                    if u == res:
                        break
                    cycle.append(u)
                cycle.reverse()
                return cycle
            
    def _cycle_vtoe(self, cycle: list[int]):
        res = []
        for i in range(len(cycle)):
            for (v, w) in self._ttg[cycle[i-1]]:
                if v == cycle[i]:
                    res.append((cycle[i-1], v, w))
        return res
                    
    def _reset(self):
        self._visited = {a: False for a in range(self.n_agents)}
        self._ancestor = {a: -1 for a in range(self.n_agents)}
        self._instack = {a: False for a in range(self.n_agents)}
    
    def evaluate(self, verbose: int = 0):
        '''Build the trading graph, and find the minimum alpha for the graph to be acyclic
        
        :param verbose: if 1, print the total number of cycles found and eliminated
        :return: the minimum alpha
        
        Algorithm: O((|V|+|E|)·|E|) in time complexity
        1. Initialize alpha <- 1
        2. Repeat for step k
            a. c_k <- FindCycle() \\
            b. e_k <- argmin_{e in c_k} w(e) \\
            c. E <- E \ {e in E | w(e) <= e_k} \\
            d. alpha <- w(e_k) \\
            until NoCycle
        3. Output alpha
        '''
        # Initialize
        cycle_cnt = 0
        alpha = 1
        self._build_trading_graph()
        
        while True:
            self._reset()
            cycle = self._find_cycle()
            if cycle is None:
                break
            cycle_cnt += 1
            alpha = sorted(map(lambda x: x[-1], self._cycle_vtoe(cycle)))[0]
            for v in range(self.n_agents):
                for (i, (u, w)) in enumerate(self._ttg[v]):
                    if w <= alpha:
                        self._ttg[v].pop(i)
        
        if verbose:
            print(f"Total cycle count:", cycle_cnt)
        return alpha