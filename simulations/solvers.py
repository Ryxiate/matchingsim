from .data import geo_3dsr
from .utils import *
import numpy as np
import random

def _pairwise_distances(factors: np.ndarray) -> np.ndarray:
    dot_product = np.matmul(factors, factors.T)   
    square_norm = np.diag(dot_product)
    distances = np.maximum(np.expand_dims(square_norm, axis=1) - 2.0 * dot_product + np.expand_dims(square_norm, axis=0), 0)
    return np.sqrt(distances)

class solver_base(object):
    def __init__(self, inst: geo_3dsr):
        self.utilities = inst.utilities
        self.revealed = inst.revealed_utilities
        self.factors = inst.factors
        self.n_agents = inst.n_agents
        self.n_rooms = self.n_agents / 3
        self.preferences = inst.preferences
        if self.n_rooms - int(self.n_rooms):
            raise Exception("3SR Problem instance has agent numbers not the multiple of 3")
        else:   self.n_rooms = int(self.n_rooms)
        self.n_factors = inst.n_factors
        
        self._solution = {k: set() for k in range(self.n_rooms)}
    
    def reset(self):
        self._solution = {k: set() for k in range(self.n_rooms)}
        
    @property
    def solution(self):
        return self._solution.copy()
    
    def _solve(self):
        raise NotImplementedError(f"Solve method not supported for class {type(self).__name__}")
    
    def solve(self):
        self.reset()
        self._solve()
        return self.solution
    
    def display_solution(self):
        for r in range(self.n_rooms):
            print(f"Room {r+1}: {self._solution[r]}")
            
    def solve_and_display(self):
        self.reset()
        self._solve()
        self.display_solution()
            
    def evaluate(self, return_mean: bool = True):
        if any([len(self._solution[r]) != 3 for r in range(self.n_rooms)]):
            raise Exception("Solution empty or incomplete. Consider running the solve method first.")
        room_utilities = np.zeros(self.n_rooms)
        for r in range(self.n_rooms):
            (i, j, k) = self._solution[r]
            room_utilities[r] = self.utilities[[i, j, k]][:, [i, j, k]].sum()
        
        if return_mean:
            return mean(room_utilities)
        else:
            return room_utilities
        
    def evaluate_agentwise(self) -> np.ndarray:
        if any([len(self._solution[r]) != 3 for r in range(self.n_rooms)]):
            raise Exception("Solution empty or incomplete. Consider running the solve method first.")
        agent_utilities = np.zeros(self.n_agents)
        for r in range(self.n_rooms):
            (i, j, k) = self._solution[r]
            agent_utilities[[i, j, k]] = self.utilities[[i, j, k]][:, [i, j, k]].sum(axis=1)
            
        return agent_utilities
        
    def evaluate_and_display(self, verbose: int = 0):
        if verbose:
            total, room_ut = self.evaluate(return_mean=verbose)
            print("Total utilitarian welfare:", total)
            for r in range(self.n_rooms):
                print(f"Room {r+1}: {room_ut[r]}")
        else:
            total = self.evaluate(return_mean=verbose)
            print("Total utilitarian welfare:", total)
        
    
class SD_by_rooms(solver_base):
    '''Agents will iteratively choose the room they prefer. 
    
    If a room isn't full when they join, they will evaluate the vacancies based on the average utilities of the remaining agents
    '''
    def _solve(self):
        for i in range(self.n_agents):
            remain_num = self.n_agents - i - 1
            if remain_num:
                remain_ut_avg = sum(self.revealed[i, i+1:]) / remain_num
                room_utilities = [sum([self.revealed[i, x] for x in self._solution[r]]) + remain_ut_avg * (2 - len(self._solution[r])) if len(self._solution[r]) < 3 
                                  else -1 for r in range(self.n_rooms)]
                room_assign = np.argmax(room_utilities)
            else:
                room_assign = np.argmin([len(self._solution[r]) for r in range(self.n_rooms)])
            self._solution[room_assign].add(i)


class serial_dictatorship(solver_base):
    '''Agents will choose between a room with 2 people to join, or start a new room with their favorite agent who hasn't been in a room.
    
    Theoretical analysis shows the produced results are efficient.
    '''
    def _solve(self):
        agent_list = list(range(self.n_agents))
        occupied_room = 0
        while True:
            a = agent_list[0]
            agent_list.remove(a)
            ut_rooms = [sum([self.revealed[a, i] for i in self._solution[r]]) if len(self._solution[r]) == 2 else -1 for r in range(self.n_rooms)]
            fav_room = np.argmax(ut_rooms)
            if occupied_room < self.n_rooms:
                ut_agents = self.revealed[a, agent_list]
                fav_agent = agent_list[np.argmax(ut_agents)]
                ut_rem = np.delete(ut_agents, np.argmax(ut_agents)).mean()
                if ut_rooms[fav_room] > self.revealed[a, fav_agent] + ut_rem:
                    self._solution[fav_room].add(a)
                else:
                    self._solution[occupied_room] = set([a, fav_agent])
                    agent_list.remove(fav_agent)
                    occupied_room += 1
            else:
                self._solution[fav_room].add(a)
                if not len(agent_list): break


class random_serial_dictatorship(solver_base):
    '''
    Like serial dictatorship, but the ordering of agents' deciding is randomized
    '''
    def _solve(self):
        agent_list = list(range(self.n_agents))
        occupied_room = 0
        while True:
            a = random.sample(agent_list, 1)[0]
            agent_list.remove(a)
            ut_rooms = [sum([self.revealed[a, i] for i in self._solution[r]]) if len(self._solution[r]) == 2 else -1 for r in range(self.n_rooms)]
            fav_room = np.argmax(ut_rooms)
            if occupied_room < self.n_rooms:
                ut_agents = self.revealed[a, agent_list]
                fav_agent = agent_list[np.argmax(ut_agents)]
                ut_rem = np.delete(ut_agents, np.argmax(ut_agents)).mean()
                if ut_rooms[fav_room] > self.revealed[a, fav_agent] + ut_rem:
                    self._solution[fav_room].add(a)
                else:
                    self._solution[occupied_room] = set([a, fav_agent])
                    agent_list.remove(fav_agent)
                    occupied_room += 1
            else:
                self._solution[fav_room].add(a)
                if not len(agent_list): break
            
            
class match_by_characteristics(solver_base):
    '''Greedy algorithm to provide a 2-stable approximation solution for Geometric 3D-SR
    
    Iteratively choose the 3 agents forming the least perimeter triangle in the factor space to be roommates
    '''
    def _solve(self):
        distances = _pairwise_distances(self.factors)
        ut_three = dict()
        for i in range(self.n_agents):
            for j in range(i+1, self.n_agents):
                for k in range(j+1, self.n_agents):
                    ut_three[(i, j, k)] = distances[i, j] + distances[j, k] + distances[k, i]
        r = 0
        while len(ut_three):
            (i, j, k) =  sorted(ut_three.keys(), key=lambda ky: ut_three[ky])[0]
            self._solution[r] = set((i, j, k))
            for key in list(ut_three.keys()):
                if i in key or j in key or k in key:
                    ut_three.pop(key)
            r += 1
    

class random_match(solver_base):
    '''Dummy matching algorithm as the baseline
    
    Randomly match agents together as roommates
    '''
    def _solve(self):
        agents = list(range(self.n_agents))
        r = 0
        while len(agents):
            a = random.sample(agents, 3)
            for i in a:
                agents.remove(i)
                self._solution[r].add(i)
            r += 1
            
    
    
if __name__ == "__main__":
    ins = geo_3dsr.get_instance(rooms=5)
    print(ins.utilities)
    solver = match_by_characteristics(inst=ins)
    solver.solve()
    solver.evaluate_and_display()