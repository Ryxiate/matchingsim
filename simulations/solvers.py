from .data import geo_3dsr
import numpy as np
import random

class solver_base(object):
    def __init__(self, inst: geo_3dsr):
        self.utilities = inst.utilities
        self.factors = inst.factors
        self.n_agents = inst.n_agents
        self.n_rooms = self.n_agents / 3
        if self.n_rooms - int(self.n_rooms):
            raise Exception("3SR Problem instance has agent numbers not the multiple of 3")
        else:   self.n_rooms = int(self.n_rooms)
        self.n_factors = inst.n_factors
        
        self._solution = {k: set() for k in range(self.n_rooms)}
    
    def reset(self):
        self._solution = {k: set() for k in range(self.n_rooms)}
        
    @property
    def solution(self):
        return self._solution
    
    def _solve(self):
        raise NotImplementedError
    
    def solve(self):
        self.reset()
        self._solve()
    
    def display_solution(self):
        for r in range(self.n_rooms):
            print(f"Room {r+1}: {self._solution[r]}")
            
    def solve_and_display(self):
        self.reset()
        self._solve()
        self.display_solution()
            
    def evaluate(self, verbose: int = 0):
        if any([len(self._solution[r]) != 3 for r in range(self.n_rooms)]):
            raise Exception("Solution empty or incomplete. Consider running the reset and solve method first.")
        room_utilities = np.zeros(self.n_rooms)
        for r in range(self.n_rooms):
            (i, j, k) = self._solution[r]
            room_utilities[r] = self.utilities[i, j] + self.utilities[j, k] + self.utilities[k, i]
        
        if verbose:
            return sum(room_utilities), room_utilities
        else:
            return sum(room_utilities)
        
    def evaluate_agentwise(self) -> np.ndarray:
        if any([len(self._solution[r]) != 3 for r in range(self.n_rooms)]):
            raise Exception("Solution empty or incomplete. Consider running the reset and solve method first.")
        agent_utilities = np.zeros(self.n_agents)
        for r in range(self.n_rooms):
            (i, j, k) = self._solution[r]
            agent_utilities[[i, j, k]] = self.utilities[[i, j, k]][:, [i, j, k]].sum(axis=1)
            
        return agent_utilities
        
    def evaluate_and_display(self, verbose: int = 0):
        if verbose:
            total, room_ut = self.evaluate(verbose=verbose)
            print("Total utilitarian welfare:", total)
            for r in range(self.n_rooms):
                print(f"Room {r+1}: {room_ut[r]}")
        else:
            total = self.evaluate(verbose=verbose)
            print("Total utilitarian welfare:", total)
        
    
class serial_dictatorship(solver_base):
    def _solve(self):
        for i in range(self.n_agents):
            remain_num = self.n_agents - i - 1
            if remain_num:
                remain_ut_avg = sum(self.utilities[i, i+1:]) / remain_num
                room_utilities = [sum([self.utilities[i, x] for x in self._solution[r]]) + remain_ut_avg * (2 - len(self._solution[r])) if len(self._solution[r]) < 3 
                                  else -1 for r in range(self.n_rooms)]
                room_assign = np.argmax(room_utilities)
            else:
                room_assign = np.argmin([len(self._solution[r]) for r in range(self.n_rooms)])
            self._solution[room_assign].add(i)
            

class match_by_characteristics(solver_base):
    def _solve(self):
        ut_three = dict()
        for i in range(self.n_agents):
            for j in range(i+1, self.n_agents):
                for k in range(j+1, self.n_agents):
                    ut_three[(i, j, k)] = np.linalg.norm(self.factors[i] - self.factors[j]) + np.linalg.norm(self.factors[j] - self.factors[k]) \
                                        + np.linalg.norm(self.factors[k] - self.factors[i])
        r = 0
        while len(ut_three):
            (i, j, k) =  sorted(ut_three.keys(), key=lambda ky: ut_three[ky])[0]
            self._solution[r] = set((i, j, k))
            for key in list(ut_three.keys()):
                if i in key or j in key or k in key:
                    ut_three.pop(key)
            r += 1
    

class random_match(solver_base):
    def _solve(self):
        agents = list(range(self.n_agents))
        r = 0
        while len(agents):
            a = random.sample(agents, 3)
            for i in a:
                agents.remove(i)
                self._solution[r].add(i)
            r += 1
            
            
class brute_force(solver_base):
    def _solve(self):
        pass
            
    
    
if __name__ == "__main__":
    ins = geo_3dsr.get_instance(rooms=5)
    print(ins.utilities)
    solver = match_by_characteristics(inst=ins)
    solver.solve()
    solver.evaluate_and_display()