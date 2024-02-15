import numpy as np
import warnings
import random
from .utils import *

def generate_factors(rooms: int = 10, n_factors: int = 5, high: int = 7) -> np.ndarray:
    '''Generates integer value factors in a np.ndarray
    :param rooms: number of rooms, the output shape is (3*rooms, n_factors)
    :param n_factor: number of factors to generate
    :param high: highest value a factor can take (i.e. factors take values in [1, high])
    :return: factors matrix with shape (3*rooms, n_factors)
    '''
    return np.array([[np.random.randint(low=1, high=high+1) for _ in range(n_factors)] for __ in range(3*rooms)])

class geo_3dsr(object):
    '''A geometric 3D-SR instance with noise when calculating utilities
    
    A few things about utilities used in this class:
    1. `_utility_matrix`: matrix of utilities directly computed from factors and kernel functions
    2. `_revealed_utilities`: initially the `_utility_matrix` plus the `_noise_matrix`. Subject to change by methods like `utilities_dropout`
    3. `utilities`: true utilities computed from `_utility_matrix + _noise_matrix`, used for evaluation
    4. `revealed_utilities`: utilities that agents perceive before matching, used for matching
    '''
    def __init__(self, factors: np.ndarray, high: int, kernel: str, c: float, noise: float, pref_premium: float, pref_dist: str, p: float, 
                 lim_prop: float, groups_p: float, groups_allowed: bool, quiet: bool, asymmetric_noise: bool):
        self._factors = factors
        self._h = high
        (self.n_agents, self.n_factors) = factors.shape
        self.c = c
        self.noise = noise
        self.quiet = quiet
        self.asy = asymmetric_noise

        if hasattr(kernel, "__call__"):
            self._kernel_func = kernel
        elif hasattr(eval(kernel), "__call__"):
            self._kernel_func = eval(kernel)
        else:
            raise Exception(f"Unrecognized kernel function: {kernel}")

        self._gen_preferences(pref_dist=pref_dist, p=p, lim_p=lim_prop, groups_allowed=groups_allowed, groups_p=groups_p)
        self._update_utilities(kernel_func=self._kernel_func, c=self.c, noise=self.noise, quiet=self.quiet, resample_noise=True, asymmetric_noise=self.asy)
        self._pref_premium = pref_premium * np.maximum(self._utility_matrix + self._noise_matrix, 0).sum() / (self.n_agents * (self.n_agents - 1))
    
    def _gen_preferences(self, pref_dist: str, p: float, lim_p: float, groups_allowed: bool, groups_p: float):
        lim_p = min(max(lim_p, 0), 1)
        p = min(max(p, 0), 1)
        
        self._preferences: set[tuple] = set()
        all_agents = list(range(self.n_agents))
        if pref_dist == "geo":
            while len(self._preferences) < self.n_agents / 3 * lim_p:
                num = 3 if groups_allowed and random.random() <= groups_p else 2
                pref_agents = random.sample(all_agents, num)
                self._preferences.add(tuple(pref_agents))
                for a in pref_agents:
                    all_agents.remove(a)
                if random.random() > p:
                    break
        elif pref_dist == "uniform":
            num_pref = max(int(random.random() * self.n_agents / 3 * lim_p), min(self.n_agents/ 3 - 1, 1))
            for _ in range(num_pref):
                num = 3 if groups_allowed and random.random() <= groups_p else 2
                pref_agents = random.sample(all_agents, num)
                self._preferences.add(tuple(pref_agents))
                for a in pref_agents:
                    all_agents.remove(a)
        else:
            raise NameError(f"Invalid preference distribution. Expected 'geo' or 'uniform'. Got {pref_dist}.")
            
        self._pref_matrix = np.zeros((self.n_agents, self.n_agents))
        for agents in self._preferences:
            self._pref_matrix[agents[0], agents[1]] = 1
            self._pref_matrix[agents[1], agents[0]] = 1
            if len(agents) == 3:
                self._pref_matrix[agents[0], agents[2]] = 1
                self._pref_matrix[agents[2], agents[0]] = 1
                self._pref_matrix[agents[1], agents[2]] = 1
                self._pref_matrix[agents[2], agents[1]] = 1
    
    def _update_utilities(self, kernel_func, c: float, noise: float, quiet: bool, resample_noise: bool, asymmetric_noise: bool):
        if resample_noise:
            if asymmetric_noise:
                self._noise_matrix = np.random.normal(loc=0, scale=noise, size=(self.n_agents, self.n_agents))
                self._noise_matrix -= np.diag(self._noise_matrix.diagonal())
            else:
                self._noise_matrix = np.triu(np.random.normal(loc=0, scale=noise, size=(self.n_agents, self.n_agents)))
                self._noise_matrix += self._noise_matrix.T - 2 * np.diag(self._noise_matrix.diagonal())
        self._utility_matrix = np.zeros((self.n_agents, self.n_agents))
        for i in range(self.n_agents):
            for j in range(i+1, self.n_agents):
                u = kernel_func(np.linalg.norm(self._factors[i] - self._factors[j]), c=c)
                self._utility_matrix[i, j] = u 
                self._utility_matrix[j, i] = u
        self._revealed_utilities = self._utility_matrix + self._noise_matrix
                
        if np.linalg.norm(self._utility_matrix - np.abs(self._noise_matrix)) <= self.n_agents * noise and not quiet and resample_noise:
            warnings.warn("Parameter c and noise inappropriately set. The utility may be too small for noise.")
       
    @property
    def preferences(self):
        return self._preferences.copy()
    
    @property
    def preferences_matrix(self):
        return self._pref_matrix.copy()
    
    @property
    def utilities(self):
        return np.maximum(self._utility_matrix + self._noise_matrix + self._pref_matrix * self._pref_premium, 0)
    
    @property
    def revealed_utilities(self):
        return np.maximum(self._revealed_utilities + self._pref_matrix * self._pref_premium, 0)
    
    @property
    def factors(self):
        return self._factors.copy()
    
    @classmethod       
    def get_instance(cls, /, rooms: int = 10, n_factors: int = 5, high: int = 7, kernel: str = "rbf", c: float = 25, noise: float = 3, pref_premium: float = 0.6, 
                     pref_dist: str = "geo", p: float = 0.75, lim_prop: float = 0.5, groups_allowed: bool = True, groups_p: float = 0.3, quiet: bool = False, 
                     asymmetric_noise: bool = False):
        '''Generates an instance of Geometric 3D-SR
        :param rooms: number of rooms (there will be 3*rooms agents)
        :param n_factors: number of factors
        :param high: highest value of factors ([1, high], integer value)
        :param kernel: kernel function to transform the distance into utilities, chosen from [`"rbf"`, `"epanechnikov"`, `"tri_cube"`]
        :param c: parameter for the kernel function
        :param noise: Gaussian noise scale when calculating noise for utilities
        :param pref_premium: preference premium markup on utilities based on the average utilities across all agents
        :param pref_dist: preference number distribution style, chosen from [`"geo"`, `"uniform"`].
        :param p: probability of generating the next preferences, ignored if `pref_dist` is `"uniform"`
        :param lim_prop: limits of proportion of people to have preferences
        :param groups_allowed: whether to be able to generate a group (3-person) preferences
        :param groups_p: probability of generating a group preferences. Ignored if `groups_allowed` is set to false
        :param quiet: If true, will not warn the user for inappropriately set parameters (noise and c)
        :param asymmetric_noise: If true, the Gaussian noise will be asymmetric (e_{ij} != e_{ji})
        :return: a geometric 3D-SR instance
        '''
        factors = generate_factors(rooms=rooms, n_factors=n_factors, high=high)
        return cls(factors=factors, high=high, kernel=kernel, c=c, noise=noise, quiet=quiet, asymmetric_noise=asymmetric_noise, pref_premium=pref_premium,
                   pref_dist=pref_dist, p=p, lim_prop=lim_prop, groups_p=groups_p, groups_allowed=groups_allowed)
    
    def factor_bias(self, bias_num: int = 1, bias_scale: int = 5, no_update: bool = True):
        '''Applies a random bias on a random factor for the Geo 3D-SR instance
        :param bias_num: number of factors affected by the bias
        :param bias_scale: maximum bias (can be negative for negative bias)
        :param no_update: do not update the utilities after applying the bias
        '''
        bias_num = min(bias_num, self.n_factors)
        biased = random.sample(range(self.n_factors), bias_num)
        self._factors[:, biased] += np.array([[int(random.random() * bias_scale) for _ in range(bias_num)] for __ in range(self.n_agents)])
        self._factors = np.maximum(np.minimum(self._factors, self._h), 0)
        if not no_update:
            self._update_utilities(kernel_func=self._kernel_func, c=self.c, noise=self.noise, quiet=self.quiet, resample_noise=False)
            
    def utilities_dropout(self, p: float = 0.6, proportion: float = 1.0, policy: str = "mean"):
        '''Randomly drops out utilities to represent indifference caused by agent's incomplete information on others
        :param p: probability of agents whose utilities will be dropped
        :param proportion: proportion of utilities of the 'drop-out' agents to another agent that will be dropped
        :param policy: can be in [`"mean"`, `"min"`, "max"`]
        '''
        p = min(1, max(0, p))
        proportion = min(1, max(0, proportion))
        if policy not in ["mean", "min", "max"]:
            raise Exception("policy should be in ['mean', 'min', 'max']")
        policy = eval(policy)
        
        drop_agents = np.array(range(self.n_agents))[[random.random() < p for _ in range(self.n_agents)]]
        for agent in drop_agents:
            rem_agents = [a for a in range(self.n_agents) if a != agent]
            drop_uts = random.sample(rem_agents, int(len(rem_agents) * proportion))
            indiff_ut = policy(self._revealed_utilities[agent, :])
            for a in drop_uts:
                self._revealed_utilities[agent, a] = indiff_ut
    
    def __repr__(self):
        return f"3D-SR Instance with {self.n_agents} agents"
    
    
    
if __name__ == "__main__":
    ins = geo_3dsr.get_instance(rooms=2)
    print(ins.utilities)