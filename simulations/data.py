import numpy as np
import warnings
import random
from .utils import *

def generate_factors(rooms: int = 10, n_factors: int = 5, high: int = 7) -> np.ndarray:
    '''Generate integer value factors in a np.ndarray
    :param rooms: number of rooms, the output shape is (3*rooms, n_factors)
    :param n_factor: number of factors to generate
    :param high: highest value a factor can take (i.e. factors take values in [1, high])
    :return: factors matrix with shape (3*rooms, n_factors)
    '''
    return np.array([[np.random.randint(low=1, high=high+1) for _ in range(n_factors)] for __ in range(3*rooms)])

class geo_3dsr(object):
    def __init__(self, factors: np.ndarray, high: int = 7, kernel: str = "rbf", c: float = 25, noise: float = 1, quiet: bool = False, asymmetric_noise: bool = False):
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

        self._update_utilities(kernel_func=self._kernel_func, c=self.c, noise=self.noise, quiet=self.quiet, asymmetric_noise=self.asy)
    
    def _update_utilities(self, kernel_func, c: float = 25, noise: float = 1, quiet: bool = False, resample_noise: bool = True, asymmetric_noise: bool = False):
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
    
    @classmethod       
    def get_instance(cls, /, rooms: int = 10, n_factors: int = 5, high: int = 7, kernel: str = "rbf", c: float = 25, noise: float = 1, quiet: bool = False, asymmetric_noise: bool = False):
        '''Generate an instance of Geometric 3D-SR
        :param rooms: number of rooms (there will be 3*rooms agents)
        :param n_factors: number of factors
        :param high: highest value of factors (1-high, integer value)
        :param kernel: kernel function to transform the distance into utilities, chosen from [`"rbf"`, `"epanechnikov"`, `"tri_cube"`]
        :param c: parameter for the kernel function
        :param noise: Gaussian noise scale when calculating noise for utilities
        :param quiet: If true, will not warn the user for inappropriately set parameters (noise and c)
        :param asymmetric_noise: If true, the Gaussian noise will be asymmetric (e_{ij} != e_{ji})
        :return: a geometric 3D-SR instance
        '''
        factors = generate_factors(rooms=rooms, n_factors=n_factors, high=high)
        return cls(factors=factors, high=high, kernel=kernel, c=c, noise=noise, quiet=quiet, asymmetric_noise=asymmetric_noise)
    
    @property
    def utilities(self):
        return np.maximum(self._utility_matrix + self._noise_matrix, 0)
    
    @property
    def revealed_utilities(self):
        return np.maximum(self._revealed_utilities, 0)
    
    @property
    def factors(self):
        return self._factors.copy()
    
    def factor_bias(self, bias_num: int = 1, bias_scale: int = 5, no_update: bool = True):
        '''Apply a random bias on a random factor for the Geo 3D-SR instance
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
        '''Randomly drop out utilities to represent indifference caused by agent's incomplete information on others
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
        return f"3SR Instance with {self.n_agents} agents"
    
    
    
if __name__ == "__main__":
    ins = geo_3dsr.get_instance(rooms=2)
    print(ins.utilities)