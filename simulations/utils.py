import numpy as np

def rbf(dist: float, c: float = 25):
    return c * np.exp(-dist**2/c)

def epanechnikov(dist: float, c: float = 25):
    return c * max(3 * (1 - dist / c) / 4, 0)

def tri_cube(dist: float, c: float = 25):
    return c * max((1 - dist / c) ** 3, 0)