import numpy as np

def rbf(dist: float, c: float = 25):
    return c * np.exp(-dist**2/c)

def epanechnikov(dist: float, c: float = 25):
    return c * max(3 * (1 - dist / c) / 4, 0)

def tri_cube(dist: float, c: float = 25):
    return c * max((1 - dist / c) ** 3, 0)

def mean(__iterable):
    return sum(__iterable) / len(__iterable)

def equal(__iterable):
    for n, item in enumerate(__iterable):
        if not n:   base = item
        elif item != base:  return False
    return True

def other(__iterable, item):
    '''Return the other element(s) in an iterable. 
    
    If there's only 2 elements, return the other element; if there's more than 2, return the others as a list
    '''
    assert item in __iterable
    others = [i for i in __iterable if i != item]
    return others[0] if len(others) == 1 else others
        