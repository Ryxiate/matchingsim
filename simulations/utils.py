import numpy as np

__all_solvers = ["SD_by_rooms", "serial_dictatorship", "random_serial_dictatorship", 
                 "match_by_characteristics", "random_match"]

__all_evaluators = ["preference_swapper", "ttc_evaluator"]

def check_solver_validity(solver_list: list[str], quiet: bool = False):
    for solver in solver_list:
        if solver not in __all_solvers:
            if quiet:
                solver_list.remove(solver)
            else:
                raise NameError(f"Solver name {solver} is invalid. Valid solvers are {__all_solvers}.")
    return solver_list

def check_evaluator_validity(eval_list: list[str], quiet: bool = False):
    for evaluator in eval_list:
        if evaluator not in __all_evaluators:
            if quiet:
                eval_list.remove(evaluator)
            else:
                raise NameError(f"Evaluator name {evaluator} is invalid. Valid evaluators are {__all_evaluators}.")
    return eval_list

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
        