from simulations import *

# inst = geo_3dsr.get_instance(pref_premium=1, rooms=3, asymmetric_noise=True, groups_allowed=True, groups_p=1, p=0)
for _ in range(10):
    inst = geo_3dsr.get_instance(rooms=20, asymmetric_noise=True)
    # print(inst.utilities)
    solver = match_by_characteristics(inst)
    solver.solve()
    evaluator = preference_swapper.from_solver(solver)
    print(evaluator.evaluate())
# print(inst.preferences)
# print(solver.evaluate())

# swapper = preference_swapper.from_solver(solver)
# print(swapper.utilities_after_swap(verbose=1))

# print(solver.utilities)
# print(solver.solution)


# print(ins.revealed_utilities)
# ins.utilities_dropout(proportion=0.3, p=1)
# diff = ins.utilities - ins.revealed_utilities
# print(diff)
# print(diff.sum(axis=1))
# solver = serial_dictatorship(inst=ins)
# solver.solve()
# print(solver.evaluate_agentwise())