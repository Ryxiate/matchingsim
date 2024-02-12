from simulations import *

inst = geo_3dsr.get_instance(pref_premium=1, rooms=3, asymmetric_noise=True, groups_allowed=True, groups_p=1)
# print(inst.utilities)
solver = random_match(inst)
solver.solve()
print(inst.preferences)
print(solver.evaluate())

# swapper = preference_swapper.from_solver(solver)
# print(swapper.utilities_after_swap())

print(solver.utilities)
print(solver.solution)

evaluator = ttg_evaluator.from_solver(solver)
print({u: [x[0] for x in l] for (u, l) in evaluator._build_trading_graph().items()})
# evaluator._ttg = {0: [(1, 1)],
#                   1: [(4, 1)],
#                   2: [], 3: [],
#                   4: [(1, 1)],
#                   5: [], 6: [],
#                   7: [], 8: []}
print(evaluator._find_cycle())
print(evaluator.evaluate())


# print(ins.revealed_utilities)
# ins.utilities_dropout(proportion=0.3, p=1)
# diff = ins.utilities - ins.revealed_utilities
# print(diff)
# print(diff.sum(axis=1))
# solver = serial_dictatorship(inst=ins)
# solver.solve()
# print(solver.evaluate_agentwise())