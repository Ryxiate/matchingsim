from simulations import *

ins = geo_3dsr.get_instance(rooms=2, kernel="tri_cube", asymmetric_noise=True)
print(ins.factors)
print(solvers._pairwise_distances(ins.factors))

# print(ins.revealed_utilities)
# ins.utilities_dropout(proportion=0.3, p=1)
# diff = ins.utilities - ins.revealed_utilities
# print(diff)
# print(diff.sum(axis=1))
# solver = serial_dictatorship(inst=ins)
# solver.solve()
# print(solver.evaluate_agentwise())