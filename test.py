from simulations import *

ins = geo_3dsr.get_instance(rooms=2, kernel="tri_cube")
u1 = ins.utilities.copy()
ins.factor_bias(bias_scale=5)
print(ins.utilities - u1)
solver = match_by_characteristics(inst=ins)
solver.solve()
print(solver.evaluate_agentwise())