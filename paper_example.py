from optimization_solver import routing_solver
from road_network import RoadNetwork
from time import time
from utils import generate_paper_network, generate_paper_demand_list

node_list, road_list = generate_paper_network()
demand_list = generate_paper_demand_list()

print("generating constraint matrix")
routing_network = RoadNetwork(node_list, road_list)
A, b, unknown_variables, predefined_variable_values = routing_network.generate_constraint_matrix(demand_list, True)

for mode in ['system', 'ue']:
    print("start:" + mode + "...")
    start_time = time()
    routing_result, total_cost = routing_solver(
        A,
        b,
        unknown_variables,
        demand_list,
        road_list,
        mode,
        5000,
        10**-10,
        10**-10,
        10**-10)
    end_time = time()
    time_cost = end_time - start_time
    print("routing cost:" + str(total_cost) + " time cost:" + str(time_cost))

