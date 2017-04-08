import os

from optimization_solver import routing_solver, test_solver
from road_network import RoadNetwork
from time import time
from utils import generate_grid_network, generate_random_demand_list, generate_random_gaussian_demand_list, \
    generate_corner_through_demand

grid_rows = 3
grid_cols = 3
road_capacity = 100.0
bg_rate = 0.5
a = 1.0
b = 1.0
demand_coverage_rate = 0.2
per_demand_vol = 10.0
noisy_cost = True

print("Building " + str(grid_rows) + "*" + str(grid_cols) + " grid:")
node_list, road_list = generate_grid_network(grid_rows,
                                             grid_cols,
                                             capacity=road_capacity,
                                             bg_volume_rate=bg_rate,
                                             alpha=a,
                                             beta=b,
                                             cost_noise=noisy_cost)

print("Building routing request...")
node_count = grid_rows * grid_cols
demand_pairs = node_count * (node_count - 1) * demand_coverage_rate
demand_list, control_traffic = generate_corner_through_demand(
    node_list,
    grid_rows,
    grid_cols,
    demand_pairs,
    per_demand_vol)

total_capacity = ((grid_cols - 1) * grid_rows + (grid_rows - 1) * grid_cols) * road_capacity
bg_traffic = total_capacity * bg_rate
control_rate = control_traffic / (bg_traffic + control_traffic)
congestion_rate = (bg_traffic + control_traffic) / total_capacity
for demand in demand_list:
    print(demand)
print("Routing " + str(demand_pairs) + " pairs (" + str(control_rate * 100) +
      "% of traffic) when congestion_rate=" + str(congestion_rate))

os.system("pause")

print("generating constraint matrix")
routing_network = RoadNetwork(node_list, road_list)
#print(routing_network._dist_matrix)
A, b, unknown_variables, predefined_variable_values = routing_network.generate_constraint_matrix(demand_list, True)

routing_network.reset_social_volume()
start_time = time()
for demand in demand_list:
    from_node = int(demand['from'])
    to_node = int(demand['to'])
    car_count = int(demand['volume'])
    for i in range(0, car_count):
        routing_network.route_one_car_greedily(from_node, to_node)
total_cost = routing_network.compute_social_routing_cost()
end_time = time()
time_cost = end_time - start_time
print("current speed greedy routing cost:" + str(total_cost) + " time cost:" + str(time_cost))

routing_network.reset_social_volume()
start_time = time()
for demand in demand_list:
    from_node = int(demand['from'])
    to_node = int(demand['to'])
    car_count = int(demand['volume'])
    for i in range(0, car_count):
        routing_network.route_one_car_socially(from_node, to_node, 0.0)
total_cost = routing_network.compute_social_routing_cost()
end_time = time()
time_cost = end_time - start_time
print("sequential greedy routing cost:" + str(total_cost) + " time cost:" + str(time_cost))

routing_network.reset_social_volume()
start_time = time()
for demand in demand_list:
    from_node = int(demand['from'])
    to_node = int(demand['to'])
    car_count = int(demand['volume'])
    for i in range(0, car_count):
        routing_network.route_one_car_socially(from_node, to_node)
total_cost = routing_network.compute_social_routing_cost()
end_time = time()
time_cost = end_time - start_time
print("sequential social routing cost:" + str(total_cost) + " time cost:" + str(time_cost))

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
    print(mode +" routing cost:" + str(total_cost) + " time cost:" + str(time_cost))