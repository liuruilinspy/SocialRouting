from random import randint, random
import numpy


def generate_paper_network():
    vertex_list = [{'id': 1}, {'id': 2}, {'id': 3}, {'id': 4}]
    edge_list = [
        {'id': 1, 'from_node': 1, 'to_node': 2, 'cost': 1, 'capacity': 100, 'bg_volume': 0, 'alpha': 1, 'beta': 1},
        {'id': 2, 'from_node': 1, 'to_node': 3, 'cost': 2, 'capacity': 100, 'bg_volume': 0, 'alpha': 0, 'beta': 0},
        {'id': 3, 'from_node': 2, 'to_node': 3, 'cost': 0.25, 'capacity': 100, 'bg_volume': 0, 'alpha': 0, 'beta': 0},
        {'id': 4, 'from_node': 2, 'to_node': 4, 'cost': 2, 'capacity': 100, 'bg_volume': 0, 'alpha': 0, 'beta': 0},
        {'id': 5, 'from_node': 3, 'to_node': 4, 'cost': 1, 'capacity': 100, 'bg_volume': 0, 'alpha': 1, 'beta': 1}]

    return vertex_list, edge_list


def generate_paper_demand_list():
    request_list = [{'from': 1, 'to': 4, 'volume': 100}]
    return request_list


def generate_grid_network(
        height,
        width,
        cost_free=120.0,
        capacity=100.0,
        bg_volume_rate=0.6,
        alpha=1.0,
        beta=1.0,
        cost_noise=False):
    vertex_list = []
    edge_list = []
    for row in range(0, height):
        for col in range(0, width):
            cur_node = row * width + col
            vertex_list.append({
                'id': cur_node
            })

            neighbors = []
            if cur_node - width >= 0:
                neighbors.append(cur_node - width)
            if cur_node % width != 0:
                neighbors.append(cur_node - 1)
            if (cur_node + 1) % width != 0:
                neighbors.append(cur_node + 1)
            if cur_node + width < height * width:
                neighbors.append(cur_node + width)

            for neighbor_node in neighbors:
                cost = cost_free if not cost_noise else cost_free * (1 + (random() - 0.5) * 0.02)
                edge_list.append(
                    {'id': len(edge_list),
                     'from_node': cur_node,
                     'to_node': neighbor_node,
                     'cost': cost,  # +- 0.01 noise
                     'capacity': capacity,
                     'bg_volume': bg_volume_rate * capacity,
                     'alpha': alpha,
                     'beta': beta}
                )

    return vertex_list, edge_list


# this function return the manhattan distance in the grid network
def get_manhattan_distance(from_index, to_index, cols):
    to_row = to_index / cols
    to_col = to_index % cols
    from_row = from_index / cols
    from_col = from_index % cols
    return abs(from_row - to_row) * abs(from_col - to_col)


# randomly sample a node from a grid network according to the given gaussian distribution.
def sample_position(mean, cov, rows, cols):
    while True:
        x, y = numpy.random.multivariate_normal(mean, cov)
        if 0 <= y < rows and 0 <= x < cols:
            index = int(y) * cols + int(x)
            return index


# return a totally random demand list
def generate_random_demand_list(vertex_list, rows, cols, demand_pair_count, demand_size):
    request_list = []
    vertex_count = len(vertex_list)
    acc_volume = 0
    while len(request_list) < demand_pair_count:
        from_index = randint(0, vertex_count - 1)
        to_index = randint(0, vertex_count - 1)
        if from_index == to_index:
            continue
        request_list.append(
            {'from': vertex_list[from_index]['id'],
             'to': vertex_list[to_index]['id'],
             'volume': demand_size}
        )
        acc_volume += get_manhattan_distance(from_index, to_index, cols) * demand_size

    return request_list, acc_volume


# return a gaussian demand_list between two given src and dst distribution
def generate_random_gaussian_demand_list(vertex_list, src_mean, src_cov, dst_mean, dst_cov,
                                         rows, cols, demand_pair_count, demand_size):
    request_list = []
    acc_volume = 0
    while len(request_list) < demand_pair_count:
        from_index = sample_position(src_mean, src_cov, rows, cols)
        to_index = sample_position(dst_mean, dst_cov, rows, cols)
        if from_index == to_index:
            continue
        request_list.append(
            {'from': vertex_list[from_index]['id'],
             'to': vertex_list[to_index]['id'],
             'volume': demand_size}
        )
        acc_volume += get_manhattan_distance(from_index, to_index, cols) * demand_size

    return request_list, acc_volume


# generate the gaussian demand list from up-left corner to the down-right corner
def generate_corner_through_demand(vertex_list, rows, cols, demand_pair_count, demand_size):
    src_mean = [0, 0]
    src_cov = [[1, 0], [0, 1]]
    dst_mean = [rows - 1, cols - 1]
    dst_cov = [[1, 0], [0, 1]]
    return generate_random_gaussian_demand_list(
        vertex_list,
        src_mean,
        src_cov,
        dst_mean,
        dst_cov,
        rows,
        cols,
        demand_pair_count,
        demand_size)
