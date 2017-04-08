import numpy

from graph_utils import all_pairs_shortest_path, compute_edge_traffic_cost, construct_shortest_path, \
    dijkstra_shortest_path, get_routing_cost


class RoadNetwork:
    def __init__(self, node_list, edge_list):
        # node_list -> list of node {'id' = id, 'attr1' = attr1 ...}
        # edge_list -> list of edge {'id' = id, 'from_node' = node_id, 'to_node' = node_id,
        #                               'alpha' = alpha, 'beta' = beta, 'cost' = free_flow_cost

        self._node_list = node_list.copy()
        self._edge_list = edge_list.copy()
        self._node_count = len(self._node_list)
        self._edge_count = len(self._edge_list)

        self._node_id_to_node_index = {}
        self._node_out_edge_indexes = []
        self._node_in_edge_indexes = []
        for i in range(0, self._node_count):
            node_id = int(self._node_list[i]['id'])
            if node_id in self._node_id_to_node_index:
                print("WARNING: duplicated node_id:" + str(node_id))
            self._node_id_to_node_index[node_id] = i
            self._node_out_edge_indexes.append([])
            self._node_in_edge_indexes.append([])

        self._edge_id_to_index = {}
        self._adjacency_matrix = {}
        self._dist_matrix = {}
        self._pred_matrix = {}
        self._max_edge_cur_cost = 0
        for i in range(0, self._edge_count):
            edge = self._edge_list[i]
            self.add_edge(edge, i)
            edge_id = int(edge['id'])
            if edge_id not in self._edge_id_to_index:
                self._edge_id_to_index[edge_id] = i
        self.update_current_edge_cost()

    def add_edge(self, edge, edge_index):
        edge_id = int(edge['id'])
        from_node = int(edge['from_node'])
        to_node = int(edge['to_node'])

        if from_node not in self._node_id_to_node_index:
            print("WARNING: start/end node not found for edge:" + str(edge_id))
        else:
            from_index = self._node_id_to_node_index[from_node]
            self._node_out_edge_indexes[from_index].append(edge_index)

        if to_node not in self._node_id_to_node_index:
            print("WARNING: end node not found for edge:" + str(edge_id))
        else:
            to_index = self._node_id_to_node_index[to_node]
            self._node_in_edge_indexes[to_index].append(edge_index)

        if 'cur_cost' not in edge:
            cur_cost = compute_edge_traffic_cost(
                edge['cost'],
                edge['alpha'],
                edge['beta'],
                edge['bg_volume'],
                edge['capacity'])
        else:
            cur_cost = edge['cur_cost']
        self._max_edge_cur_cost = max(self._max_edge_cur_cost, cur_cost)

        if from_node not in self._adjacency_matrix:
            self._adjacency_matrix[from_node] = {}
        self._adjacency_matrix[from_node][to_node] = {
            'free_cost': edge['cost'],
            'alpha': edge['alpha'],
            'beta': edge['beta'],
            'capacity': edge['capacity'],
            'bg_volume': edge['bg_volume'],
            'social_volume': 0.0,
            'cur_cost': cur_cost
        }

    def update_current_edge_cost(self, node_pair_cost_list=None):
        if node_pair_cost_list is not None:
            self._max_edge_cur_cost = self.update_edge_info(node_pair_cost_list, 'cur_cost')
        max_path_len = self._edge_count * self._max_edge_cur_cost
        self._dist_matrix, self._pred_matrix = all_pairs_shortest_path(self._adjacency_matrix, max_path_len)

    def reset_social_volume(self):
        for from_node in self._adjacency_matrix:
            for to_node in self._adjacency_matrix[from_node]:
                self._adjacency_matrix[from_node][to_node]['social_volume'] = 0

    def update_edge_info(self, pair_wise_edge_info, value_type, overwrite=True):
        max_value = 0
        for node_pair_cost in pair_wise_edge_info:
            from_node = int(node_pair_cost['from_node'])
            to_node = int(node_pair_cost['to_node'])
            if from_node not in self._adjacency_matrix or to_node not in self._adjacency_matrix[from_node]:
                print("Warning: this node pair not in original graph: " + str(node_pair_cost))
                continue
            max_value = max(max_value, node_pair_cost[value_type])
            if overwrite:
                self._adjacency_matrix[from_node][to_node][value_type] = node_pair_cost[value_type]
            else:
                self._adjacency_matrix[from_node][to_node][value_type] += node_pair_cost[value_type]

        return max_value

    def add_routed_car_to_social_volume(self, path):
        if path is None:
            return None

        generated_volume = []
        for i in range(0, len(path)):
            if (i + 1) < len(path):
                generated_volume.append(
                    {
                        "from_node": path[i],
                        'to_node': path[i + 1],
                        'social_volume': 1.0
                    }
                )

        self.update_edge_info(generated_volume, 'social_volume', overwrite=False)

    def route_one_car_greedily(self, src_id, dst_id):
        path = construct_shortest_path(int(src_id), int(dst_id), self._pred_matrix)
        self.add_routed_car_to_social_volume(path)

    def route_one_car_socially(self, src_id, dst_id, altruism_factor=1.0):
        path = dijkstra_shortest_path(
            self._adjacency_matrix,
            int(src_id),
            int(dst_id),
            self._max_edge_cur_cost * self._edge_count,
            False,
            altruism_factor)
        path.insert(0, src_id)
        self.add_routed_car_to_social_volume(path)

    def compute_social_routing_cost(self):
        total_cost = 0.0
        for from_node in self._adjacency_matrix:
            for end_node in self._adjacency_matrix[from_node]:
                total_cost += self._adjacency_matrix[from_node][end_node]['social_volume'] * \
                              get_routing_cost(self._adjacency_matrix[from_node][end_node], False, 0)
        return total_cost

    # generate the basic A x = b constraint then call the filter_constraints function to simplify A and b
    def generate_constraint_matrix(self, demand_list, remove_redundant):
        # demand_list = [{'from'= from_node_id, 'to'= to_node_id, 'volume'=v}, ...]
        # return A, b, and unknown_variables, variable_values including those p_i(e) = 0 before calculation

        # A columns from p_0(e_0), ... p_0(e_E-1), p_1..., ... p_n-1, f(e_0), ... f(e_E-1)
        # b rows from (d_0, node_0),...,(d_0, node_V-1), d_1 ..., 0s for sum p_i(e) - f(e) = 0
        demand_count = len(demand_list)
        variable_count = demand_count * self._edge_count + self._edge_count

        A = []
        b = []
        # A and b for p_i(e)
        for demand_index in range(0, demand_count):
            demand = demand_list[demand_index]
            source_index = self._node_id_to_node_index[demand["from"]]
            dest_index = self._node_id_to_node_index[demand['to']]

            b_demand = [0] * self._node_count
            b_demand[source_index] = -1.
            b_demand[dest_index] = 1.

            A_demand = []
            for node_index in range(0, self._node_count):
                a_row = [0] * variable_count
                for in_edge_index in self._node_in_edge_indexes[node_index]:
                    a_row[demand_index * self._edge_count + in_edge_index] = 1.
                for out_edge_index in self._node_out_edge_indexes[node_index]:
                    a_row[demand_index * self._edge_count + out_edge_index] = -1.
                A_demand.append(a_row)

            A.extend(A_demand)
            b.extend(b_demand)

        # A and b for f(e)
        b.extend([0] * self._edge_count)
        for edge_index in range(0, self._edge_count):
            a_row = [0] * variable_count
            for demand_index in range(0, demand_count):
                a_row[demand_index * self._edge_count + edge_index] = 1.0 * float(demand_list[demand_index]['volume'])
            a_row[demand_count * self._edge_count + edge_index] = -1.
            A.append(a_row)

        return self.filter_constraints(demand_list, A, b, remove_redundant)

    def filter_constraints(self, demand_list, A, b, remove_redundant, remove_predefined=False):
        demand_count = len(demand_list)
        variable_count = demand_count * self._edge_count + self._edge_count
        unknown_variables = [1] * variable_count
        variable_predefined_values = [0] * variable_count

        # remove redundant constraints for each demand (to make A a full rank matrix)
        redundant_rows = []
        if remove_redundant:
            for demand_index in range(0, demand_count):
                demand_constraint_matrix = A[demand_index * self._node_count:(demand_index + 1) * self._node_count]
                rank = numpy.linalg.matrix_rank(numpy.matrix(demand_constraint_matrix))
                for demand_row in range(demand_index * self._node_count + rank, (demand_index + 1) * self._node_count):
                    redundant_rows.append(demand_row)

        # set some unknown_variables[i] = 0 based on the route length/shortest length,
        # therefore, the predefined_values[i] = 0

        # iteratively remove pre_defined variables in A and b
        rows_including_unknown_vars = list(range(0, len(A)))
        if remove_predefined:
            while remove_predefined:
                rows_including_unknown_vars = []
                pre_defined_columns = []
                for variable_index in range(0, variable_count):
                    variable_validity = unknown_variables[variable_index]
                    if variable_validity == 0:
                        pre_defined_columns.append(variable_index)

                induced_variables = []
                for row_index in range(0, len(A)):
                    a_row = A[row_index]
                    non_zero_count = 0
                    induced_variable_index = -1
                    accumulative_value = 0.0
                    for variable_index in range(0, variable_count):
                        coef = a_row[variable_index]
                        if variable_index in pre_defined_columns:
                            accumulative_value += coef * variable_predefined_values[variable_index]

                        if variable_index not in pre_defined_columns and coef != 0:
                            non_zero_count += 1
                            induced_variable_index = variable_index

                    if non_zero_count > 1:
                        rows_including_unknown_vars.append(row_index)
                    else:
                        coef = a_row[induced_variable_index]
                        variable_value = (b[row_index] - accumulative_value) / coef
                        variable_predefined_values[induced_variable_index] = variable_value
                        induced_variables.append(induced_variable_index)
                if len(induced_variables) == 0:
                    break
                else:
                    for induced_variable_index in induced_variables:
                        unknown_variables[induced_variable_index] = 0

            unknown_variable_columns = []
            for variable_index in range(0, variable_count):
                variable_not_predefined = unknown_variables[variable_index]
                if variable_not_predefined == 1:
                    unknown_variable_columns.append(variable_index)

            adjust_b = [0] * len(b)
            for row_index in range(0, len(b)):
                a_row = A[row_index]
                accumulative_value = 0.0
                for variable_index in range(0, variable_count):
                    if variable_index not in unknown_variable_columns:
                        accumulative_value += a_row[variable_index] * variable_predefined_values[variable_index]
                adjust_b[row_index] = - accumulative_value
        else:
            adjust_b = [0] * len(b)
            unknown_variable_columns = list(range(0, variable_count))

        valid_rows = []
        for row in rows_including_unknown_vars:
            if row not in redundant_rows:
                valid_rows.append(row)

        filtered_A = [[A[i][j] for j in unknown_variable_columns] for i in valid_rows]
        filtered_b = [b[i] + adjust_b[i] for i in valid_rows]

        return filtered_A, filtered_b, unknown_variables, variable_predefined_values

