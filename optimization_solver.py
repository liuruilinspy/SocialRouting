from cvxopt import solvers, matrix, spdiag, log
from cvxopt import spmatrix


def routing_solver(A_listform, b_listform, unknown_variables, demand_list, edge_list, type='system',
                   maxiters=100, abstol=10**(-7), reltivetol=10**(-6), feastol=10**(-7)):
    A = matrix(A_listform).T
    b = matrix(b_listform)
    all_variable_count = (len(demand_list) + 1) * len(edge_list)
    routing_variable_count = len(demand_list) * len(edge_list)
    valid_routing_variable_count = 0
    flow_variable_to_edge_index = {}
    predefined_flow_variable_count = 0
    for index in range(0, all_variable_count):
        if index < routing_variable_count:
            if unknown_variables[index] == 1:
                valid_routing_variable_count += 1
        else:
            if unknown_variables[index] == 0:
                predefined_flow_variable_count += 1
            else:
                flow_variable_edge_index = index - routing_variable_count
                valid_flow_variable_index = flow_variable_edge_index - predefined_flow_variable_count
                flow_variable_to_edge_index[valid_routing_variable_count + valid_flow_variable_index] = \
                    flow_variable_edge_index

    linear_constraint_count, variable_count = A.size

    G = spmatrix(-1.0, range(0, variable_count), range(0, variable_count))
    #print(G)
    h = matrix([0.0] * variable_count)
    #print(h)
    dims = {'l': variable_count, 'q': [], 's': []}

    solvers.options['maxiters'] = maxiters
    solvers.options['abstol'] = abstol
    solvers.options['reltol'] = reltivetol
    solvers.options['feastol'] = feastol

    def sys_op_f(x=None, z=None):
        if x is None:
            return 0, matrix(1.0, (variable_count, 1))
        if min(x) < 0.0:
            return None

        # in our case, non-linear constraint m = 0, i.e., only f_0(x) = g_0(x_0) + g_i(x_i) + ... != 0
        # f(m+1)*1=1*1 f[0] = f_0(x) = g_0(x_0) + g_i(x_i) + ...
        f = 0
        for var_index in range(valid_routing_variable_count, variable_count):
            edge_index = flow_variable_to_edge_index[var_index]
            cost = edge_list[edge_index]['cost']
            capacity = edge_list[edge_index]['capacity']
            bg_volume = edge_list[edge_index]['bg_volume']
            alpha = edge_list[edge_index]['alpha']
            beta = edge_list[edge_index]['beta']
            #f += cost * (1 + alpha * ((bg_volume + x[var_index]) / capacity) ** beta)
            t = cost * (1 + alpha * ((bg_volume + x[var_index]) / capacity) ** beta)
            f += t * x[var_index]

        # Df(m+1)*n = 1*n f[0,:] = df_0/dx_i
        df_values = list()  # derivative towards each x_i
        ddf_values = list()  # second derivative towards each x_i
        for var_index in range(0, valid_routing_variable_count):
            df_values.append(0.0)
            ddf_values.append(0.0)

        for var_index in range(valid_routing_variable_count, variable_count):
            edge_index = flow_variable_to_edge_index[var_index]
            cost = edge_list[edge_index]['cost']
            capacity = edge_list[edge_index]['capacity']
            bg_volume = edge_list[edge_index]['bg_volume']
            alpha = edge_list[edge_index]['alpha']
            beta = edge_list[edge_index]['beta']
            t = cost * (1 + alpha * ((bg_volume + x[var_index]) / capacity) ** beta)
            dt = cost * alpha * beta * ((bg_volume + x[var_index] / capacity) ** (beta - 1)) / capacity
            dt2 = cost * alpha * beta * (beta - 1) * ((bg_volume + x[var_index] / capacity) ** (beta - 2)) / (capacity ** 2)
            df_values.append(x[var_index] * dt + t)
            ddf_values.append(x[var_index] * dt2 + 2 * dt)

        Df = matrix(df_values, (1, variable_count))
        ddf = matrix(ddf_values, (variable_count, 1))
        if z is None:
            return f, Df

        H = spdiag(z[0] * ddf)  # diagonal matrix, h[:i] = z[i] * f_i''(x)
        return f, Df, H

    def ue_f(x=None, z=None):
        if x is None:
            return 0, matrix(1.0, (variable_count, 1))
        if min(x) < 0.0:
            return None

        # in our case, non-linear constraint m = 0, i.e., only f_0(x) = g_0(x_0) + g_i(x_i) + ... != 0
        # f(m+1)*1=1*1 f[0] = f_0(x) = g_0(x_0) + g_i(x_i) + ...
        f = 0
        for var_index in range(valid_routing_variable_count, variable_count):
            edge_index = flow_variable_to_edge_index[var_index]
            cost = edge_list[edge_index]['cost']
            capacity = edge_list[edge_index]['capacity']
            bg_volume = edge_list[edge_index]['bg_volume']
            alpha = edge_list[edge_index]['alpha']
            beta = edge_list[edge_index]['beta']
            f += cost * x[var_index] + cost * alpha * capacity ** -beta / (beta + 1) * \
                                       ((bg_volume + x[var_index]) ** (beta + 1) - bg_volume ** (beta + 1))

        # Df(m+1)*n = 1*n f[0,:] = df_0/dx_i
        df_values = list()  # derivative towards each x_i
        ddf_values = list()  # second derivative towards each x_i
        for var_index in range(0, valid_routing_variable_count):
            df_values.append(0.0)
            ddf_values.append(0.0)

        for var_index in range(valid_routing_variable_count, variable_count):
            edge_index = flow_variable_to_edge_index[var_index]
            cost = edge_list[edge_index]['cost']
            capacity = edge_list[edge_index]['capacity']
            bg_volume = edge_list[edge_index]['bg_volume']
            alpha = edge_list[edge_index]['alpha']
            beta = edge_list[edge_index]['beta']
            t = cost * (1 + alpha * ((bg_volume + x[var_index]) / capacity) ** beta)
            dt = cost * alpha * beta * ((bg_volume + x[var_index] / capacity) ** (beta - 1)) / capacity
            df_values.append(t)
            ddf_values.append(dt)

        Df = matrix(df_values, (1, variable_count))
        ddf = matrix(ddf_values, (variable_count, 1))
        if z is None:
            return f, Df

        H = spdiag(z[0] * ddf)  # diagonal matrix, h[:i] = z[i] * f_i''(x)
        return f, Df, H

    def social_op_f(x=None, z=None):
        if x is None:
            return 0, matrix(1.0, (variable_count, 1))
        if min(x) < 0.0:
            return None

        # in our case, non-linear constraint m = 0, i.e., only f_0(x) = g_0(x_0) + g_i(x_i) + ... != 0
        # f(m+1)*1=1*1 f[0] = f_0(x) = g_0(x_0) + g_i(x_i) + ...
        f = 0
        for var_index in range(valid_routing_variable_count, variable_count):
            edge_index = flow_variable_to_edge_index[var_index]
            cost = edge_list[edge_index]['cost']
            capacity = edge_list[edge_index]['capacity']
            bg_volume = edge_list[edge_index]['bg_volume']
            alpha = edge_list[edge_index]['alpha']
            beta = edge_list[edge_index]['beta']
            integral_t = cost * x[var_index] + cost * alpha * capacity ** -beta / (beta + 1) * \
                                       ((bg_volume + x[var_index]) ** (beta + 1) - bg_volume ** (beta + 1))
            integral_xdt = cost * alpha * capacity ** (-beta) * (x[var_index] * (bg_volume + x[var_index]) ** beta -
                            1 / (beta + 1) * ((bg_volume + x[var_index]) ** (beta + 1) - bg_volume ** (beta + 1)))
            f += integral_t + integral_xdt

        # Df(m+1)*n = 1*n f[0,:] = df_0/dx_i
        df_values = list()  # derivative towards each x_i
        ddf_values = list()  # second derivative towards each x_i
        for var_index in range(0, valid_routing_variable_count):
            df_values.append(0.0)
            ddf_values.append(0.0)

        for var_index in range(valid_routing_variable_count, variable_count):
            edge_index = flow_variable_to_edge_index[var_index]
            cost = edge_list[edge_index]['cost']
            capacity = edge_list[edge_index]['capacity']
            bg_volume = edge_list[edge_index]['bg_volume']
            alpha = edge_list[edge_index]['alpha']
            beta = edge_list[edge_index]['beta']
            t = cost * (1 + alpha * ((bg_volume + x[var_index]) / capacity) ** beta)
            dt = cost * alpha * beta * ((bg_volume + x[var_index] / capacity) ** (beta - 1)) / capacity
            dt2 = cost * alpha * beta * (beta - 1) * ((bg_volume + x[var_index] / capacity) ** (beta - 2)) / (capacity ** 2)
            df_values.append(t + x[var_index] * dt)
            ddf_values.append(2 * dt + x[var_index] * dt2)

        Df = matrix(df_values, (1, variable_count))
        ddf = matrix(ddf_values, (variable_count, 1))
        if z is None:
            return f, Df

        H = spdiag(z[0] * ddf)  # diagonal matrix, h[:i] = z[i] * f_i''(x)
        return f, Df, H

    if type == 'ue':
        planning_results = solvers.cp(ue_f, G=G, h=h, dims=dims, A=A, b=b)['x']
    elif type == 'social':
        planning_results = solvers.cp(social_op_f, G=G, h=h, dims=dims, A=A, b=b)['x']
    else:
        planning_results = solvers.cp(sys_op_f, G=G, h=h, dims=dims, A=A, b=b)['x']

    total_cost = 0.0
    for flow_variable_index, edge_index in flow_variable_to_edge_index.items():
        flow_variable = planning_results[flow_variable_index]
        cost = edge_list[edge_index]['cost']
        capacity = edge_list[edge_index]['capacity']
        bg_volume = edge_list[edge_index]['bg_volume']
        alpha = edge_list[edge_index]['alpha']
        beta = edge_list[edge_index]['beta']
        t = cost * (1 + alpha * ((bg_volume + flow_variable) / capacity) ** beta)
        total_cost += t * flow_variable

    return planning_results[0:valid_routing_variable_count], total_cost


def test_solver(A, b):
    linear_constraint_count, variable_count = A.size

    def F(x=None, z=None):
        if x is None:
            return 0, matrix(1.0, (variable_count, 1))
        if min(x) < 0.0:
            return None

        # in our case, non-linear constraint m = 0, i.e., only f_0(x) = g_0(x_0) + g_i(x_i) + ... != 0
        # f(m+1)*1=1*1 f[0] = f_0(x) = g_0(x_0) + g_i(x_i) + ...
        f = -sum(log(x))
        Df = -(x ** -1).T
        if z is None: return f, Df
        H = spdiag(z[0] * x ** -2)
        return f, Df, H

    G = spmatrix(-1.0, range(0, variable_count), range(0, variable_count))
    #print(G)
    h = matrix([0.0] * variable_count)
    #print(h)
    dims = {'l': variable_count, 'q': [], 's': []}
    return solvers.cp(F, G=G, h=h, dims=dims, A=A, b=b)['x']