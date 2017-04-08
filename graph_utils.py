import heapq


# key are vertices; each edge has weight and that's encoded as well
# graph = {0: {1: {'free_cost': 1, 'alpha': 1, 'beta':1, 'capacity': 100, 'bg_volume': 1, 'social_volume': 1, 'cur_cost': 1},
#               4: ...},
#         1: ...,
#         ...
#       }


def compute_edge_traffic_cost(free_cost, alpha, beta, bg_volume, capacity, social_volume=0, altruism_factor=0):
    total_volume = bg_volume + social_volume
    cf = free_cost * (1 + alpha * (total_volume / capacity) ** beta)
    dcf = free_cost * alpha * (capacity ** -beta) * beta * (total_volume ** (beta - 1))
    return cf + altruism_factor * social_volume * dcf


def get_routing_cost(edge_info, use_cur_cost, altruism_factor):
    if use_cur_cost:
        return edge_info['cur_cost']

    free_cost = edge_info['free_cost']
    alpha = edge_info['alpha']
    beta = edge_info['beta']
    bg_volume = edge_info['bg_volume']
    capacity = edge_info['capacity']
    social_volume = edge_info['social_volume']
    return compute_edge_traffic_cost(free_cost, alpha, beta, bg_volume, capacity, social_volume, altruism_factor)


def dijkstra_shortest_path(g, src_id, dst_id, max_dist, use_cur_cost, altruism_factor=0):
    distances = {}  # Distance from start to node
    previous = {}  # Previous node in optimal path from source
    nodes = []  # Priority queue of all nodes in Graph

    for vertex in g:
        if vertex == src_id:  # Set root node as distance of 0
            distances[vertex] = 0
            heapq.heappush(nodes, [0, vertex])
        else:
            distances[vertex] = max_dist
            heapq.heappush(nodes, [max_dist, vertex])
        previous[vertex] = None

    while nodes:
        smallest = heapq.heappop(nodes)[1]  # Vertex in nodes with smallest distance in distances
        if smallest == dst_id:  # If the closest node is our target we're done so print the path
            path = []
            while previous[smallest] is not None:  # Traverse through nodes til we reach the root which is 0
                path.insert(0, smallest)
                smallest = previous[smallest]
            return path
        if distances[smallest] == max_dist:  # All remaining vertices are inaccessible from source
            break

        for neighbor in g[smallest]:  # Look at all the nodes that this vertex is attached to
            # Alternative path distance
            alt = distances[smallest] + get_routing_cost(g[smallest][neighbor], use_cur_cost, altruism_factor)
            if alt < distances[neighbor]:  # If there is a new shortest path update our priority queue (relax)
                distances[neighbor] = alt
                previous[neighbor] = smallest
                for n in nodes:
                    if n[1] == neighbor:
                        n[0] = alt
                        break
                heapq.heapify(nodes)
    return None


# All Pairs Shortest Path Implementation
# Return distance structure as computed
def all_pairs_shortest_path(g, max_dist, use_cur_cost=True):
    dist = {}
    pred = {}
    for u in g:
        dist[u] = {}
        pred[u] = {}
        for v in g:
            dist[u][v] = max_dist
            pred[u][v] = None

        dist[u][u] = 0
        pred[u][u] = None

        for v in g[u]:
            dist[u][v] = get_routing_cost(g[u][v], use_cur_cost, 0)
            pred[u][v] = u

    for mid in g:
        for u in g:
            for v in g:
                new_len = dist[u][mid] + dist[mid][v]
                if new_len < dist[u][v]:
                    dist[u][v] = new_len
                    pred[u][v] = pred[mid][v]

    return dist, pred


def construct_shortest_path(s, t, pred):
    """Reconstruct shortest path from s to t using information in pred"""
    path = [t]

    while t != s:
        t = pred[s][t]

        if t is None:
            return None
        path.insert(0, t)

    return path