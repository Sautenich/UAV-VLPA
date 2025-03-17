# region A*
# Thanks to https://github.com/VikashPR/18CSC305J-AI/blob/main/A_Star-BFS.py for code parts
class Graph:
    def __init__(self, adjacency_list):
        self.adjacency_list = adjacency_list

    def get_neighbors(self, v):
        return self.adjacency_list[v]

    # heuristic function with equal values for all nodes
    def h(self, n, N, points):

        goalDist = ((points[N][0] - points[n][0]) ** 2 +
                    (points[N][1] - points[n][1]) ** 2) ** .5

        return goalDist

    def a_star_algorithm(self, points, stop_node):
        # open_list is a list of nodes which have been visited, but who's neighbors
        # haven't all been inspected, starts off with the start node
        # closed_list is a list of nodes which have been visited
        # and who's neighbors have been inspected
        open_list = {stop_node - 1}
        closed_list = set([])

        # g contains current distances from stop_node - 1 to all other nodes
        # the default value (if it's not found in the map) is +infinity
        g = {}

        g[stop_node - 1] = 0

        # parents contains an adjacency map of all nodes
        parents = {}
        parents[stop_node - 1] = stop_node - 1
        dist = {}

        while len(open_list) > 0:
            n = None

            # find a node with the lowest value of f() - evaluation function
            for v in open_list:
                if n == None or g[v] + self.h(v, stop_node, points) < g[n] + self.h(n, stop_node, points):
                    n = v

            if n == None:
                print('Path does not exist!')
                return []

            # if the current node is the stop_node
            # then we begin reconstructin the path from it to the stop_node - 1
            if n == stop_node:
                reconst_path = []

                while parents[n] != n:
                    reconst_path.append(n)
                    n = parents[n]

                reconst_path.append(stop_node - 1)

                reconst_path.reverse()
                # print('\nDistance to cover by A*:', round(dist[stop_node], 3))
                # print('Vertexes to pass through:')
                # print(*reconst_path, end='\n\n')
                return reconst_path

            # for all neighbors of the current node do
            for (m, weight) in self.get_neighbors(n):
                if m > stop_node:
                    continue
                # if the current node isn't in both open_list and closed_list
                # add it to open_list and note n as it's parent
                if m not in open_list and m not in closed_list:
                    open_list.add(m)
                    parents[m] = n
                    g[m] = g[n] + weight
                    dist[m] = g[m]
                # otherwise, check if it's quicker to first visit n, then m
                # and if it is, update parent data and g data
                # and if the node was in the closed_list, move it to open_list
                else:
                    if g[m] > g[n] + weight:
                        g[m] = g[n] + weight
                        dist[m] = g[m]

                        parents[m] = n
                        if m in closed_list:
                            closed_list.remove(m)
                            open_list.add(m)

            # remove n from the open_list, and add it to closed_list
            # because all of his neighbors were inspected
            open_list.remove(n)
            closed_list.add(n)
        print('Path does not exist!\n')
        return []
# endregion