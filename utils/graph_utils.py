import networkx as nx
import numpy as np

def random_walk(graph, walk_length, start_node):
    """
    Parameters
    --------------

    graph : Graph
      networkx Graph to do random walk
    walk_length : int
      length of random walks
    start_node : Node
      starting point of random walk
    """

    walks = [start_node]

    for _ in range(walk_length):
        adj_node = [n for n in graph.neighbors(walks[-1])]

        if len(adj_node) == 0:
            # If no edges from node go back
            if len(walks) == 1:
                # If node is starting point, random walk fails
                return None
            walks.append(walks[-2])

        idx = np.random.randint(len(adj_node))
        walks.append(adj_node[idx])


    return walks

if __name__ == '__main__':
    G = nx.DiGraph()
    G.add_edges_from([
        ('a', 'b'), ('b', 'c'), ('a', 'c'), ('c', 'x'),
        ('x', 'y'), ('y', 'z'), ('x', 'z'), ('a', 'a')
    ])

    print([n for n in nx.neighbors(G, 'a')])

    print([n for n in G.nodes] * 2)