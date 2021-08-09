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

        walks.append(np.random.choice(adj_node))

    return walks

if __name__ == '__main__':
    G = nx.Graph()
    G.add_edges_from([
        ('a', 'b'), ('b', 'c'), ('a', 'c'), ('c', 'x'),
        ('x', 'y'), ('y', 'z'), ('x', 'z')
    ])

    print(random_walk(G, 10, 'a'))