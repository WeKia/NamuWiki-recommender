import time
import numpy as np
from pympler import asizeof

def random_walk(graph, walk_length, start_node):
    """
    Parameters
    --------------

    graph : Graph or Dict(node, list(node))
      networkx Graph to do random walk
    walk_length : int
      length of random walks
    start_node : Node
      starting point of random walk
    """

    if isinstance(graph, dict):
        get_neigh = lambda x : graph[x]
    else:
        get_neigh = lambda x : graph.neighbors(x)

    walks = [start_node]

    for _ in range(walk_length):
        adj_node = [n for n in get_neigh(walks[-1])]

        if len(adj_node) == 0:
            # If no edges from node go back
            if len(walks) == 1:
                # If node is starting point, random walk fails
                return None
            walks.append(walks[-2])
            continue

        idx = np.random.randint(len(adj_node))
        walks.append(adj_node[idx])


    return walks

class CompactList:
    def __init__(self, blocksize, dtype=np.int32):
        """
        Using list allocate memory much more than numpy array.
        But numpy array is not dynamic and use it with append is slow.
        So make list block and if elements exceed block size, append it to np array
        :param blocksize: Size of list block
        """
        self.blocksize = blocksize
        self.list = []
        self.array = None
        self.dtype= dtype
        self.block_idx = 0

    def __len__(self):
        return len(self.list) + len(self.array)

    def __getitem__(self, idx):
        if idx >= self.block_idx * self.blocksize:
            idx -= self.block_idx * self.blocksize

            return self.list[idx]

        return self.array[idx]

    def append(self, data):
        self.list.append(data)

        if len(self.list) >= self.blocksize:
            self.block_idx += 1

            if self.array is not None:
                self.array = np.append(self.array, self.list, axis=0)
            else:
                self.array = np.array(self.list, dtype=self.dtype)
            self.list = []

if __name__ == '__main__':
    st = time.time()

    compact = CompactList(blocksize=4096)

    for i in range(1000000):
        compact.append(i)

    print(asizeof.asizeof(compact))
    print(time.time() - st)
