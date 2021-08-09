# Implementation Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba
import argparse
import numpy as np
import networkx as nx
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import multiprocessing as mp
import matplotlib.pyplot as plt
from itertools import combinations
from utils.graph_utils import random_walk
from utils.tokenization_kobert import KoBertTokenizer
from transformers import DistilBertModel

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pad_num', type=str, help='Padding size of side information')
    parser.add_argument('--pair_path', type=str, help='Path to user-item pair csv file')
    parser.add_argument('--graph_path', type=str, help='Path to graph file if not exist make new file', default='Graph.gz')
    parser.add_argument('--num_process', type=int, help='Number of process that processing graph parse', default=-1)

    args = parser.parse_args()

    return args


class EGES(nn.Module):
    def __init__(self, G, embed_dim):
        self.G = G

        self.NodeEmbedding = nn.Embedding(len(self.G.nodes), embed_dim)
        self.tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
        self.SideEmbedding = DistilBertModel.from_pretrained('monologg/distilkobert')

        self.attention = torch.rand((len(G.nodes), 2), requires_grad=True)

    def forward(self, node, side_info, context_node):
        """
        param: node, Tensor,  [batch_size, 1]
        param: side_info, Tensor, [batch_size, ,]
        param: context_node, Tensor, [batch_size, 1]
        """

        H_node = self.embed(node, side_info)
        context_embed = self.NodeEmbedding(context_node)


        pass

    def embed(self, node, side_info):
        """
        param: node, Tensor,  [batch_size, 1]
        param: side_info, Tensor, [batch_size, pad_num, ]
        """

        Node_embed = self.NodeEmbedding(node)

class GraphDataset(Dataset):
    def __init__(self, G, w, l, k, ns, side_info):
        """
        Parameters
        --------------

        G : Graph
          networkx Graph to make dataset
        w : int
          number of walks per node
        l : int
          walk length
        ns : int
          number of negative sampling
        side_info : dict(node, string)
          side information of each nodes
        """

        self.G = G
        self.side_info = side_info
        self.pairs = []
        self.labels = []

        for _ in range(w):
            for v in self.G.nodes:
                walks = random_walk(self.G, l, v)

                for i, walk_node in enumerate(walks):
                    for j in range(max(0, i - k), min(l - 1, i +k) + 1):
                        if i == j: continue

                        u = walks[j]
                        self.pairs.append((walk_node, u))
                        self.labels.append(1)

                        for _ in range(ns):
                            neighbors = [n for n in self.G.neighbors(walk_node)] + [walk_node]
                            nodes = self.G.nodes

                            negative = set(nodes) - set(neighbors)
                            negative_sample = np.random.choice(list(negative))

                            self.pairs.append((walk_node, negative_sample))
                            self.labels.append(0)

    def __getitem__(self, i):
        node, context_node = self.pairs[i]
        side_info = self.side_info[node]

        return node, context_node, side_info, self.labels[i]

    def __len__(self):
        pass

def Find_Pairs(df):
    """
    Make possible pairs that one contributors contributed documents
    :param df:
    :return pairs:
    """
    pairs = []

    for index, row in df.iterrows():
        related = row['title']
        # Add every pairs of documents that same contributor joins.
        for pair in combinations(related, 2):
            pairs.append(pair)

    return pairs

def Make_Graph(args):
    """
    Make Graph from user-item pairs

    :param args: pair_path path to csv file
    :return Grpah:
    """

    if args.num_process == -1:
        proc_num = mp.cpu_count() - 1
    else:
        proc_num = args.num_process

    pairs = pd.read_csv(args.pair_path)

    G = nx.Graph()

    # make list by grouping contributors
    pairs = pairs.groupby('contributors').agg({'title' : lambda x : x.tolist()})

    print(pairs.head(5))

    print(f'total contributors : {len(pairs)}')
    pairs = pairs[:1000]

    #To remove duplicates, use Set object
    pair_list = set()

    for index, row in tqdm(pairs.iterrows(), total=len(pairs)):
        related = row['title']
        # Add every pairs of documents that same contributor joins.
        for pair in combinations(related, 2):
            G.add_edges_from([pair])

    # pool = mp.Pool(proc_num)
    #
    # res = pool.map(Find_Pairs, np.array_split(pairs[:10000], proc_num))
    #
    # pool.close()
    # pool.join()

    # pair = []
    #
    # for pair_list in res:
    #     pair += pair_list


    nx.write_edgelist(G, "Graph.gz")
    nx.draw(G)
    plt.show()

    return G

def main(args):

    if not os.path.exists(args.graph_path):
        print("Graph file not exists! process pairs to graph...")
        Make_Graph(args)
    else:
        G = nx.read_edgelist(args.graph_path)

        nx.write_gexf(G, 'Graph_gephi.gexf')


if __name__ == '__main__':
    args = get_parser()
    main(args)