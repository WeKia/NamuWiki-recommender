# Implementation Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba
import argparse
import numpy as np
import networkx as nx
import pandas as pd
import os
import time
import multiprocessing as mp
import parmap
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils.graph_utils import random_walk
from utils.tokenization_kobert import KoBertTokenizer
from transformers import DistilBertModel

parser = argparse.ArgumentParser()
parser.add_argument('--max_length', type=int, help='Maximum length of each categories', default=64)
parser.add_argument('--max_category', type=int, help='Maximum number of categories', default=45)
parser.add_argument('--info_path', type=str, help='Path to documents information csv file')
parser.add_argument('--graph_path', type=str, help='Path to graph file if not exist make new file', default='Graph.gz')
parser.add_argument('--num_process', type=int, help='Number of process that processing graph parse', default=-1)

args = parser.parse_args()


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

def Make_pairs(nodes, G, l, k, ns):

    pairs = []
    labels = []

    node_num = [int(n) for n in nodes]
    node_num.sort()

    # To sampling negatives, we need to get not connected nodes
    # Calculating them is slow when using setdiff1d
    # So make mask and do bit-wise compare

    max_nodelen = node_num[-1]
    mask = np.zeros((max_nodelen + 1,))
    mask[node_num] = 1

    # Make mask to bool type
    mask = mask > 0

    arr = np.array(range(max_nodelen + 1))

    for v in nodes:
        # start = time.time()
        walks = random_walk(G, l, v)
        # print(f"afte random_walk : {time.time() - start}")
        # start = time.time()

        for i, walk_node in enumerate(walks):
            iter = 0
            for j in range(max(0, i - k), min(l - 1, i + k) + 1):
                if i == j: continue

                u = walks[j]
                pairs.append((walk_node, u))
                labels.append(1)
                iter += 1

            # start = time.time()

            neighbors = [int(n) for n in G.neighbors(walk_node)] + [int(walk_node)]
            mask[neighbors] = False

            negative = arr[mask]

            # print(len(nodes))
            # print(len(negative))
            # print(neighbors)

            # Restore
            mask[neighbors] = True

            # In paper, Do sampling in each loops but that is time consuming
            # So in my code, do sampling at once

            # np.choice is slow
            # negative_samples = np.random.choice(list(negative), ns * iter)

            idx = np.random.randint(len(negative), size=ns * iter)
            negative_samples = [negative[i] for i in idx]

            for neg in negative_samples:
                pairs.append((walk_node, neg))
                labels.append(0)

            # print(f"afte one node process : {time.time() - start}")
            # start = time.time()

    return pairs, labels

class GraphDataset(Dataset):
    def __init__(self, G, w, l, k, ns, side_info):
        """
        EGES Dataset created by Graph

        :param G: networkx Graph to make dataset
        :type G: Graph

        :param w: number of walks per node
        :type w: int

        :param l: walk length
        :type l: int

        :param k: Skip-Gram window size k
        :type k: int

        :param ns: number of negative sampling
        :type ns: int

        :param side_info: side information of each nodes
        :type side_info: dict(node, string)
        """

        self.G = G
        self.side_info = {}
        self.pairs = []
        self.labels = []

        tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')

        # To use array, every side informations must be same shape.
        padding = [1 for _ in range(args.max_length)]
        pad_mask = [0 for _ in range(args.max_length)]

        for k, v in side_info.items():
            assert len(v) <= args.max_category

            dict = {"input_ids": [], "attention_mask": []}

            input_ids = []
            attention_masks = []

            for category in v:
                token = tokenizer(category, padding='max_length', max_length=args.max_length, truncation=True)

                input_ids.append(token["input_ids"])
                attention_masks.append(token["attention_mask"])

            for _ in range(args.max_category - len(v)):
                input_ids.append(padding)
                attention_masks.append(pad_mask)

            dict["input_ids"] = input_ids
            dict["attention_mask"] = attention_masks

            self.side_info[k] = dict
        #

        if args.num_process == -1:
            proc_num = mp.cpu_count() - 1
        else:
            proc_num = args.num_process

        nodes = [n for n in self.G.nodes] * w

        result = parmap.map(Make_pairs, np.array_split(nodes, proc_num), self.G, l, k, ns, pm_pbar=True, pm_processes=proc_num)

        # process_bar = tqdm(total=len(G.nodes) * w)
        # nodes = [int(n) for n in self.G.nodes]
        # nodes.sort()
        #
        # # To sampling negatives, we need to get not connected nodes
        # # Calculating them is slow when using setdiff1d
        # # So make mask and do bit-wise compare
        #
        # max_nodelen = nodes[-1]
        # mask = np.zeros((max_nodelen + 1,))
        # mask[nodes] = 1
        #
        # # Make mask to bool type
        # mask = mask > 0
        #
        # arr = np.array(range(max_nodelen + 1))
        #
        # for _ in range(w):
        #     for v in self.G.nodes:
        #         # start = time.time()
        #         walks = random_walk(self.G, l, v)
        #         # print(f"afte random_walk : {time.time() - start}")
        #         # start = time.time()
        #
        #         for i, walk_node in enumerate(walks):
        #             iter = 0
        #             for j in range(max(0, i - k), min(l - 1, i + k) + 1):
        #                 if i == j: continue
        #
        #                 u = walks[j]
        #                 self.pairs.append((walk_node, u))
        #                 self.labels.append(1)
        #                 iter += 1
        #
        #             # start = time.time()
        #
        #             neighbors = [int(n) for n in self.G.neighbors(walk_node)] + [int(walk_node)]
        #             mask[neighbors] = False
        #
        #             negative = arr[mask]
        #
        #             # print(len(nodes))
        #             # print(len(negative))
        #             # print(neighbors)
        #
        #             # Restore
        #             mask[neighbors] = True
        #
        #             # In paper, Do sampling in each loops but that is time consuming
        #             # So in my code, do sampling at once
        #
        #             # np.choice is slow
        #             # negative_samples = np.random.choice(list(negative), ns * iter)
        #
        #             idx = np.random.randint(len(negative), size=ns * iter)
        #             negative_samples = [negative[i] for i in idx]
        #
        #             for neg in negative_samples:
        #                 self.pairs.append((walk_node, neg))
        #                 self.labels.append(0)
        #
        #             # print(f"afte one node process : {time.time() - start}")
        #             # start = time.time()
        #
        #         process_bar.update(1)
        # process_bar.close()

    def __getitem__(self, i):
        node, context_node = self.pairs[i]
        side_info = self.side_info[node]

        return node, context_node, side_info, self.labels[i]

    def __len__(self):
        return len(self.pairs)

    def save_data(self, path):
        pass

def Make_Graph(docs):
    """
    Make Graph from documents link

    :param args: pair_path path to csv file
    :param docs: documents information Pandas
    :return Grpah:
    """
    docs['links'] = docs.links.apply(eval)

    G = nx.DiGraph()

    print(f'total documents : {len(docs)}')

    for index, row in tqdm(docs.iterrows(), total=len(docs)):
        node = row['id']

        # Add edges from node to links.
        for to in row['links']:
            # Do not allow edge to itself
            if (node == to) : continue
            G.add_edges_from([(node, to)])

    nx.write_edgelist(G, args.graph_path)

    return G

def main():
    docs = pd.read_csv(args.info_path)
    docs['category'] = docs.category.apply(eval)

    side_info = dict(zip(docs.id, docs.category))

    if not os.path.exists(args.graph_path):
        print("Graph file not exists! process pairs to graph...")
        G = Make_Graph(docs)
    else:
        G = nx.read_edgelist(args.graph_path)

    print(f"Nuber of nodes : {len(G.nodes)}")
    print(f"Nuber of edges : {len(G.edges)}")

    dataset = GraphDataset(G, 2, 10, 2, 2, side_info)



if __name__ == '__main__':
    main()