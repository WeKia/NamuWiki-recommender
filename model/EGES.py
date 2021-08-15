# Implementation Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba
import argparse
import sys

import numpy as np
import networkx as nx
import pandas as pd
from functools import partial
import os
import time
import torch
import torch.nn as nn
from pympler.asizeof import asizeof
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils.graph_utils import random_walk, CompactList
from utils.tokenization_kobert import KoBertTokenizer
from transformers import DistilBertModel

parser = argparse.ArgumentParser()
parser.add_argument('--max_length', type=int, help='Maximum length of each categories', default=64)
parser.add_argument('--max_category', type=int, help='Maximum number of categories', default=45)
parser.add_argument('--info_path', type=str, help='Path to documents information csv file')
parser.add_argument('--graph_path', type=str, help='Path to graph file if not exist make new file', default='data/Graph.gz')
parser.add_argument('--pass_graph', action='store_true', default=False, help='Pass loading graph. Use it if data already processed')
parser.add_argument('--data_path', type=str, help='Path to preprocessed data file', default=None)
parser.add_argument('--num_process', type=int, help='Number of process that processing graph parse', default=-1)
parser.add_argument('--base_path', type=str, help='Base path to running directory', default='./')

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

class GraphDataset(Dataset):
    def __init__(self, G, w, l, k, ns, side_info, pair_path=None):
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
        self.nodes = CompactList(1000000, dtype=np.int32)
        self.contexts = CompactList(1000000, dtype=np.int32)
        self.labels = CompactList(1000000, dtype=np.int8)
        self.w = w
        self.l = l
        self.k = k
        self.ns = ns

        tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')

        # To use array, every side informations must be same shape.
        padding = [1 for _ in range(args.max_length)]
        pad_mask = [0 for _ in range(args.max_length)]

        for k, v in tqdm(side_info.items()):
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


        if pair_path is None or (not os.path.exists(pair_path)):
            self.process_pairs()
            self.save(pair_path)
        else:
            self.load(pair_path)

    def process_pairs(self):
        process_bar = tqdm(total=len(self.G.nodes) * self.w)
        nodes = [int(n) for n in self.G.nodes]
        nodes.sort()

        # To sampling negatives, we need to get not connected nodes
        # Calculating them is slow when using setdiff1d
        # So make mask and do bit-wise compare

        max_nodelen = nodes[-1]
        mask = np.zeros((max_nodelen + 1,))
        mask[nodes] = 1

        # Make mask to bool type
        mask = mask > 0

        arr = np.array(range(max_nodelen + 1))

        for _ in range(self.w):
            for v in nodes:
                walks = random_walk(self.G, self.l, v)

                for i, walk_node in enumerate(walks):
                    iter = 0
                    for j in range(max(0, i - self.k), min(self.l - 1, i + self.k) + 1):
                        if i == j: continue

                        u = walks[j]
                        self.nodes.append(walk_node)
                        self.contexts.append(u)
                        self.labels.append(1)
                        iter += 1

                    neighbors = [int(n) for n in self.G.neighbors(walk_node)] + [int(walk_node)]
                    mask[neighbors] = False

                    negative = arr[mask]

                    # Restore
                    mask[neighbors] = True

                    # In paper, Do sampling in each loops but that is time consuming
                    # So in my code, do sampling at once

                    # np.choice is slow
                    # negative_samples = np.random.choice(list(negative), ns * iter)

                    idx = np.random.randint(len(negative), size=self.ns * iter)
                    negative_samples = [negative[i] for i in idx]

                    for neg in negative_samples:
                        self.nodes.append(walk_node)
                        self.contexts.append(neg)
                        self.labels.append(0)

                process_bar.update(1)
        process_bar.close()

    def __getitem__(self, i):
        node, context_node = self.nodes[i], self.contexts[i]
        side_info = self.side_info[node]
        # side_info = None

        return node, context_node, self.labels[i], side_info

    def __len__(self):
        return len(self.nodes)

    def load(self, path):
        """
        load pairs, labels data from txt file
        :param path: data txt file
        """
        with open(os.path.join(path), 'r') as f:
            for line in tqdm(f, total=122718024):
                line = line.strip()
                node, context, label = line.split(' ')
                self.nodes.append(int(node))
                self.contexts.append(int(context))
                self.labels.append(int(label))

                # self.contexts.append(int(context))
                # self.labels.append(int(label))

    def save(self, path):
        """
        Save pairs, labels data to txt file
        :param path: output txt file
        """
        with open(os.path.join(path), 'w') as f:
            for i in range(len(self)):
                f.write('%s %s %s\n' % (self[i]))

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

    nx.write_edgelist(G, args.base_path + args.graph_path)

    return G

def main():
    docs = pd.read_csv(args.base_path + args.info_path)
    docs['category'] = docs.category.apply(eval)

    side_info = dict(zip(docs.id, docs.category))

    G = None

    if not args.pass_graph:
        if not os.path.exists(args.base_path + args.graph_path):
            print("Graph file not exists! process pairs to graph...")
            G = Make_Graph(docs)
        else:
            G = nx.read_edgelist(args.base_path + args.graph_path)

        print(f"Nuber of nodes : {len(G.nodes)}")
        print(f"Nuber of edges : {len(G.edges)}")

    dataset = GraphDataset(G, 2, 10, 2, 2, side_info, args.base_path + args.data_path)

    data = DataLoader(dataset, batch_size=4)

    for d in data:
        print(d)

if __name__ == '__main__':
    if args.base_path == './':
        args.base_path = '../'

    main()