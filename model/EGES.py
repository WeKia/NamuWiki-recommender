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
from transformers import ElectraModel, ElectraTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--max_length', type=int, help='Maximum length of each categories', default=64)
parser.add_argument('--max_category', type=int, help='Maximum number of categories', default=16)
parser.add_argument('--info_path', type=str, help='Path to documents information csv file')
parser.add_argument('--graph_path', type=str, help='Path to graph file if not exist make new file', default='data/Graph.gz')
parser.add_argument('--pass_graph', action='store_true', default=False, help='Pass loading graph. Use it if data already processed')
parser.add_argument('--test', action='store_true', default=False, help='Try load only 1000 nodes')
parser.add_argument('--data_path', type=str, help='Path to preprocessed data file', default=None)
parser.add_argument('--num_process', type=int, help='Number of process that processing graph parse', default=-1)
parser.add_argument('--base_path', type=str, help='Base path to running directory', default='./')
parser.add_argument('--epochs', type=int, help='Epochs for learning', default=2)
parser.add_argument('--lr', type=float, help="Learning rate for model", default=0.0005)
parser.add_argument('--max_grad_norm', type=float, help='max_grad_norm for model', default=0.5)
parser.add_argument('--log_interval', type=int, help='Log interval for model training', default=1000)
parser.add_argument('--device', type=str, help='Device for training model', default='cpu')
parser.add_argument('--output', type=str, help='Output path for embedding vectors', default='embed.npy')

args = parser.parse_args()


class EGES(nn.Module):
    def __init__(self, node_len, embed_dim, fine_tune=False):
        super(EGES, self).__init__()
        self.NodeEmbedding = nn.Embedding(node_len, embed_dim)
        self.SideEmbedding = ElectraModel.from_pretrained("monologg/koelectra-small-v3-discriminator")
        if not fine_tune:
            for param in self.SideEmbedding.parameters():
                param.requires_grad = False

        self.attention = torch.nn.Parameter(torch.rand((node_len, 2)))

    def forward(self, node, side_info, context_node):
        """
        param: node, Tensor,  [batch_size, 1]
        param: side_info, Tensor, [batch_size, max_category, max_length]
        param: context_node, Tensor, [batch_size, 1]
        """

        H = self.embed(node, side_info) #[batch_size, embed]
        context_embed = self.NodeEmbedding(context_node) #[batch_size, embed]

        output = torch.bmm(H.unsqueeze(1), context_embed.unsqueeze(2)).squeeze() # batch-wise dot

        return output

    def embed(self, node, side_info):
        """
        param: node, Tensor,  [batch_size, 1]
        param: side_info, Tensor, [batch_size, max_category, max_length]
        """

        batch_size = node.shape[0]

        Node_embed = self.NodeEmbedding(node) # [batch_size, embed_dim]
        attention = torch.exp(self.attention[node]) # [batch_size, 2], exp(a_j)

        input_ids = side_info['input_ids'].reshape(-1, args.max_length).long() # [batch_size * max_category, 1, max_length]
        attention_mask = side_info['attention_mask'].reshape(-1, args.max_length).long() # [batch_size * max_category, 1, max_length]

        side_embed = self.SideEmbedding(input_ids=input_ids, attention_mask=attention_mask)[0][:, 0]
        side_embed = side_embed.reshape(batch_size, args.max_category, -1) # [batch_size, max_category, embed]

        # remove Nan
        is_nan = ~(attention_mask.reshape(batch_size, max_category, -1).bool().any(dim=2))
        side_embed =  torch.masked_fill(side_embed, torch.isnan(side_embed), 0)

        # Calculate Mean excluding nan values, Same with nanmean function in pytorch 1.9
        side_mean = torch.sum(side_embed, dim=1) / (~is_nan).sum(dim=1).unsqueeze(1)  # [batch_size, embed]

        embed = torch.cat([Node_embed.unsqueeze(1), side_mean.unsqueeze(1)], dim=1) #[batch_size, 2, embed]
        H = torch.bmm(attention.unsqueeze(1), embed).squeeze(1) / attention.sum(dim=1).unsqueeze(1)

        return H

    def save(self, nodes, infos):
        for i in nodes:
            self.embed(i, infos[i])


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
        self.node2idx = {}

        tokenizer = ElectraTokenizer.from_pretrained('monologg/koelectra-base-v3-discriminator')

        # To use array, every side informations must be same shape.
        padding = np.ones((1, args.max_length), dtype=np.int8)
        pad_mask = np.zeros((1, args.max_length), dtype=np.int8)

        for k, v in tqdm(side_info.items()):
            dict = {"input_ids": [], "attention_mask": []}

            input_ids = []
            attention_masks = []

            if len(v) > args.max_category:
                v = v[:args.max_category]

            for category in v:
                token = tokenizer(category, padding='max_length', max_length=args.max_length, truncation=True, return_tensors='np')

                # print(token)
                input_ids.append(token["input_ids"])
                attention_masks.append(token["attention_mask"])

            for _ in range(args.max_category - len(v)):
                input_ids.append(padding)
                attention_masks.append(pad_mask)

            dict["input_ids"] = np.concatenate(input_ids, axis=0)
            dict["attention_mask"] = np.concatenate(attention_masks, axis=0)

            self.side_info[k] = dict


        if pair_path is None or (not os.path.exists(pair_path)):
            print("Preprocessed pair data not found!")
            self.process_pairs()
            self.save(pair_path)
        else:
            self.load(pair_path)

    def process_pairs(self):
        process_bar = tqdm(total=len(self.G.nodes) * self.w)
        nodes = [int(n) for n in self.G.nodes]
        nodes.sort()

        self.node2idx = {n: i for i, n in enumerate(nodes)}

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

        nodes = set()
        with open(os.path.join(path), 'r') as f:
            for line in tqdm(f, total=122718024):
                line = line.strip()
                node, context, label = line.split(' ')
                self.nodes.append(int(node))
                self.contexts.append(int(context))
                self.labels.append(int(label))

                # self.contexts.append(int(context))
                # self.labels.append(int(label))

                nodes.update([int(node), int(context)])
        nodes = list(nodes)
        nodes.sort()

        self.node2idx = {n: i for i, n in enumerate(nodes)}

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

class EmbeddingDataset(Dataset):
  def __init__(self, nodes, side_info):
    self.side_info = side_info
    self.nodes = nodes

  def __getitem__(self, idx):
    node = self.nodes[idx]
    info = self.side_info[node]

    return node, info

  def __len__(self):
    return len(self.nodes)

def Save_Embed(model, nodes, side_info):
    embed_data = EmbeddingDataset(nodes, side_info)

    embedLoader = DataLoader(embed_data, batch_size=512)

    embedVec = None

    model.eval()
    for nodes, infos in tqdm(embedLoader):
        nodes = nodes.to(args.device)
        infos['input_ids'] = infos['input_ids'].to(args.device)
        infos['attention_mask'] = infos['attention_mask'].to(args.device)

        embed = model.embed(nodes, infos).data.cpu().numpy()

        if embedVec is None:
            embedVec = embed
        else:
            embedVec = np.concatenate([embedVec, embed], axis=0)

    with open(args.base_path + args.output, 'wb') as f:
        np.save(f, embedVec)

def main():
    docs = pd.read_csv(args.base_path + args.info_path)
    docs['category'] = docs.category.apply(eval)

    G = None
    node_len = 0

    if not args.pass_graph:
        if not os.path.exists(args.base_path + args.graph_path):
            print("Graph file not exists! process pairs to graph...")
            G = Make_Graph(docs)
        else:
            G = nx.read_edgelist(args.base_path + args.graph_path)

        node_len = len(G.nodes)
        print(f"Nuber of nodes : {node_len}")
        print(f"Nuber of edges : {len(G.edges)}")

    if args.test:
        if G is not None:
            nodes = [n for n in G.nodes]
            nodes.sort(key=lambda x : int(x))

            G = nx.subgraph(G, nodes[:1000])

        docs = docs[docs['id'] < 1000]

        print(len(docs))
        node_len = 1000

    device = args.device

    side_info = dict(zip(docs.id, docs.category))

    dataset = GraphDataset(G, 2, 10, 2, 2, side_info, args.base_path + args.data_path)

    model = EGES(node_len, 256)
    model.to(device)

    dataloader = DataLoader(dataset, batch_size=128)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        model.train()
        i = 0
        for nodes, contexts, labels, infos in tqdm(dataloader):
            nodes = nodes.to(device)
            contexts = contexts.to(device)
            labels = labels.to(device).float()
            infos['input_ids'] = infos['input_ids'].to(device)
            infos['attention_mask'] = infos['attention_mask'].to(device)

            embed = model(nodes, infos, contexts)

            optimizer.zero_grad()

            loss = loss_fn(embed, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            i += 1
            if (i % args.log_interval) == 0:
                print(f"Epoch {epoch} train loss {loss.data.cpu()}")


    nodes = None
    if args.test:
        nodes = range(1000)
    else:
        nodes = dataset.nodes

    Save_Embed(model, nodes, dataset.side_info)

if __name__ == '__main__':
    if args.base_path == './':
        args.base_path = '../'


    main()
