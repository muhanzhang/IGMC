from __future__ import print_function
import numpy as np
import random
from tqdm import tqdm
import os, sys, pdb, math, time
from copy import deepcopy
import multiprocessing as mp
import networkx as nx
import argparse
import scipy.io as sio
import scipy.sparse as ssp
import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset
import warnings
warnings.simplefilter('ignore', ssp.SparseEfficiencyWarning)
cur_dir = os.path.dirname(os.path.realpath(__file__))
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


class MyDataset(InMemoryDataset):
    def __init__(self, data_list, root, transform=None, pre_transform=None):
        self.data_list = data_list
        super(MyDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = self.data_list

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        del self.data_list


class MyDynamicDataset(Dataset):
    def __init__(self, root, A, indices, labels, h, sample_ratio, max_nodes_per_hop, u_features, v_features, max_node_label, class_values):
        super(MyDynamicDataset, self).__init__(root)
        self.A = A
        self.indices = indices
        self.labels = labels
        self.h = h
        self.sample_ratio = sample_ratio
        self.max_nodes_per_hop = max_nodes_per_hop
        self.u_features = u_features
        self.v_features = v_features
        self.max_node_label = max_node_label
        self.class_values = class_values

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def _download(self):
        pass

    def _process(self):
        pass

    def __len__(self):
        return len(self.indices[0])

    def get(self, idx):
        i, j = self.indices[0][idx], self.indices[1][idx]
        g, n_labels, n_features = subgraph_extraction_labeling((i, j), self.A, self.h, self.sample_ratio, self.max_nodes_per_hop, self.u_features, self.v_features, self.class_values)
        g_label = self.labels[idx]
        return nx_to_PyGGraph(g, g_label, n_labels, n_features, self.max_node_label, self.class_values)

       
def nx_to_PyGGraph(g, graph_label, node_labels, node_features, max_node_label, class_values):
    # convert networkx graph to pytorch_geometric data format
    y = torch.FloatTensor([class_values[graph_label]])
    if len(g.edges()) == 0:
        i, j = [], []
    else:
        i, j = zip(*g.edges())
    edge_index = torch.LongTensor([i+j, j+i])
    edge_type_dict = nx.get_edge_attributes(g, 'type')
    edge_type = torch.LongTensor([edge_type_dict[(ii, jj)] for ii, jj in zip(i, j)])
    edge_type = torch.cat([edge_type, edge_type], 0)
    edge_attr = torch.FloatTensor(class_values[edge_type]).unsqueeze(1)  # continuous ratings, num_edges * 1
    x = torch.FloatTensor(one_hot(node_labels, max_node_label+1))
    if node_features is not None:
        if type(node_features) == list:
            # node features are only provided for target user and item
            u_feature, v_feature = node_features
        else:
            # node features are provided for all nodes
            x2 = torch.FloatTensor(node_features)
            x = torch.cat([x, x2], 1)

    data = Data(x, edge_index, edge_attr=edge_attr, y=y)
    data.edge_type = edge_type
    if type(node_features) == list:
        data.u_feature = torch.FloatTensor(u_feature).unsqueeze(0)
        data.v_feature = torch.FloatTensor(v_feature).unsqueeze(0)
    return data
    

def PyGGraph_to_nx(data):
    edges = list(zip(data.edge_index[0, :].tolist(), data.edge_index[1, :].tolist()))
    g = nx.from_edgelist(edges)
    g.add_nodes_from(range(len(data.x)))  # in case some nodes are isolated
    edge_types = {(u, v): data.edge_type[i].item() for i, (u, v) in enumerate(edges)}  # transform r back to rating label
    nx.set_edge_attributes(g, name='type', values=edge_types)
    node_types = dict(zip(range(data.num_nodes), torch.argmax(data.x, 1).tolist()))
    nx.set_node_attributes(g, name='type', values=node_types)
    g.graph['rating'] = data.y.item()
    return g


def links2subgraphs(
        A,
        train_indices, 
        val_indices, 
        test_indices, 
        train_labels, 
        val_labels, 
        test_labels, 
        h=1, 
        sample_ratio=1.0, 
        max_nodes_per_hop=None, 
        u_features=None, 
        v_features=None, 
        max_node_label=None, 
        class_values=None, 
        testing=False, 
        parallel=True):
    # extract enclosing subgraphs
    if max_node_label is None:  # if not provided, infer from graphs
        max_n_label = {'max_node_label': 0}

    def helper(A, links, g_labels):
        g_list = []
        if not parallel or max_node_label is None:
            with tqdm(total=len(links[0])) as pbar:
                for i, j, g_label in zip(links[0], links[1], g_labels):
                    g, n_labels, n_features = subgraph_extraction_labeling((i, j), A, h, sample_ratio, max_nodes_per_hop, u_features, v_features, class_values)
                    if max_node_label is None:
                        max_n_label['max_node_label'] = max(max(n_labels), max_n_label['max_node_label'])
                        g_list.append((g, g_label, n_labels, n_features))
                    else:
                        g_list.append(nx_to_PyGGraph(g, g_label, n_labels, n_features, max_node_label, class_values))
                    pbar.update(1)
        else:
            start = time.time()
            pool = mp.Pool(mp.cpu_count())
            results = pool.starmap_async(parallel_worker, [(g_label, (i, j), A, h, sample_ratio, max_nodes_per_hop, u_features, v_features, class_values) for i, j, g_label in zip(links[0], links[1], g_labels)])
            remaining = results._number_left
            pbar = tqdm(total=remaining)
            while True:
                pbar.update(remaining - results._number_left)
                if results.ready(): break
                remaining = results._number_left
                time.sleep(1)
            results = results.get()
            pool.close()
            pbar.close()
            end = time.time()
            print("Time eplased for subgraph extraction: {}s".format(end-start))
            print("Transforming to pytorch_geometric graphs...".format(end-start))
            g_list += [nx_to_PyGGraph(g, g_label, n_labels, n_features, max_node_label, class_values) for g_label, g, n_labels, n_features in tqdm(results)]
            del results
            end2 = time.time()
            print("Time eplased for transforming to pytorch_geometric graphs: {}s".format(end2-end))
        return g_list

    print('Enclosing subgraph extraction begins...')
    train_graphs = helper(A, train_indices, train_labels)
    if not testing:
        val_graphs = helper(A, val_indices, val_labels)
    else:
        val_graphs = []
    test_graphs = helper(A, test_indices, test_labels)

    if max_node_label is None:
        train_graphs = [nx_to_PyGGraph(*x, **max_n_label, class_values=class_values) for x in train_graphs]
        val_graphs = [nx_to_PyGGraph(*x, **max_n_label, class_values=class_values) for x in val_graphs]
        test_graphs = [nx_to_PyGGraph(*x, **max_n_label, class_values=class_values) for x in test_graphs]
    
    return train_graphs, val_graphs, test_graphs


def subgraph_extraction_labeling(ind, A, h=1, sample_ratio=1.0, max_nodes_per_hop=None, u_features=None, v_features=None, class_values=None):
    # extract the h-hop enclosing subgraph around link 'ind'
    dist = 0
    u_nodes, v_nodes = [ind[0]], [ind[1]]
    u_dist, v_dist = [0], [0]
    u_visited, v_visited = set([ind[0]]), set([ind[1]])
    u_fringe, v_fringe = set([ind[0]]), set([ind[1]])
    for dist in range(1, h+1):
        v_fringe, u_fringe = neighbors(u_fringe, A, True), neighbors(v_fringe, A, False)
        u_fringe = u_fringe - u_visited
        v_fringe = v_fringe - v_visited
        u_visited = u_visited.union(u_fringe)
        v_visited = v_visited.union(v_fringe)
        if sample_ratio < 1.0:
            u_fringe = random.sample(u_fringe, int(sample_ratio*len(u_fringe)))
            v_fringe = random.sample(v_fringe, int(sample_ratio*len(v_fringe)))
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(u_fringe):
                u_fringe = random.sample(u_fringe, max_nodes_per_hop)
            if max_nodes_per_hop < len(v_fringe):
                v_fringe = random.sample(v_fringe, max_nodes_per_hop)
        if len(u_fringe) == 0 and len(v_fringe) == 0:
            break
        u_nodes = u_nodes + list(u_fringe)
        v_nodes = v_nodes + list(v_fringe)
        u_dist = u_dist + [dist] * len(u_fringe)
        v_dist = v_dist + [dist] * len(v_fringe)
    subgraph = A[u_nodes, :][:, v_nodes]
    # remove link between target nodes
    subgraph[0, 0] = 0
    # construct nx graph
    g = nx.Graph()
    g.add_nodes_from(range(len(u_nodes)), bipartite='u')
    g.add_nodes_from(range(len(u_nodes), len(u_nodes)+len(v_nodes)), bipartite='v')
    u, v, r = ssp.find(subgraph)  # r is 1, 2... (rating labels + 1)
    r = r.astype(int)
    v += len(u_nodes)
    #g.add_weighted_edges_from(zip(u, v, r))
    g.add_edges_from(zip(u, v))

    edge_types = dict(zip(zip(u, v), r-1))  # transform r back to rating label
    nx.set_edge_attributes(g, name='type', values=edge_types)
    # get structural node labels
    node_labels = [x*2 for x in u_dist] + [x*2+1 for x in v_dist]

    # get node features
    if u_features is not None:
        u_features = u_features[u_nodes]
    if v_features is not None:
        v_features = v_features[v_nodes]
    node_features = None
    if False: 
        # directly use padded node features
        if u_features is not None and v_features is not None:
            u_extended = np.concatenate([u_features, np.zeros([u_features.shape[0], v_features.shape[1]])], 1)
            v_extended = np.concatenate([np.zeros([v_features.shape[0], u_features.shape[1]]), v_features], 1)
            node_features = np.concatenate([u_extended, v_extended], 0)
    if False:
        # use identity features (one-hot encodings of node idxes)
        u_ids = one_hot(u_nodes, A.shape[0]+A.shape[1])
        v_ids = one_hot([x+A.shape[0] for x in v_nodes], A.shape[0]+A.shape[1])
        node_ids = np.concatenate([u_ids, v_ids], 0)
        #node_features = np.concatenate([node_features, node_ids], 1)
        node_features = node_ids
    if True:
        # only output node features for the target user and item
        if u_features is not None and v_features is not None:
            node_features = [u_features[0], v_features[0]]

    return g, node_labels, node_features


def parallel_worker(g_label, ind, A, h=1, sample_ratio=1.0, max_nodes_per_hop=None, u_features=None, v_features=None, class_values=None):
    g, node_labels, node_features = subgraph_extraction_labeling(ind, A, h, sample_ratio, max_nodes_per_hop, u_features, v_features, class_values)
    return g_label, g, node_labels, node_features

    
def neighbors(fringe, A, row=True):
    # find all 1-hop neighbors of nodes in fringe from A
    res = set()
    for node in fringe:
        if row:
            _, nei, _ = ssp.find(A[node, :])
        else:
            nei, _, _ = ssp.find(A[:, node])
        nei = set(nei)
        res = res.union(nei)
    return res


def one_hot(idx, length):
    idx = np.array(idx)
    x = np.zeros([len(idx), length])
    x[np.arange(len(idx)), idx] = 1.0
    return x


# Below are copied from newer versions of pytorch_geometric
from torch_sparse import coalesce
def filter_adj(row, col, edge_attr, mask):
    return row[mask], col[mask], None if edge_attr is None else edge_attr[mask]


def dropout_adj(edge_index,
                edge_attr=None,
                p=0.5,
                force_undirected=False,
                num_nodes=None,
                training=True):
    r"""Randomly drops edges from the adjacency matrix
    :obj:`(edge_index, edge_attr)` with propability :obj:`p` using samples from
    a Bernoulli distribution.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        p (float, optional): Dropout probability. (default: :obj:`0.5`)
        force_undirected (bool, optional): If set to :obj:`True`, will either
            drop or keep both edges of an undirected edge.
            (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        training (bool, optional): If set to :obj:`False`, this operation is a
            no-op. (default: :obj:`True`)
    """

    if p < 0. or p > 1.:
        raise ValueError('Dropout probability has to be between 0 and 1, '
                         'but got {}'.format(p))

    if not training:
        return edge_index, edge_attr

    N = num_nodes
    row, col = edge_index

    if force_undirected:
        row, col, edge_attr = filter_adj(row, col, edge_attr, row < col)

    mask = edge_index.new_full((row.size(0), ), 1 - p, dtype=torch.float)
    mask = torch.bernoulli(mask).to(torch.uint8)

    row, col, edge_attr = filter_adj(row, col, edge_attr, mask)

    if force_undirected:
        edge_index = torch.stack(
            [torch.cat([row, col], dim=0),
             torch.cat([col, row], dim=0)],
            dim=0)
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
        edge_index, edge_attr = coalesce(edge_index, edge_attr, N, N)
    else:
        edge_index = torch.stack([row, col], dim=0)

    return edge_index, edge_attr
