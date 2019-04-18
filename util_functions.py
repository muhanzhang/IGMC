from __future__ import print_function
import numpy as np
import random
from tqdm import tqdm
import os, sys, pdb, math
import networkx as nx
import argparse
import scipy.io as sio
import scipy.sparse as ssp
from sklearn import metrics
import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset
import warnings
warnings.simplefilter('ignore', ssp.SparseEfficiencyWarning)
cur_dir = os.path.dirname(os.path.realpath(__file__))
#sys.path.append('%s/../../pytorch_DGCNN' % cur_dir)
#sys.path.append('%s/software/node2vec/src' % cur_dir)
#import node2vec


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
        x2 = torch.FloatTensor(node_features)
        x = torch.cat([x, x2], 1)
    data = Data(x, edge_index, edge_attr=edge_attr, y=y)
    data.edge_type = edge_type
    return data
    

def PyGGraph_to_nx(data):
    edges = list(zip(data.edge_index[0, :].tolist(), data.edge_index[1, :].tolist()))
    g = nx.from_edgelist(edges)
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
        max_nodes_per_hop=None, 
        u_features=None, 
        v_features=None, 
        max_node_label=None, 
        class_values=None, 
        testing=False):
    # extract enclosing subgraphs
    if max_node_label is None:  # if not provided, infer from graphs
        max_n_label = {'max_node_label': 0}

    def helper(A, links, g_labels):
        g_list = []
        with tqdm(total=len(links[0])) as pbar:
            for i, j, g_label in zip(links[0], links[1], g_labels):
                g, n_labels, n_features = subgraph_extraction_labeling((i, j), A, h, max_nodes_per_hop, u_features, v_features, class_values)
                if max_node_label is None:
                    max_n_label['max_node_label'] = max(max(n_labels), max_n_label['max_node_label'])
                    g_list.append((g, g_label, n_labels, n_features))
                else:
                    g_list.append(nx_to_PyGGraph(g, g_label, n_labels, n_features, max_node_label, class_values))
                pbar.update(1)
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


def subgraph_extraction_labeling(ind, A, h=1, max_nodes_per_hop=None, u_features=None, v_features=None, class_values=None):
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
        if u_features is not None and v_features is not None:
            u_extended = np.concatenate([u_features, np.zeros([u_features.shape[0], v_features.shape[1]])], 1)
            v_extended = np.concatenate([np.zeros([v_features.shape[0], u_features.shape[1]]), v_features], 1)
            node_features = np.concatenate([u_extended, v_extended], 0)

        # get identity features (one-hot encodings of node idxes)
        u_ids = one_hot(u_nodes, A.shape[0]+A.shape[1])
        v_ids = one_hot([x+A.shape[0] for x in v_nodes], A.shape[0]+A.shape[1])
        node_ids = np.concatenate([u_ids, v_ids], 0)

        #node_features = np.concatenate([node_features, node_ids], 1)
        node_features = None
        #node_features = node_ids
        #node_labels = [1] * len(labels)
        #node_labels = u_nodes + [x+A.shape[0] for x in v_nodes]

    return g, node_labels, node_features


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


def node_label(subgraph):
    # an implementation of the proposed double-radius node labeling (DRNL)
    K = subgraph.shape[0]
    subgraph_wo0 = subgraph[1:, 1:]
    subgraph_wo1 = subgraph[[0]+range(2, K), :][:, [0]+range(2, K)]
    dist_to_0 = ssp.csgraph.shortest_path(subgraph_wo0, directed=False, unweighted=True)
    dist_to_0 = dist_to_0[1:, 0]
    dist_to_1 = ssp.csgraph.shortest_path(subgraph_wo1, directed=False, unweighted=True)
    dist_to_1 = dist_to_1[1:, 0]
    d = (dist_to_0 + dist_to_1).astype(int)
    d_over_2, d_mod_2 = np.divmod(d, 2)
    labels = 1 + np.minimum(dist_to_0, dist_to_1).astype(int) + d_over_2 * (d_over_2 + d_mod_2 - 1)
    labels = np.concatenate((np.array([1, 1]), labels))
    labels[np.isinf(labels)] = 0
    labels[labels>1e6] = 0  # set inf labels to 0
    labels[labels<-1e6] = 0  # set -inf labels to 0
    return labels

    
def generate_node2vec_embeddings(A, emd_size=128, negative_injection=False, train_neg=None):
    if negative_injection:
        row, col = train_neg
        A = A.copy()
        A[row, col] = 1  # inject negative train
        A[col, row] = 1  # inject negative train
    nx_G = nx.from_scipy_sparse_matrix(A)
    G = node2vec.Graph(nx_G, is_directed=False, p=1, q=1)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks=10, walk_length=80)
    walks = [map(str, walk) for walk in walks]
    model = Word2Vec(walks, size=emd_size, window=10, min_count=0, sg=1, 
            workers=8, iter=1)
    wv = model.wv
    embeddings = np.zeros([A.shape[0], emd_size], dtype='float32')
    sum_embeddings = 0
    empty_list = []
    for i in range(A.shape[0]):
        if str(i) in wv:
            embeddings[i] = wv.word_vec(str(i))
            sum_embeddings += embeddings[i]
        else:
            empty_list.append(i)
    mean_embedding = sum_embeddings / (A.shape[0] - len(empty_list))
    embeddings[empty_list] = mean_embedding
    return embeddings


def AA(A, test_pos, test_neg):
    # Adamic-Adar score
    A_ = A / np.log(A.sum(axis=1))
    A_[np.isnan(A_)] = 0
    A_[np.isinf(A_)] = 0
    sim = A.dot(A_)
    return CalcAUC(sim, test_pos, test_neg)
    
        
def CN(A, test_pos, test_neg):
    # Common Neighbor score
    sim = A.dot(A)
    return CalcAUC(sim, test_pos, test_neg)


def CalcAUC(sim, test_pos, test_neg):
    pos_scores = np.asarray(sim[test_pos[0], test_pos[1]]).squeeze()
    neg_scores = np.asarray(sim[test_neg[0], test_neg[1]]).squeeze()
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.hstack([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc


