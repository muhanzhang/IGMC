import json
from util_functions import neighbors, MyDynamicDataset
from sparseindexer import *
import os
import torch
import random
from models import *
from train_eval import *


def logger(info, model, optimizer, res_dir, save_interval = 3):
    epoch, train_loss, test_rmse = info['epoch'], info['train_loss'], info['test_rmse']
    if not os.path.isdir(res_dir):
        os.mkdir(res_dir)
    with open(os.path.join(res_dir, 'log.txt'), 'a') as f:
        f.write('Epoch {}, train loss {:.4f}, test rmse {:.6f}\n'.format(
            epoch, train_loss, test_rmse))
    if type(epoch) == int and epoch % save_interval == 0:
        print('Saving model states...')
        model_name = os.path.join(res_dir, 'model_checkpoint{}.pth'.format(epoch))
        optimizer_name = os.path.join(
            res_dir, 'optimizer_checkpoint{}.pth'.format(epoch)
        )
        if model is not None:
            torch.save(model.state_dict(), model_name)
        if optimizer is not None:
            torch.save(optimizer.state_dict(), optimizer_name)


def neighbor_songs(pl, A, Acsc, sample_ratio=1,
                   max_nodes_per_hop=None):

    songs = set(Acsc[pl].indices)
    u_nodes, v_nodes = list(songs), [pl]
    # u_dist, v_dist = [0], [0]
    u_visited, v_visited = songs, set(v_nodes)
    u_fringe, v_fringe = songs, set(v_nodes)
    # for dist in range(1, h+1):

    # u_fringe = neighbors(v_fringe, Acsc, False)
    v_fringe = neighbors(u_fringe, A, True)

    # u_fringe = u_fringe - u_visited
    v_fringe = v_fringe - v_visited
    # u_visited = u_visited.union(u_fringe)
    v_visited = v_visited.union(v_fringe)
    if sample_ratio < 1.0:
        u_fringe = random.sample(u_fringe, int(sample_ratio * len(u_fringe)))
        v_fringe = random.sample(v_fringe, int(sample_ratio * len(v_fringe)))
    if max_nodes_per_hop is not None:
        if max_nodes_per_hop < len(u_fringe):
            u_fringe = random.sample(u_fringe, max_nodes_per_hop)
        if max_nodes_per_hop < len(v_fringe):
            v_fringe = random.sample(v_fringe, max_nodes_per_hop)
    # if len(u_fringe) == 0 and len(v_fringe) == 0:
    #     break
    # u_nodes = u_nodes + list(u_fringe)
    v_nodes = v_nodes + list(v_fringe)

    u_fringe = neighbors(v_fringe, Acsc, False)
    u_fringe = u_fringe - u_visited
    if sample_ratio < 1.0:
        u_fringe = random.sample(u_fringe, int(sample_ratio * len(u_fringe)))
    if max_nodes_per_hop is not None:
        if max_nodes_per_hop < len(u_fringe):
            u_fringe = random.sample(u_fringe, max_nodes_per_hop)
    # u_nodes = u_nodes + list(u_fringe)

    # u_dist = u_dist + [dist] * len(u_fringe)
    # v_dist = v_dist + [dist] * len(v_fringe)
    return u_nodes, list(u_fringe)


def get_scores(test_dataset,
               model,
               batch_size,
               logger=None,
               ensemble=False,
               checkpoints=None):
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    model.to(device)
    scores = []
    t_start = time.perf_counter()

    model.eval()
    # rmse = eval_rmse(model, test_loader, device, show_progress=True)
    pbar = tqdm(test_loader)
    # s = time.time()
    for data in pbar:
        # s = time_passed("data",s)
        data = data.to(device)
        # s = time_passed("data to device",s)
        with torch.no_grad():
            out = model(data)
            scores.append(out)
        # s = time_passed("model out",s)

    t_end = time.perf_counter()
    torch.cuda.empty_cache()
    duration = t_end - t_start

    return scores


def get_model(graphs,
             model_dir = None,
             opt_dir = None,
             with_optimizer = False):

    model = IGMC(
        graphs,
        latent_dim=[32, 32, 32, 32],
        num_relations=2,
        num_bases=4,
        regression=True,
        adj_dropout=0.2,  # args.adj_dropout,
        force_undirected=False,  # args.force_undirected,
        side_features=False,  # args.use_features,
        n_side_features=0,
        multiply_by=1
    )
    if model_dir:
        model.load_state_dict(torch.load(model_dir))
        if with_optimizer:
            lr = 1e-3
            weight_decay = 0
            optimizer = optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            optimizer.load_state_dict(torch.load(opt_dir))

            return model, optimizer
    return model

def r_precision(G, R):
  """
  G: Ground truth songs
  R: Recommended songs
  type: list
  """
  n_match = 0
  for r in R[:len(G)]:
    if r in G:
      n_match += 1
  return n_match/len(G)

def load_data(file_dir = "drive/My Drive/music recommendation/",
              train_size = 10000,
              val_split = 0.01):
    adj_train = sp.load_npz(file_dir + "spotify_mpd.npz")
    adj_train = sp.csr_matrix.astype(adj_train, np.int8)
    adj_train_csc = adj_train.tocsc()

    adj_train = adj_train.tocoo()
    row = adj_train.row
    col = adj_train.col
    adj_train = adj_train.tocsr()
    # row = row[col < 1000000] # exclude challenge dataset
    # col = col[col < 1000000]

    # half of the set are correct song-playlist pairs (ones), half of them are random, so mostly zeros
    train_idx_ones = np.random.randint(0, len(row), train_size // 2)
    u_train_idx = row[train_idx_ones]
    v_train_idx = col[train_idx_ones]
    num_songs = max(row)
    num_playlists = max(col)


    u_train_idx_zeros = np.random.randint(0, num_songs, train_size // 2)
    v_train_idx_zeros = np.random.randint(0, num_playlists, train_size // 2)

    u_val_idx = np.append(u_train_idx[int((train_size / 2) * (1 - val_split)):],
                          u_train_idx_zeros[int((train_size / 2) * (1 - val_split)):])
    v_val_idx = np.append(v_train_idx[int((train_size / 2) * (1 - val_split)):],
                          v_train_idx_zeros[int((train_size / 2) * (1 - val_split)):])

    u_train_idx = np.append(u_train_idx[:int((train_size / 2) * (1 - val_split))],
                            u_train_idx_zeros[:int((train_size / 2) * (1 - val_split))])
    v_train_idx = np.append(v_train_idx[:int((train_size / 2) * (1 - val_split))],
                            v_train_idx_zeros[:int((train_size / 2) * (1 - val_split))])

    train_labels = np.squeeze(np.asarray(adj_train[u_train_idx, v_train_idx]))
    val_labels = np.squeeze(np.asarray(adj_train[u_val_idx, v_val_idx]))

    train_indices = (u_train_idx, v_train_idx)
    val_indices = (u_val_idx, v_val_idx)
    u_features, v_features = None, None
    n_features = 0

    appx = "_" + str(train_size)[:-3] + "k"
    save_appx = ""
    data_name = "spotify"
    dataset_class = 'MyDynamicDataset' if True else 'MyDataset'
    data_combo = (data_name, appx, save_appx)

    Aridxer = SparseRowIndexer(adj_train)
    Acidxer = SparseColIndexer(adj_train_csc)

    train_graphs = eval(dataset_class)(
        'data/{}{}/{}/train'.format(*data_combo),
        Aridxer,
        Acidxer,
        train_indices,
        train_labels,
        1,  # args.hop
        1,  # args.sample_ratio,
        10000,  # args.max_nodes_per_hop,
        u_features,
        v_features,
        class_values = np.array([0,1]),
        max_num=None  # args.max_train_num,
        # parallel = True
    )

    val_graphs = eval(dataset_class)(
        'data/{}{}/{}/val'.format(*data_combo),
        Aridxer,
        Acidxer,
        val_indices,
        val_labels,
        1,  # args.hop
        1,  # args.sample_ratio,
        10000,  # args.max_nodes_per_hop,
        u_features,
        v_features,
        class_values = np.array([0,1]),
        max_num=None,  # args.max_val_num
    )

    val_test_appendix = "val" + appx
    res_dir = os.path.join(
        file_dir, 'results/{}{}_{}'.format(
            data_name, save_appx, val_test_appendix
        ))

    return train_graphs, val_graphs, res_dir




