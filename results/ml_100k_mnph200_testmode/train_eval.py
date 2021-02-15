import time
import os
import math
import multiprocessing as mp
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader
from tqdm import tqdm
import pdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from util_functions import PyGGraph_to_nx

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_multiple_epochs(train_dataset,
                          test_dataset,
                          model,
                          epochs,
                          batch_size,
                          lr,
                          lr_decay_factor,
                          lr_decay_step_size,
                          weight_decay,
                          ARR=0, 
                          test_freq=1, 
                          logger=None, 
                          continue_from=None, 
                          res_dir=None):

    rmses = []

    if train_dataset.__class__.__name__ == 'MyDynamicDataset':
        num_workers = mp.cpu_count()
    else:
        num_workers = 2
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, 
                              num_workers=num_workers)
    if test_dataset.__class__.__name__ == 'MyDynamicDataset':
        num_workers = mp.cpu_count()
    else:
        num_workers = 2
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, 
                             num_workers=num_workers)

    model.to(device).reset_parameters()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    start_epoch = 1
    if continue_from is not None:
        model.load_state_dict(
            torch.load(os.path.join(res_dir, 'model_checkpoint{}.pth'.format(continue_from)))
        )
        optimizer.load_state_dict(
            torch.load(os.path.join(res_dir, 'optimizer_checkpoint{}.pth'.format(continue_from)))
        )
        start_epoch = continue_from + 1
        epochs -= continue_from

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    batch_pbar = len(train_dataset) >= 100000
    t_start = time.perf_counter()
    if not batch_pbar:
        pbar = tqdm(range(start_epoch, epochs + start_epoch))
    else:
        pbar = range(start_epoch, epochs + start_epoch)
    for epoch in pbar:
        train_loss = train(model, optimizer, train_loader, device, regression=True, ARR=ARR, 
                           show_progress=batch_pbar, epoch=epoch)
        if epoch % test_freq == 0:
            rmses.append(eval_rmse(model, test_loader, device, show_progress=batch_pbar))
        else:
            rmses.append(np.nan)
        eval_info = {
            'epoch': epoch,
            'train_loss': train_loss,
            'test_rmse': rmses[-1],
        }
        if not batch_pbar:
            pbar.set_description(
                'Epoch {}, train loss {:.6f}, test rmse {:.6f}'.format(*eval_info.values())
            )
        else:
            print('Epoch {}, train loss {:.6f}, test rmse {:.6f}'.format(*eval_info.values()))

        if epoch % lr_decay_step_size == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_decay_factor * param_group['lr']

        if logger is not None:
            logger(eval_info, model, optimizer)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t_end = time.perf_counter()
    duration = t_end - t_start

    print('Final Test RMSE: {:.6f}, Duration: {:.6f}'.
          format(rmses[-1],
                 duration))

    return rmses[-1]


def test_once(test_dataset,
              model,
              batch_size,
              logger=None, 
              ensemble=False, 
              checkpoints=None):

    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    model.to(device)
    t_start = time.perf_counter()
    if ensemble and checkpoints:
        rmse = eval_rmse_ensemble(model, checkpoints, test_loader, device, show_progress=True)
    else:
        rmse = eval_rmse(model, test_loader, device, show_progress=True)
    t_end = time.perf_counter()
    duration = t_end - t_start
    print('Test Once RMSE: {:.6f}, Duration: {:.6f}'.format(rmse, duration))
    epoch_info = 'test_once' if not ensemble else 'ensemble'
    eval_info = {
        'epoch': epoch_info,
        'train_loss': 0,
        'test_rmse': rmse,
        }
    if logger is not None:
        logger(eval_info, None, None)
    return rmse


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def train(model, optimizer, loader, device, regression=False, ARR=0, 
          show_progress=False, epoch=None):
    model.train()
    total_loss = 0
    if show_progress:
        pbar = tqdm(loader)
    else:
        pbar = loader
    for data in pbar:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)
        if regression:
            loss = F.mse_loss(out, data.y.view(-1))
        else:
            loss = F.nll_loss(out, data.y.view(-1))
        if show_progress:
            pbar.set_description('Epoch {}, batch loss: {}'.format(epoch, loss.item()))
        if ARR != 0:
            for gconv in model.convs:
                w = torch.matmul(
                    gconv.att, 
                    gconv.basis.view(gconv.num_bases, -1)
                ).view(gconv.num_relations, gconv.in_channels, gconv.out_channels)
                reg_loss = torch.sum((w[1:, :, :] - w[:-1, :, :])**2)
                loss += ARR * reg_loss
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
        torch.cuda.empty_cache()
    return total_loss / len(loader.dataset)


def eval_loss(model, loader, device, regression=False, show_progress=False):
    model.eval()
    loss = 0
    if show_progress:
        print('Testing begins...')
        pbar = tqdm(loader)
    else:
        pbar = loader
    for data in pbar:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
        if regression:
            loss += F.mse_loss(out, data.y.view(-1), reduction='sum').item()
        else:
            loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
        torch.cuda.empty_cache()
    return loss / len(loader.dataset)


def eval_rmse(model, loader, device, show_progress=False):
    mse_loss = eval_loss(model, loader, device, True, show_progress)
    rmse = math.sqrt(mse_loss)
    return rmse


def eval_loss_ensemble(model, checkpoints, loader, device, regression=False, show_progress=False):
    loss = 0
    Outs = []
    for i, checkpoint in enumerate(checkpoints):
        if show_progress:
            print('Testing begins...')
            pbar = tqdm(loader)
        else:
            pbar = loader
        model.load_state_dict(torch.load(checkpoint))
        model.eval()
        outs = []
        if i == 0:
            ys = []
        for data in pbar:
            data = data.to(device)
            if i == 0:
                ys.append(data.y.view(-1))
            with torch.no_grad():
                out = model(data)
                outs.append(out)
        if i == 0:
            ys = torch.cat(ys, 0)
        outs = torch.cat(outs, 0).view(-1, 1)
        Outs.append(outs)
    Outs = torch.cat(Outs, 1).mean(1)
    if regression:
        loss += F.mse_loss(Outs, ys, reduction='sum').item()
    else:
        loss += F.nll_loss(Outs, ys, reduction='sum').item()
    torch.cuda.empty_cache()
    return loss / len(loader.dataset)


def eval_rmse_ensemble(model, checkpoints, loader, device, show_progress=False):
    mse_loss = eval_loss_ensemble(model, checkpoints, loader, device, True, show_progress)
    rmse = math.sqrt(mse_loss)
    return rmse


def visualize(model, graphs, res_dir, data_name, class_values, num=5, sort_by='prediction'):
    model.eval()
    model.to(device)
    R = []
    Y = []
    graph_loader = DataLoader(graphs, 50, shuffle=False)
    for data in tqdm(graph_loader):
        data = data.to(device)
        r = model(data).detach()
        y = data.y
        R.extend(r.view(-1).tolist())
        Y.extend(y.view(-1).tolist())
    if sort_by == 'true':  # sort graphs by their true ratings
        order = np.argsort(Y).tolist()
    elif sort_by == 'prediction':
        order = np.argsort(R).tolist()
    elif sort_by == 'random':  # randomly select graphs to visualize
        order = np.random.permutation(range(len(R))).tolist()
    highest = [PyGGraph_to_nx(graphs[i]) for i in order[-num:][::-1]]
    lowest = [PyGGraph_to_nx(graphs[i]) for i in order[:num]]
    highest_scores = [R[i] for i in order[-num:][::-1]]
    lowest_scores = [R[i] for i in order[:num]]
    highest_ys = [Y[i] for i in order[-num:][::-1]]
    lowest_ys = [Y[i] for i in order[:num]]
    scores = highest_scores + lowest_scores
    ys = highest_ys + lowest_ys
    type_to_label = {0: 'u0', 1: 'v0', 2: 'u1', 3: 'v1', 4: 'u2', 5: 'v2'}
    type_to_color = {0: 'xkcd:red', 1: 'xkcd:blue', 2: 'xkcd:orange', 
                     3: 'xkcd:lightblue', 4: 'y', 5: 'g'}
    plt.axis('off')
    f = plt.figure(figsize=(20, 10))
    axs = f.subplots(2, num)
    cmap = plt.cm.get_cmap('rainbow')
    vmin, vmax = min(class_values), max(class_values)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    for i, g in enumerate(highest + lowest):
        u_nodes = [x for x, y in g.nodes(data=True) if y['type'] % 2 == 0]
        u0, v0 = 0, len(u_nodes)
        pos = nx.drawing.layout.bipartite_layout(g, u_nodes)
        bottom_u_node = min(pos, key=lambda x: (pos[x][0], pos[x][1]))
        bottom_v_node = min(pos, key=lambda x: (-pos[x][0], pos[x][1]))
        # swap u0 and v0 with bottom nodes if they are not already
        if u0 != bottom_u_node:
            pos[u0], pos[bottom_u_node] = pos[bottom_u_node], pos[u0]
        if v0 != bottom_v_node:
            pos[v0], pos[bottom_v_node] = pos[bottom_v_node], pos[v0]
        labels = {x: type_to_label[y] for x, y in nx.get_node_attributes(g, 'type').items()}
        node_colors = [type_to_color[y] for x, y in nx.get_node_attributes(g, 'type').items()]
        edge_types = nx.get_edge_attributes(g, 'type')
        edge_types = [class_values[edge_types[x]] for x in g.edges()]
        axs[i//num, i%num].axis('off')
        nx.draw_networkx(g, pos, 
                #labels=labels, 
                with_labels=False, 
                node_size=150, 
                node_color=node_colors, edge_color=edge_types, 
                ax=axs[i//num, i%num], edge_cmap=cmap, edge_vmin=vmin, edge_vmax=vmax, 
                )
        # make u0 v0 on top of other nodes
        nx.draw_networkx_nodes(g, {u0: pos[u0]}, nodelist=[u0], node_size=150,
                node_color='xkcd:red', ax=axs[i//num, i%num])
        nx.draw_networkx_nodes(g, {v0: pos[v0]}, nodelist=[v0], node_size=150,
                node_color='xkcd:blue', ax=axs[i//num, i%num])
        axs[i//num, i%num].set_title('{:.4f} ({:})'.format(
            scores[i], ys[i]), x=0.5, y=-0.05, fontsize=20
        )
    f.subplots_adjust(right=0.85)
    cbar_ax = f.add_axes([0.88, 0.15, 0.02, 0.7])
    if len(class_values) > 20:
        class_values = np.linspace(min(class_values), max(class_values), 20, dtype=int).tolist()
    cbar = plt.colorbar(sm, cax=cbar_ax, ticks=class_values)
    cbar.ax.tick_params(labelsize=22)
    f.savefig(os.path.join(res_dir, "visualization_{}_{}.pdf".format(data_name, sort_by)), 
            interpolation='nearest', bbox_inches='tight')
    
    
    
