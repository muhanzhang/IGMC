import time
import math
import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader
from tqdm import tqdm
import pdb

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
                          logger=None):

    rmses = []

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    model.to(device).reset_parameters()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t_start = time.perf_counter()
    pbar = tqdm(range(1, epochs + 1))
    for epoch in pbar:
        train_loss = train(model, optimizer, train_loader, device, regression=True)
        rmses.append(eval_rmse(model, test_loader, device))
        eval_info = {
            'epoch': epoch,
            'train_loss': train_loss,
            'test_rmse': rmses[-1],
        }
        pbar.set_description('Epoch {}, train loss {:.4f}, test rmse {:.4f}'.format(*eval_info.values()))
        if logger is not None:
            logger(eval_info, model, optimizer)

        if epoch % lr_decay_step_size == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_decay_factor * param_group['lr']

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t_end = time.perf_counter()
    duration = t_end - t_start


    print('Final Test RMSE: {:.3f}, Duration: {:.3f}'.
          format(rmses[-1],
                 duration))

    return rmses[-1]


def test_once(test_dataset,
              model,
              batch_size,
              logger=None):

    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    model.to(device)
    rmse = eval_rmse(model, test_loader, device, show_progress=True)
    eval_info = {
        'epoch': 0,
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


def train(model, optimizer, loader, device, regression=False):
    model.train()

    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)
        if regression:
            loss = F.mse_loss(out, data.y.view(-1))
        else:
            loss = F.nll_loss(out, data.y.view(-1))
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
        torch.cuda.empty_cache()
    return total_loss / len(loader.dataset)


def eval_acc(model, loader, device):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)


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


def visualize(model, graphs, num=3):
    model.eval()
    model.to(device)
    preds = []
    for data in graphs:
        pred = model(data)
        preds.append(pred.item())
    order = np.argsort(preds)
    highest = [graphs[i] for i in order[-num:]]
    lowest = [graphs[i] for i in order[:num]]
    
