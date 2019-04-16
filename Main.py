import torch
import numpy as np
import sys, copy, math, time, pdb
import pickle
import scipy.io as sio
import scipy.sparse as ssp
import os.path
import random
import argparse
from shutil import copy, rmtree, copytree
from torch.optim.lr_scheduler import ReduceLROnPlateau
#from models import *
#sys.path.append('%s/../pytorch_DGCNN' % os.path.dirname(os.path.realpath(__file__)))
from util_functions import *
from data_utils import *
from preprocessing import *
from PyG_GNN.train_eval import *
from PyG_GNN.models import DGCNN, DGCNN_RS



parser = argparse.ArgumentParser(description='Link Prediction with SEAL')
# general settings
parser.add_argument('--testing', action='store_true', default=False,
                    help='turn on testing mode')
parser.add_argument('--debug', action='store_true', default=False,
                    help='turn on debugging mode')
parser.add_argument('--data-name', default='ml_100k', help='dataset name')
parser.add_argument('--data-appendix', default='', 
                    help='what to append to save-names when saving datasets')
parser.add_argument('--save-appendix', default='', 
                    help='what to append to save-names when saving results')
parser.add_argument('--max-train-num', type=int, default=None, 
                    help='set maximum number of train links (to fit into memory)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--data-seed', type=int, default=1234, metavar='S',
                    help='seed to shuffle data (1234,2341,3412,4123,1324 are used)')
parser.add_argument('--reprocess', action='store_true', default=False,
                    help='if True, reprocess data instead of using prestored .pkl data')
parser.add_argument('--keep-old', action='store_true', default=False,
                    help='if True, do not remove any old data in the result folder')
parser.add_argument('--save-interval', type=int, default=10,
                    help='save model states every * epochs ')
# model settings
parser.add_argument('--hop', default=1, metavar='S', 
                    help='enclosing subgraph hop number, \
                    options: 1, 2,..., "auto"')
parser.add_argument('--max-nodes-per-hop', default=10000, 
                    help='if > 0, upper bound the # nodes per hop by subsampling')
parser.add_argument('--use-features', action='store_true', default=False,
                    help='whether to use node features (side information)')
parser.add_argument('--k', default=0.6, type=float, 
                    help='k used in sortpooling, if k < 1, \
                    treat as a percentile')
# optimization settings
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                    help='batch size during training')
# transfer learning settings
parser.add_argument('--transfer', action='store_true', default=False,
                    help='if True, load a pretrained model instead of training')
parser.add_argument('--model-pos', default='', 
                    help="where to load the transferred model's state")

args = parser.parse_args()
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
print(args)

random.seed(args.seed)
np.random.seed(args.seed)
if args.hop != 'auto':
    args.hop = int(args.hop)
if args.max_nodes_per_hop is not None:
    args.max_nodes_per_hop = int(args.max_nodes_per_hop)


'''Prepare data'''
args.file_dir = os.path.dirname(os.path.realpath('__file__'))
if args.testing:
    val_test_appendix = 'testmode'
else:
    val_test_appendix = 'valmode'
args.res_dir = os.path.join(args.file_dir, 'results/{}{}_{}'.format(args.data_name, args.save_appendix, val_test_appendix))
if args.transfer:
    args.res_dir += '_transfer'
if not os.path.exists(args.res_dir):
    os.makedirs(args.res_dir) 

# delete old result files
remove_list = [f for f in os.listdir(args.res_dir) if not f.endswith(".pkl") and 
        not f.endswith('.pth')]
for f in remove_list:
    tmp = os.path.join(args.res_dir, f)
    if not os.path.isdir(tmp) and not args.keep_old:
        os.remove(tmp)

if not args.keep_old:
    # backup current main.py, model.py files
    copy('Main.py', args.res_dir)
    copy('util_functions.py', args.res_dir)
    copy('PyG_GNN/models.py', args.res_dir)
    copy('PyG_GNN/train_eval.py', args.res_dir)
    if args.transfer: copy(args.model_pos, args.res_dir)
    # save command line input
    cmd_input = 'python ' + ' '.join(sys.argv)
    with open(os.path.join(args.res_dir, 'cmd_input.txt'), 'w') as f:
        f.write(cmd_input)
    print('Command line input: ' + cmd_input + ' is saved.')


if args.data_name == 'ml_1m' or args.data_name == 'ml_10m':
    if args.use_features:
        datasplit_path = 'data/' + args.data_name + '/withfeatures_split_seed' + str(args.data_seed) + '.pickle'
    else:
        datasplit_path = 'data/' + args.data_name + '/split_seed' + str(args.data_seed) + '.pickle'
elif args.use_features:
    datasplit_path = 'data/' + args.data_name + '/withfeatures.pickle'
else:
    datasplit_path = 'data/' + args.data_name + '/nofeatures.pickle'

if args.data_name == 'flixster' or args.data_name == 'douban' or args.data_name == 'yahoo_music':
    u_features, v_features, adj_train, train_labels, train_u_indices, train_v_indices, \
        val_labels, val_u_indices, val_v_indices, test_labels, \
        test_u_indices, test_v_indices, class_values = load_data_monti(args.data_name, args.testing)

elif args.data_name == 'ml_100k':
    print("Using official MovieLens dataset split u1.base/u1.test with 20% validation set size...")
    u_features, v_features, adj_train, train_labels, train_u_indices, train_v_indices, \
        val_labels, val_u_indices, val_v_indices, test_labels, \
        test_u_indices, test_v_indices, class_values = load_official_trainvaltest_split(args.data_name, args.testing)
else:
    print("Using random dataset split ...")
    u_features, v_features, adj_train, train_labels, train_u_indices, train_v_indices, \
        val_labels, val_u_indices, val_v_indices, test_labels, \
        test_u_indices, test_v_indices, class_values = create_trainvaltest_split(args.data_name, 1234, args.testing,
                                                                                 datasplit_path, True,
                                                                                 True)

print('All ratings are:')
print(class_values)
'''
Explanations of the above preprocessing:
    class_values are all the original continuous ratings, e.g. 0.5, 2...
    They are transformed to rating labels 0, 1, 2... acsendingly.
    Thus, to get the original rating from a rating label, apply: class_values[label]
    Note that train_labels etc. are all rating labels.
    But the numbers in adj_train are rating labels + 1, why? Because to accomodate neutral ratings 0! Thus, to get any edge label from adj_train, remember to substract 1.
    If testing=True, adj_train will include both train and val ratings, and all train data will be the combination of train and val.
'''

if args.use_features:
    u_features, v_features = u_features.toarray(), v_features.toarray()
else:
    u_features, v_features = None, None

if args.debug:
    num_data = 1000
    train_u_indices, train_v_indices = train_u_indices[:num_data], train_v_indices[:num_data]
    val_u_indices, val_v_indices = val_u_indices[:num_data], val_v_indices[:num_data]
    test_u_indices, test_v_indices = test_u_indices[:num_data], test_v_indices[:num_data]

train_indices = (train_u_indices, train_v_indices)
val_indices = (val_u_indices, val_v_indices)
test_indices = (test_u_indices, test_v_indices)

print('#train: %d, #val: %d, #test: %d' % (len(train_u_indices), len(val_u_indices), len(test_u_indices)))

'''
Create train/val/test datasets.
If reprocess=True, delete the previously cached data and reprocess.
Note that we must create different datasets for testing mode and validating mode, since the adj_train will be different (thus extracted subgraphs will be different in different modes).
'''
train_graphs, val_graphs, test_graphs = None, None, None
data_combo = (args.data_name, args.data_appendix, val_test_appendix)
if args.reprocess or not os.path.isdir('data/{}{}/{}/train'.format(*data_combo)):
    if os.path.isdir('data/{}{}/{}/train'.format(*data_combo)):
        rmtree('data/{}{}/{}/train'.format(*data_combo))
    if os.path.isdir('data/{}{}/{}/val'.format(*data_combo)):
        rmtree('data/{}{}/{}/val'.format(*data_combo))
    if os.path.isdir('data/{}{}/{}/test'.format(*data_combo)):
        rmtree('data/{}{}/{}/test'.format(*data_combo))
    # extract enclosing subgraphs and build the datasets
    train_graphs, val_graphs, test_graphs = links2subgraphs(
            adj_train,
            train_indices, 
            val_indices, 
            test_indices,
            train_labels, 
            val_labels, 
            test_labels, 
            args.hop, 
            args.max_nodes_per_hop, 
            u_features, 
            v_features, 
            args.hop*2+1, 
            class_values, 
            args.testing)

if not args.testing:
    val_graphs = MyDataset(val_graphs, root='data/{}{}/{}/val'.format(*data_combo))
test_graphs = MyDataset(test_graphs, root='data/{}{}/{}/test'.format(*data_combo))
train_graphs = MyDataset(train_graphs, root='data/{}{}/{}/train'.format(*data_combo))


'''Determine testing data'''
if not args.testing: 
    #test_graphs = val_graphs
    pass

'''sample certain number of train'''
if args.max_train_num is not None:  
    perm = np.random.permutation(len(train_graphs))[:args.max_train_num]
    train_graphs = train_graphs[torch.tensor(perm)]


'''Train and apply model'''
# GNN configurations
#model = DGCNN(train_graphs, latent_dim=[500, 500], k=0.6, regression=True)
model = DGCNN_RS(train_graphs, 
                 latent_dim=[32, 32, 32, 1], 
                 k=args.k, 
                 num_relations=len(class_values), 
                 num_bases=4, 
                 regression=True)


def logger(info, model, optimizer):
    epoch, train_loss, test_rmse = info['epoch'], info['train_loss'], info['test_rmse']
    with open(os.path.join(args.res_dir, 'log.txt'), 'a') as f:
        f.write('Epoch {}, train loss {:.4f}, test rmse {:.4f}\n'.format(
            epoch, train_loss, test_rmse
            ))
    if epoch % args.save_interval == 0:
        print('Saving model states...')
        model_name = os.path.join(args.res_dir, 'model_checkpoint{}.pth'.format(epoch))
        optimizer_name = os.path.join(args.res_dir, 'optimizer_checkpoint{}.pth'.format(epoch))
        if model is not None:
            torch.save(model.state_dict(), model_name)
        if optimizer is not None:
            torch.save(optimizer.state_dict(), optimizer_name)

if not args.transfer:
    train_multiple_epochs(train_graphs,
                          test_graphs,
                          model,
                          args.epochs, 
                          args.batch_size, 
                          args.lr, 
                          lr_decay_factor=0.1, 
                          lr_decay_step_size=100, 
                          weight_decay=0, 
                          logger=logger)
else:
    model.load_state_dict(torch.load(args.model_pos))
    rmse = test_once(test_graphs, model, args.batch_size, logger)
    print('Transfer learning rmse is: {:.4f}'.format(rmse))

    



pdb.set_trace()

'''
optimizer = optim.Adam(model.parameters(), lr=cmd_args.learning_rate)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)

if args.continue_from is not None:
    epoch = args.continue_from
    model.load_state_dict(torch.load(os.path.join(args.res_dir, 'model_checkpoint{}.pth'.format(epoch))))
    optimizer.load_state_dict(torch.load(os.path.join(args.res_dir, 'optimizer_checkpoint{}.pth'.format(epoch))))
    scheduler.load_state_dict(torch.load(os.path.join(args.res_dir, 'scheduler_checkpoint{}.pth'.format(epoch))))

train_idxes = list(range(len(train_graphs)))
best_loss = None
start_epoch = args.continue_from if args.continue_from is not None else 0
for epoch in range(start_epoch + 1, cmd_args.num_epochs + 1):
    random.shuffle(train_idxes)
    model.train()
    avg_loss = loop_dataset(train_graphs, model, train_idxes, optimizer=optimizer)
    print('\033[92maverage training of epoch %d: RMSE_loss %.5f MAE_loss %.5f\033[0m' % (epoch, avg_loss[0], avg_loss[1]))
    scheduler.step(avg_loss[0])

    model.eval()
    test_loss = loop_dataset(test_graphs, model, list(range(len(test_graphs))))
    print('\033[93maverage test of epoch %d: RMSE_loss %.5f MAE_loss %.5f\033[0m' % (epoch, test_loss[0], test_loss[1]))

    if epoch % args.save_interval == 0:
        model_name = os.path.join(args.res_dir, 'model_checkpoint{}.pth'.format(epoch))
        optimizer_name = os.path.join(args.res_dir, 'optimizer_checkpoint{}.pth'.format(epoch))
        scheduler_name = os.path.join(args.res_dir, 'scheduler_checkpoint{}.pth'.format(epoch))
        torch.save(model.state_dict(), model_name)
        torch.save(optimizer.state_dict(), optimizer_name)
        torch.save(scheduler.state_dict(), scheduler_name)

    with open(os.path.join(args.res_dir, 'train_RMSE_results.txt'), 'a+') as f:
        f.write(str(avg_loss[0]) + '\n')
    with open(os.path.join(args.res_dir, 'train_MAE_results.txt'), 'a+') as f:
        f.write(str(avg_loss[1]) + '\n')
    with open(os.path.join(args.res_dir, 'test_RMSE_results.txt'), 'a+') as f:
        f.write(str(test_loss[0]) + '\n')
    with open(os.path.join(args.res_dir, 'test_MAE_results.txt'), 'a+') as f:
        f.write(str(test_loss[1]) + '\n')
'''

