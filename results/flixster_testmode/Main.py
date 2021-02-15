import torch
import numpy as np
import sys, copy, math, time, pdb, warnings, traceback
import scipy.io as sio
import scipy.sparse as ssp
import os.path
import random
import argparse
from shutil import copy, rmtree, copytree
from torch.optim.lr_scheduler import ReduceLROnPlateau
from util_functions import *
from data_utils import *
from preprocessing import *
from train_eval import *
from models import *

import traceback
import warnings
import sys

# used to traceback which code cause warnings, can delete
def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

warnings.showwarning = warn_with_traceback


def logger(info, model, optimizer):
    epoch, train_loss, test_rmse = info['epoch'], info['train_loss'], info['test_rmse']
    with open(os.path.join(args.res_dir, 'log.txt'), 'a') as f:
        f.write('Epoch {}, train loss {:.4f}, test rmse {:.6f}\n'.format(
            epoch, train_loss, test_rmse))
    if type(epoch) == int and epoch % args.save_interval == 0:
        print('Saving model states...')
        model_name = os.path.join(args.res_dir, 'model_checkpoint{}.pth'.format(epoch))
        optimizer_name = os.path.join(
            args.res_dir, 'optimizer_checkpoint{}.pth'.format(epoch)
        )
        if model is not None:
            torch.save(model.state_dict(), model_name)
        if optimizer is not None:
            torch.save(optimizer.state_dict(), optimizer_name)


# Arguments
parser = argparse.ArgumentParser(description='Inductive Graph-based Matrix Completion')
# general settings
parser.add_argument('--testing', action='store_true', default=False,
                    help='if set, use testing mode which splits all ratings into train/test;\
                    otherwise, use validation model which splits all ratings into \
                    train/val/test and evaluate on val only')
parser.add_argument('--no-train', action='store_true', default=False,
                    help='if set, skip the training and directly perform the \
                    transfer/ensemble/visualization')
parser.add_argument('--debug', action='store_true', default=False,
                    help='turn on debugging mode which uses a small number of data')
parser.add_argument('--data-name', default='ml_100k', help='dataset name')
parser.add_argument('--data-appendix', default='', 
                    help='what to append to save-names when saving datasets')
parser.add_argument('--save-appendix', default='', 
                    help='what to append to save-names when saving results')
parser.add_argument('--max-train-num', type=int, default=None, 
                    help='set maximum number of train data to use')
parser.add_argument('--max-val-num', type=int, default=None, 
                    help='set maximum number of val data to use')
parser.add_argument('--max-test-num', type=int, default=None, 
                    help='set maximum number of test data to use')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--data-seed', type=int, default=1234, metavar='S',
                    help='seed to shuffle data (1234,2341,3412,4123,1324 are used), \
                    valid only for ml_1m and ml_10m')
parser.add_argument('--reprocess', action='store_true', default=False,
                    help='if True, reprocess data instead of using prestored .pkl data')
parser.add_argument('--dynamic-train', action='store_true', default=False,
                    help='extract training enclosing subgraphs on the fly instead of \
                    storing in disk; works for large datasets that cannot fit into memory')
parser.add_argument('--dynamic-test', action='store_true', default=False)
parser.add_argument('--dynamic-val', action='store_true', default=False)
parser.add_argument('--keep-old', action='store_true', default=False,
                    help='if True, do not overwrite old .py files in the result folder')
parser.add_argument('--save-interval', type=int, default=10,
                    help='save model states every # epochs ')
# subgraph extraction settings
parser.add_argument('--hop', default=1, metavar='S', 
                    help='enclosing subgraph hop number')
parser.add_argument('--sample-ratio', type=float, default=1.0, 
                    help='if < 1, subsample nodes per hop according to the ratio')
parser.add_argument('--max-nodes-per-hop', default=10000, 
                    help='if > 0, upper bound the # nodes per hop by another subsampling')
parser.add_argument('--use-features', action='store_true', default=False,
                    help='whether to use node features (side information)')
# edge dropout settings
parser.add_argument('--adj-dropout', type=float, default=0.2, 
                    help='if not 0, random drops edges from adjacency matrix with this prob')
parser.add_argument('--force-undirected', action='store_true', default=False, 
                    help='in edge dropout, force (x, y) and (y, x) to be dropped together')
# optimization settings
parser.add_argument('--continue-from', type=int, default=None, 
                    help="from which epoch's checkpoint to continue training")
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--lr-decay-step-size', type=int, default=50,
                    help='decay lr by factor A every B steps')
parser.add_argument('--lr-decay-factor', type=float, default=0.1,
                    help='decay lr by factor A every B steps')
parser.add_argument('--epochs', type=int, default=80, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                    help='batch size during training')
parser.add_argument('--test-freq', type=int, default=1, metavar='N',
                    help='test every n epochs')
parser.add_argument('--ARR', type=float, default=0.001, 
                    help='The adjacenct rating regularizer. If not 0, regularize the \
                    differences between graph convolution parameters W associated with\
                    adjacent ratings')
# transfer learning, ensemble, and visualization settings
parser.add_argument('--transfer', default='',
                    help='if not empty, load the pretrained models in this path')
parser.add_argument('--num-relations', type=int, default=5,
                    help='if transfer, specify num_relations in the transferred model')
parser.add_argument('--multiply-by', type=int, default=1,
                    help='if transfer, specify how many times to multiply the predictions by')
parser.add_argument('--visualize', action='store_true', default=False,
                    help='if True, load a pretrained model and do visualization exps')
parser.add_argument('--ensemble', action='store_true', default=False,
                    help='if True, load a series of model checkpoints and ensemble the results')
parser.add_argument('--standard-rating', action='store_true', default=False,
                    help='if True, maps all ratings to standard 1, 2, 3, 4, 5 before training')
# sparsity experiment settings
parser.add_argument('--ratio', type=float, default=1.0,
                    help="For ml datasets, if ratio < 1, downsample training data to the\
                    target ratio")


'''
    Set seeds, prepare for transfer learning (if --transfer)
'''
args = parser.parse_args()
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
print(args)
random.seed(args.seed)
np.random.seed(args.seed)
args.hop = int(args.hop)
if args.max_nodes_per_hop is not None:
    args.max_nodes_per_hop = int(args.max_nodes_per_hop)

rating_map, post_rating_map = None, None
if args.standard_rating:
    if args.data_name in ['flixster', 'ml_10m']: # original 0.5, 1, ..., 5
        rating_map = {x: int(math.ceil(x)) for x in np.arange(0.5, 5.01, 0.5).tolist()}
    elif args.data_name == 'yahoo_music':  # original 1, 2, ..., 100
        rating_map = {x: (x-1)//20+1 for x in range(1, 101)}
    else:
        rating_map = None

if args.transfer:
    if args.data_name in ['flixster', 'ml_10m']: # original 0.5, 1, ..., 5
        post_rating_map = {
            x: int(i // (10 / args.num_relations)) 
            for i, x in enumerate(np.arange(0.5, 5.01, 0.5).tolist())
        }
    elif args.data_name == 'yahoo_music':  # original 1, 2, ..., 100
        post_rating_map = {
            x: int(i // (100 / args.num_relations)) 
            for i, x in enumerate(np.arange(1, 101).tolist())
        }
    else:  # assume other datasets have standard ratings 1, 2, 3, 4, 5
        post_rating_map = {
            x: int(i // (5 / args.num_relations)) 
            for i, x in enumerate(np.arange(1, 6).tolist())
        }


'''
    Prepare train/test (testmode) or train/val/test (valmode) splits
'''
args.file_dir = os.path.dirname(os.path.realpath('__file__'))
if args.testing:
    val_test_appendix = 'testmode'
else:
    val_test_appendix = 'valmode'
args.res_dir = os.path.join(
    args.file_dir, 'results/{}{}_{}'.format(
        args.data_name, args.save_appendix, val_test_appendix
    )
)
if args.transfer == '':
    args.model_pos = os.path.join(args.res_dir, 'model_checkpoint{}.pth'.format(args.epochs))
else:
    args.model_pos = os.path.join(args.transfer, 'model_checkpoint{}.pth'.format(args.epochs))
if not os.path.exists(args.res_dir):
    os.makedirs(args.res_dir) 

if not args.keep_old and not args.transfer:
    # backup current main.py, model.py files
    copy('Main.py', args.res_dir)
    copy('util_functions.py', args.res_dir)
    copy('models.py', args.res_dir)
    copy('train_eval.py', args.res_dir)
# save command line input
cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
with open(os.path.join(args.res_dir, 'cmd_input.txt'), 'a') as f:
    f.write(cmd_input)
print('Command line input: ' + cmd_input + ' is saved.')

if args.data_name in ['ml_1m', 'ml_10m', 'ml_25m']:
    if args.use_features:
        datasplit_path = (
            'raw_data/' + args.data_name + '/withfeatures_split_seed' + 
            str(args.data_seed) + '.pickle'
        )
    else:
        datasplit_path = (
            'raw_data/' + args.data_name + '/split_seed' + str(args.data_seed) + 
            '.pickle'
        )
elif args.use_features:
    datasplit_path = 'raw_data/' + args.data_name + '/withfeatures.pickle'
else:
    datasplit_path = 'raw_data/' + args.data_name + '/nofeatures.pickle'

if args.data_name in ['flixster', 'douban', 'yahoo_music']:
    (
        u_features, v_features, adj_train, train_labels, train_u_indices, train_v_indices,
        val_labels, val_u_indices, val_v_indices, test_labels, test_u_indices, 
        test_v_indices, class_values
    ) = load_data_monti(args.data_name, args.testing, rating_map, post_rating_map)
elif args.data_name == 'ml_100k':
    print("Using official MovieLens split u1.base/u1.test with 20% validation...")
    (
        u_features, v_features, adj_train, train_labels, train_u_indices, train_v_indices,
        val_labels, val_u_indices, val_v_indices, test_labels, test_u_indices, 
        test_v_indices, class_values
    ) = load_official_trainvaltest_split(
        args.data_name, args.testing, rating_map, post_rating_map, args.ratio
    )
else:
    (
        u_features, v_features, adj_train, train_labels, train_u_indices, train_v_indices,
        val_labels, val_u_indices, val_v_indices, test_labels, test_u_indices, 
        test_v_indices, class_values
    ) = create_trainvaltest_split(
        args.data_name, 1234, args.testing, datasplit_path, True, True, rating_map, 
        post_rating_map, args.ratio
    )

print('All ratings are:')
print(class_values)
'''
Explanations of the above preprocessing:
    class_values are all the original continuous ratings, e.g. 0.5, 2...
    They are transformed to rating labels 0, 1, 2... acsendingly.
    Thus, to get the original rating from a rating label, apply: class_values[label]
    Note that train_labels etc. are all rating labels.
    But the numbers in adj_train are rating labels + 1, why? Because to accomodate 
    neutral ratings 0! Thus, to get any edge label from adj_train, remember to substract 1.
    If testing=True, adj_train will include both train and val ratings, and all train 
    data will be the combination of train and val.
'''

if args.use_features:
    u_features, v_features = u_features.toarray(), v_features.toarray()
    n_features = u_features.shape[1] + v_features.shape[1]
    print('Number of user features {}, item features {}, total features {}'.format(
        u_features.shape[1], v_features.shape[1], n_features))
else:
    u_features, v_features = None, None
    n_features = 0

if args.debug:  # use a small number of data to debug
    num_data = 1000
    train_u_indices, train_v_indices = train_u_indices[:num_data], train_v_indices[:num_data]
    val_u_indices, val_v_indices = val_u_indices[:num_data], val_v_indices[:num_data]
    test_u_indices, test_v_indices = test_u_indices[:num_data], test_v_indices[:num_data]

train_indices = (train_u_indices, train_v_indices)
val_indices = (val_u_indices, val_v_indices)
test_indices = (test_u_indices, test_v_indices)
print('#train: %d, #val: %d, #test: %d' % (
    len(train_u_indices), 
    len(val_u_indices), 
    len(test_u_indices), 
))

'''
    Extract enclosing subgraphs to build the train/test or train/val/test graph datasets.
    (Note that we must extract enclosing subgraphs for testmode and valmode separately, 
    since the adj_train is different.)
'''
train_graphs, val_graphs, test_graphs = None, None, None
data_combo = (args.data_name, args.data_appendix, val_test_appendix)
if args.reprocess:
    # if reprocess=True, delete the previously cached data and reprocess.
    if os.path.isdir('data/{}{}/{}/train'.format(*data_combo)):
        rmtree('data/{}{}/{}/train'.format(*data_combo))
    if os.path.isdir('data/{}{}/{}/val'.format(*data_combo)):
        rmtree('data/{}{}/{}/val'.format(*data_combo))
    if os.path.isdir('data/{}{}/{}/test'.format(*data_combo)):
        rmtree('data/{}{}/{}/test'.format(*data_combo))
# create dataset, either dynamically extract enclosing subgraphs, 
# or extract in preprocessing and save to disk.
dataset_class = 'MyDynamicDataset' if args.dynamic_train else 'MyDataset'
train_graphs = eval(dataset_class)(
    'data/{}{}/{}/train'.format(*data_combo),
    adj_train, 
    train_indices, 
    train_labels, 
    args.hop, 
    args.sample_ratio, 
    args.max_nodes_per_hop, 
    u_features, 
    v_features, 
    class_values, 
    max_num=args.max_train_num
)
dataset_class = 'MyDynamicDataset' if args.dynamic_test else 'MyDataset'
test_graphs = eval(dataset_class)(
    'data/{}{}/{}/test'.format(*data_combo),
    adj_train, 
    test_indices, 
    test_labels, 
    args.hop, 
    args.sample_ratio, 
    args.max_nodes_per_hop, 
    u_features, 
    v_features, 
    class_values, 
    max_num=args.max_test_num
)
if not args.testing:
    dataset_class = 'MyDynamicDataset' if args.dynamic_val else 'MyDataset'
    val_graphs = eval(dataset_class)(
        'data/{}{}/{}/val'.format(*data_combo),
        adj_train, 
        val_indices, 
        val_labels, 
        args.hop, 
        args.sample_ratio, 
        args.max_nodes_per_hop, 
        u_features, 
        v_features, 
        class_values, 
        max_num=args.max_val_num
    )

# Determine testing data (on which data to evaluate the trained model
if not args.testing: 
    test_graphs = val_graphs

print('Used #train graphs: %d, #test graphs: %d' % (
    len(train_graphs), 
    len(test_graphs), 
))

'''
    Train and apply the GNN model
'''
if False:
    # DGCNN_RS GNN model
    model = DGCNN_RS(
        train_graphs, 
        latent_dim=[32, 32, 32, 1], 
        k=0.6, 
        num_relations=len(class_values), 
        num_bases=4, 
        regression=True, 
        adj_dropout=args.adj_dropout, 
        force_undirected=args.force_undirected
    )
    # record the k used in sortpooling
    if not args.transfer:
        with open(os.path.join(args.res_dir, 'cmd_input.txt'), 'a') as f:
            f.write(' --k ' + str(model.k) + '\n')
            print('k is saved.')
else:
    # IGMC GNN model (default)
    if args.transfer:
        num_relations = args.num_relations
        multiply_by = args.multiply_by
    else:
        num_relations = len(class_values)
        multiply_by = 1
    model = IGMC(
        train_graphs, 
        latent_dim=[32, 32, 32, 32], 
        num_relations=num_relations, 
        num_bases=4, 
        regression=True, 
        adj_dropout=args.adj_dropout, 
        force_undirected=args.force_undirected, 
        side_features=args.use_features, 
        n_side_features=n_features, 
        multiply_by=multiply_by
    )
    total_params = sum(p.numel() for param in model.parameters() for p in param)
    print(f'Total number of parameters is {total_params}')
    

if not args.no_train:
    train_multiple_epochs(
        train_graphs,
        test_graphs,
        model,
        args.epochs, 
        args.batch_size, 
        args.lr, 
        lr_decay_factor=args.lr_decay_factor, 
        lr_decay_step_size=args.lr_decay_step_size, 
        weight_decay=0, 
        ARR=args.ARR, 
        test_freq=args.test_freq, 
        logger=logger, 
        continue_from=args.continue_from, 
        res_dir=args.res_dir
    )

if args.visualize:
    model.load_state_dict(torch.load(args.model_pos))
    visualize(
        model, 
        test_graphs, 
        args.res_dir, 
        args.data_name, 
        class_values, 
        sort_by='prediction'
    )
    if args.transfer:
        rmse = test_once(test_graphs, model, args.batch_size, logger)
        print('Transfer learning rmse is: {:.6f}'.format(rmse))
else:
    if args.ensemble:
        if args.data_name == 'ml_1m':
            start_epoch, end_epoch, interval = args.epochs-15, args.epochs, 5
        else: 
            start_epoch, end_epoch, interval = args.epochs-30, args.epochs, 10
        if args.transfer:
            checkpoints = [
                os.path.join(args.transfer, 'model_checkpoint%d.pth' %x) 
                for x in range(start_epoch, end_epoch+1, interval)
            ]
            epoch_info = 'transfer {}, ensemble of range({}, {}, {})'.format(
                args.transfer, start_epoch, end_epoch, interval
            )
        else:
            checkpoints = [
                os.path.join(args.res_dir, 'model_checkpoint%d.pth' %x) 
                for x in range(start_epoch, end_epoch+1, interval)
            ]
            epoch_info = 'ensemble of range({}, {}, {})'.format(
                start_epoch, end_epoch, interval
            )
        rmse = test_once(
            test_graphs, 
            model, 
            args.batch_size, 
            logger=None, 
            ensemble=True, 
            checkpoints=checkpoints
        )
        print('Ensemble test rmse is: {:.6f}'.format(rmse))
    else:
        if args.transfer:
            model.load_state_dict(torch.load(args.model_pos))
            rmse = test_once(test_graphs, model, args.batch_size, logger=None)
            epoch_info = 'transfer {}, epoch {}'.format(args.transfer, args.epoch)
        print('Test rmse is: {:.6f}'.format(rmse))

    eval_info = {
        'epoch': epoch_info,
        'train_loss': 0,
        'test_rmse': rmse,
    }
    logger(eval_info, None, None)



