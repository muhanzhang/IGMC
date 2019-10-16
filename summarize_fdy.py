from __future__ import print_function
import os
import re
import numpy as np
import pdb
from scipy import stats

line_num = -1

seed_range = range(1, 6)

datasets = ['yahoo_music', 'douban', 'flixster']

prefixs = ['_s']

print()
for prefix in prefixs:
    print('Results of ' + prefix)
    for dataset in datasets:
        res_base = 'results/' + dataset + prefix
        RMSE = []
        for seed in seed_range:
            res_dir = res_base + str(seed) + '_testmode/log.txt'
            with open(res_dir, 'r') as f:
                line = f.readlines()[line_num]
                rmse = float(line.split(' ')[-1])
                RMSE.append(rmse)
        RMSE = np.array(RMSE)
        print('\033[91m Results of ' + dataset + '\033[00m')
        print(RMSE)
        print('Mean and std of test rmse:')
        print('%.4f$\pm$%.4f'%(np.around(np.mean(RMSE), 4), np.around(np.std(RMSE), 4)))

