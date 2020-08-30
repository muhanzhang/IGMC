IGMC -- Inductive Graph-based Matrix Completion
===============================================================================

![alt text](https://github.com/muhanzhang/IGMC/raw/master/overall2.svg?sanitize=true "Illustration of IGMC")

Update
------
8/27/2020: Significantly improved the subgraph extraction speed. With an 8-core machine, now it only takes 30 seconds and 10 minutes to extract subgraphs for ml_100k and ml_1m, respectively. Using --dynamic-dataset also shows about 50% speed-up.

About
-----

IGMC is an __inductive__ matrix completion model based on graph neural networks __without__ using any side information. Traditional matrix factorization approaches factorize the (rating) matrix into the product of low-dimensional latent embeddings of rows (users) and columns (items), which are __transductive__ since the learned embeddings cannot generalize to unseen rows/columns or to new matrices. To make matrix completion __inductive__, content (side information), such as user's age or movie's genre, has to be used previously. However, high-quality content is not always available, and can be hard to extract. Under the extreme setting where __not any__ side information is available other than the matrix to complete, can we still learn an inductive matrix completion model? IGMC achieves this by training a graph neural network (GNN) based purely on local subgraphs around (user, item) pairs extracted from the bipartite graph formed by the rating matrix, and maps these subgraphs to their corresponding ratings. It does not rely on any global information specific to the rating matrix or the task, nor does it learn embeddings specific to the observed users/items. Thus, IGMC is a completely inductive model. 

Since IGMC is inductive, it can generalize to users/items unseen during the training (given that their interactions exist), and can even __transfer__ to new tasks. Our transfer learning experiments show that a model trained out of the MovieLens dataset can be directly used to predict Douban movie ratings and works surprisingly well. For more information, please check our paper:
> M. Zhang and Y. Chen, Inductive Matrix Completion Based on Graph Neural Networks. [\[PDF\]](https://openreview.net/pdf?id=ByxxgCEYDS)

Requirements
------------

Latest tested combination: Python 3.8.1 + PyTorch 1.4.0 + PyTorch_Geometric 1.4.2.

Install [PyTorch](https://pytorch.org/)

Install [PyTorch_Geometric](https://rusty1s.github.io/pytorch_geometric/build/html/notes/installation.html)

Other required python libraries: numpy, scipy, pandas, h5py, networkx, tqdm etc.

Usages
------

### Flixster, Douban and YahooMusic

To train on Flixster, type:

    python Main.py --data-name flixster --epochs 40 --testing --ensemble

The results will be saved in "results/flixster\_testmode/". The processed enclosing subgraphs will be saved in "data/flixster/testmode/". Change flixster to douban or yahoo\_music to do the same experiments on Douban and YahooMusic datasets, respectively. Delete --testing to evaluate on a validation set to do hyperparameter tuning.

### MovieLens-100K and MovieLens-1M

To train on MovieLens-100K, type:

    python Main.py --data-name ml_100k --save-appendix _mnph200 --data-appendix _mnph200 --epochs 80 --max-nodes-per-hop 200 --testing --ensemble --dynamic-train

where the --max-nodes-per-hop argument specifies the maximum number of neighbors to sample for each node during the enclosing subgraph extraction, whose purpose is to limit the subgraph size to accomodate large datasets. The --dynamic-train option makes the training enclosing subgraphs dynamically generated rather than generated in a preprocessing step and saved in disk, whose purpose is to reduce memory consumption. However, you may remove the option to generate a static dataset for future reuses. Append "--dynamic-test" to make the test dataset also dynamic. The default batch size is 50, if a batch cannot fit into your GPU memory, you can reduce batch size by appending "--batch-size 25" to the above command.

The results will be saved in "results/ml\_100k\_mnph200\_testmode/". The processed enclosing subgraphs will be saved in "data/ml\_100k\_mnph200/testmode/" if you do not use dynamic datasets. 

To train on MovieLens-1M, type:
    
    python Main.py --data-name ml_1m --save-appendix _mnhp100 --data-appendix _mnph100 --max-nodes-per-hop 100 --testing --epochs 40 --save-interval 5 --adj-dropout 0 --lr-decay-step-size 20 --ensemble --dynamic-train

### Sparse rating matrix

To repeat the sparsity experiment in the paper (sparsify MovieLens-1M' rating matrix to keep 20% ratings only), type the following:

    python Main.py --data-name ml_1m --save-appendix _mnhp100_ratio02 --ratio 0.2 --data-appendix _mnph100 --max-nodes-per-hop 100 --testing --epochs 40 --save-interval 5 --adj-dropout 0 --lr-decay-step-size 20 --ensemble --dynamic-train

Modify --ratio 0.2 to change the sparsity ratios. Attach --ensemble and run again to get the ensemble test results.

### Transfer learning

To repeat the transfer learning experiment in the paper (transfer the model trained previously on MovieLens-100K to Flixster, Douban, and YahooMusic), use the provided script by typing:

    ./run_transfer_exps.sh DATANAME

Replace DATANAME with flixster, douban and yahoo_music to transfer to each dataset. The results will be attached to each dataset's original "log.txt" file.

### Visualization

After training a model on a dataset, to visualize the testing enclosing subgraphs with the highest and lowest predicted ratings, type the following (we use Flixster as an example):

    python Main.py --data-name flixster --epochs 40 --testing --no-train --visualize

It will load "results/flixster\_testmode/model\_checkpoint40.pth" and save the visualization in "results/flixster\_testmode/visualization_flixster_prediction.pdf".

Check "Main.py" and "train\_eval.py" for more options to play with. Check "models.py" for the graph neural network used.

Reference
---------

If you find the code useful, please cite our paper.

    @inproceedings{
      Zhang2020Inductive,
      title={Inductive Matrix Completion Based on Graph Neural Networks},
      author={Muhan Zhang and Yixin Chen},
      booktitle={International Conference on Learning Representations},
      year={2020},
      url={https://openreview.net/forum?id=ByxxgCEYDS}
    }

Check out our another successful work of inductive [link prediction](https://github.com/muhanzhang/SEAL). 

Muhan Zhang, Washington University in St. Louis
muhan@wustl.edu
10/13/2019
