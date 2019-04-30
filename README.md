IGPL -- Inductive Graph Pattern Learning for Recommender Systems
===============================================================================

About
-----

IGPL is an inductive learning framework for recommender systems which uses graph structure features to make predictions. Graph structure features are some structure patterns related to ratings. For example, if a user $u_0$ likes an item $v_0$, we may expect to see very often that $v_0$ is also liked by some other user $u_1$ who shares a similar taste to $u_0$. By similar taste, we mean $u_1$ and $u_0$ have together both liked some other item $v_1$. In the bipartite graph built from the rating matrix, such a pattern is realized as a _like_ path connecting $(u_0,v_1,u_1,v_0)$. If there are many such paths between $u_0$ and $v_0$, we may infer that $u_0$ is highly likely to like $v_0$. Such paths are exactly some graph structure features useful for rating prediction. 

Instead of using predefined graph structure features, IGPL learns general graph structure features from local enclosing subgraphs around ratings based on a graph neural network (GNN). In other words, these local enclosing subgraphs around ratings are used as these ratings' characteristic graph representations which are directly fed to a GNN to make predictions. This idea has been successfully used in [link prediction](https://github.com/muhanzhang/SEAL). This work proves its effectiveness for recommender systems.

IGPL is completely different from transductive matrix factorization methods, where the learned latent features are associated with specific users/items and thus not generalizable to unseen ones. The model learned by IGPL is inductive, meaning that it can be applied to unseen users and items and can be transferred to new tasks. IGPL achieves state-of-the-art results on several benchmark datasets, outperforming other GNN approaches such as GC-MC and sRGCNN.

Requirements
------------

Tested with Python 3.6, Pytorch 1.0.0.

Install [PyTorch](https://pytorch.org/) >= 1.0.0

Install [PyTorch_Geometric](https://rusty1s.github.io/pytorch_geometric/build/html/notes/installation.html)

Other required python libraries: numpy, scipy, pandas, h5py, networkx, tqdm etc.

Usages
------

To train and test on Flixster, type:

    python Main.py --data-name flixster --hop 1 --epochs 40 --testing

The results will be saved in "results/flixster\_testmode/". The processed enclosing subgraphs will be saved in "data/flixster/testmode/". Change *flixster* to *douban* or *yahoo\_music* to do the same experiments on Douban and YahooMusic datasets, respectively. Delete _--testing_ to evaluate on validation set to do hyperparameter tuning.

To train and test on MovieLens, type:

    python Main.py --data-name ml_100k --save-appendix _mnph100 --data-appendix _mnph100 --max-nodes-per-hop 100 --hop 1 --epochs 60 --testing

The results will be saved in "results/ml\_100k\_mnph100\_testmode/". The processed enclosing subgraphs will be saved in "data/ml\_100k\_mnph100/testmode/". The *--max-nodes-per-hop* limits the number of maximum users or items to include in each hop when constructing an enclosing subgraph, which is based on random sampling. It effectively reduces the enclosing subgraph sizes for dense graphs such as MovieLens.

To repeat the transfer learning experiment in the paper (transfer a model pretrained on Flixster to Douban), first type the following:

    python Main.py --data-name flixster --save-appendix _stdrating --data-appendix _stdrating --hop 1 --epochs 40 --testing --standard-rating 

to train a model on Flixster by rounding Flixster's rating types to standard rating types 1, 2, 3, 4, 5. Then type:

    python Main.py --data-name douban --hop 1 --epochs 40 --testing --transfer --model-pos results/flixster_stdrating_testmode/model_checkpoint40.pth --k 41

to apply the pretrained model to Douban. 

To repeat the sparsity experiment in the paper (sparsify MovieLens' rating matrix to keep 20% ratings only), type the following:

    python Main.py --data-name ml_100k --save-appendix _ratio20_mnph100 --data-appendix _ratio20_mnph100 --ratio 0.2 --max-nodes-per-hop 100 --hop 1 --epochs 60 --testing

Then modify _--ratio 0.2_ to change the sparsity ratios.

After training a model on a dataset, to visualize the testing enclosing subgraphs with the highest and lowest predicted ratings, type the following (we use Flixster as an example):

    python Main.py --data-name flixster --hop 1 --epochs 40 --testing --visualize --keep-old

It will load "results/flixster\_testmode/model\_checkpoint40.pth" and save the visualization in "results/flixster\_testmode/visualization_flixster.pdf".

Check "Main.py" and "train\_eval.py" for more options and functions to play with. Check "models.py" to see the used graph neural network.

Reference
---------

If you find the code useful, please cite our paper.

    @article{zhang2019inductive,
      title={Inductive Graph Pattern Learning for Recommender Systems Based on a Graph Neural Network},
      author={Zhang, Muhan and Chen, Yixin},
      journal={arXiv preprint arXiv:1904.12058},
      year={2019}
    }

Muhan Zhang, Washington University in St. Louis
muhan@wustl.edu
4/26/2019
