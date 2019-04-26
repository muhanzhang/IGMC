IGPL -- Inductive Graph Pattern Learning from Recommender Systems
===============================================================================

About
-----

IGPL is an inductive learning framework that learns graph structure features for recommender systems. Graph structure features are some repeated structure patterns related to ratings. For example, if a user $u_0$ likes an item $v_0$, we may expect to see very often that $v_0$ is also liked by some other user $u_1$ who shares a similar taste to $u_0$. By similar taste, we mean $u_1$ and $u_0$ have together both liked some other item $v_1$. In the bipartite graph, such a pattern is realized as a _like_ path connecting $(u_0,v_1,u_1,v_0)$. If there are many such paths between $u_0$ and $v_0$, we may infer that $u_0$ is highly likely to like $v_0$. Such paths are exactly graph structure features useful for rating prediction. IGPL learns general graph structure features from local subgraphs around ratings based on a graph neural network (GNN). Different from transductive matrix factorization methods, the model learned by IGPL is inductive, meaning it can be applied to unseen users and items and can be transferred to new tasks. IGPL has achieved new state-of-the-art results on several benchmark datasets, outperforming other GNN approaches such as GC-MC and sRGCNN.

Installation
------------

Install [PyTorch](https://pytorch.org/) >= 1.0.0

Install [PyTorch_Geometric](https://rusty1s.github.io/pytorch_geometric/build/html/notes/installation.html)

Type

    bash ./install.sh



Usages
------

Type "python Main.py" to have a try of SEAL on the USAir network.

Type:

    python Main.py --data-name NS --test-ratio 0.5 --hop 'auto' --use-embedding

to run SEAL on the NS network with 50% observed links randomly removed as testing links, hop number set automatically from {1, 2}, and node2vec embeddings included.

Type:

    python Main.py --data-name PPI_subgraph --test-ratio 0.5 --use-embedding --use-attribute

to run SEAL on PPI_subgraph with node attributes included. The node attributes are assumed to be saved in the  _group_ of the _.mat_ file.

Type:

    python Main.py --train-name PB_train.txt --test-name PB_test.txt --hop 1

to run SEAL on a custom splitting of train and test links, where each row of "PB_train.txt" is an observed training link, each row of "PB_test.txt" is an unobserved testing link. Note that links in "PB_train.txt" will be used to construct the observed network, yet it is not necessary to train SEAL on all links in "PB_train.txt" especially when the number of observed links is huge. To set a maximum number of links to train on, append "--max-train-num 10000" for example.

Sometimes even extracting 1-hop enclosing subgraphs for some links leads to unaffordable number of nodes in the enclosing subgraphs, especially in Twitter-type networks where a hub node can have millions of followers. To deal with this case, append "--max-nodes-per-hop 100" for example to restrict the number of nodes in each hop to be less than 100 using random sampling. SEAL still shows excellent performance.


Requirements
------------

Tested with Python 3.6, Pytorch 1.0.0.

Required python libraries: numpy, scipy, pandas, h5py, networkx, tqdm etc.

Reference
---------

If you find the code useful, please cite our paper.

Muhan Zhang, Washington University in St. Louis
muhan@wustl.edu
4/26/2019
