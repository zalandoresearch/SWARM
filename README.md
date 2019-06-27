SWARM
=====

**SWARM Layers** are a new approach to build powerful learnable set functions. These are multi-variate function over a set (or population) of entities that exhibit permutation equivariance or invariance, i.e. these functions are explicitly agnostic to the order of entities.

This repository contains a prototypical PyTorch implementation of SWARM Layers plus the source code for the experiments performed in the paper

***Learning Set-equivariant Functions with SWARM Mappings*** (<http://arxiv.org/abs/1906.09400>)


## Architecture
Building bocks for **SWARM Layers** are **SWARM Cells**. These are modified LSTM cells that are augmented with an additional input. Together with the network input *x* and the previous hidden state *h* they also receive an input *p* which resembles pooled information over the whole population. 

![SWARM cell image](https://github.com/zalandoresearch/SWARM/raw/master/swarm_cell.png "SWARM cell")

The same SWARM cell (S), shared over the whole population of entities, is executed repeatedly for a number of iterations, where the number of iterations is a model hyper parameter. The pooling operation (P) can be simple average pooling, or any other low level set-invariant function. To build a SWARM based Transformer-like architecture, causal average pooling is used (thereby giving up the strict permutation equivariance, for details see the paper). The iterations plus pooling together with an output fully connected layer form the SWARM Layer

![SWARM layer image](https://github.com/zalandoresearch/SWARM/raw/master/swarm.png "SWARM layer")

SWARM Layer can provide state-of-the-art performance in various applications **already with single layer architectures**. They can, of course, also be stacked to multi-layer architectures or can be combined with other set-equivariant functions. In particular, because of their permutation-equivariant semantics, they can be used as a direct replacement for self attention blocks in various architectures. In the recently proposed powerful Transformer architectures, they can lead to significantly smaller models, as we have shown in the paper.

## Experiments

Both experiments shown in the paper can be run from the command line with a Python 3 interpreter

### Amortized Clustering

To get an overview of the command line options run 

```
>$ python -u amortised_clustering.py -h
usage: amortised_clustering.py [-h]
                               [-type {Swarm,SetLinear,SetTransformer,LSTM,LSTMS}]
                               [-n_hidden N_HIDDEN] [-n_layers N_LAYERS]
                               [-n_iter N_ITER] [-non_lin {relu,elu,lrelu}]
                               [-n_heads N_HEADS] [-n_ind_points N_IND_POINTS]
                               [-dropout DROPOUT] [-tasks TASKS] [-bs BS]
                               [-wc WC] [-update_interval UPDATE_INTERVAL]
                               [-max_epochs MAX_EPOCHS]
                               [-bt_horizon BT_HORIZON] [-bt_alpha BT_ALPHA]
                               [-lr LR] [-no_cuda] [-dry_run] [-to_stdout]
                               [-run RUN] [-resume RESUME]

optional arguments:
  -h, --help            show this help message and exit
  -type {Swarm,SetLinear,SetTransformer,LSTM,LSTMS}
                        type of set-equivariant model to be used
  -n_hidden N_HIDDEN    number of hidden units inside the model
  -n_layers N_LAYERS    number of layers for multi-layered models
  -n_iter N_ITER        number of iterations to be done in Swarm layers
  -non_lin {relu,elu,lrelu}
                        non-linearity used between different layers
  -n_heads N_HEADS      number of attention heads in SetTransfomer
  -n_ind_points N_IND_POINTS
                        number of inducing points if SetTransformer
  -dropout DROPOUT      dropout rate
  -tasks TASKS          number of tasks to be created (training and validation
                        together)
  -bs BS                batch size
  -wc WC                allowed wall clock time for training (in minutes)
  -update_interval UPDATE_INTERVAL
                        update interval to generate trace and sample plots (in
                        minutes)
  -max_epochs MAX_EPOCHS
                        maximum number of epochs in training
  -bt_horizon BT_HORIZON
                        backtracking horizon
  -bt_alpha BT_ALPHA    backtracking learning rate discount factor
  -lr LR                learning rate
  -no_cuda              dont use CUDA even if it is available
  -dry_run              just print out the model name and exit
  -to_stdout            log all output to stdout instead of modelname/log
  -run RUN              additional run id to be appended to the model name,has
                        no function otherwise
  -resume RESUME        resume model from modelname/best.pkl
```

The best performing SWARM model was trained with the command line 

```
>$ python -u amortised_clustering.py -type Swarm -n_hidden 128 -n_iter 10 -n_layers 1 -dropout 0.0 -bs.50 -wc 60
```

### SWARM Image Transformer

To get an overview of the command line options run 

```
>$ python -u swarm_transformer.py -h
usage: swarm_transformer.py [-h]
                            [-data {MNIST,FashionMNIST,CIFAR10,CIFAR100,BWCIFAR,SMALL}]
                            [-n_hidden N_HIDDEN] [-n_layers N_LAYERS]
                            [-n_iter N_ITER] [-non_lin {relu,elu,lrelu}]
                            [-bs BS] [-wc WC]
                            [-update_interval UPDATE_INTERVAL] [-lr LR]
                            [-no_cuda] [-name NAME] [-dry_run] [-to_stdout]
                            [-bt_horizon BT_HORIZON] [-bt_alpha BT_ALPHA]
                            [-cond] [-resume RESUME] [-learn_loc LEARN_LOC]

optional arguments:
  -h, --help            show this help message and exit
  -data {MNIST,FashionMNIST,CIFAR10,CIFAR100,BWCIFAR,SMALL}
                        dataset to be used in the experiment
  -n_hidden N_HIDDEN    number of hidden units inside the model
  -n_layers N_LAYERS    number of layers for multi-layered models
  -n_iter N_ITER        number of iterations to be done in Swarm layers
  -non_lin {relu,elu,lrelu}
                        non-linearity used between different layers
  -bs BS                batch size
  -wc WC                allowed wall clock time for training (in minutes)
  -update_interval UPDATE_INTERVAL
                        update interval to generate trace and sample plots (in
                        minutes)
  -lr LR                learning rate
  -no_cuda              dont use CUDA even if it is available
  -name NAME            you can provide a model name that will be parsed into
                        cmd line options
  -dry_run              just print out the model name and exit
  -to_stdout            log all output to stdout instead of modelname/log
  -bt_horizon BT_HORIZON
                        backtracking horizon
  -bt_alpha BT_ALPHA    backtracking learning rate discount factor
  -cond                 do class conditional modeling
  -resume RESUME        resume model from modelname/best.pkl
  -learn_loc LEARN_LOC
```

The best performing model was trained with the command line 

```
>$ python -u swarm_transformer.py -data CIFAR10 -n_layers 1 -n_hidden 512 -n_iter 20 -wc 7200 -lr 0.002 -bs 20
>$ python -u swarm_transformer.py -data FashionMNIST -n_layers 1 -n_hidden 256 -n_iter 10 -wc 480 -lr 0.001 -bs 50
>$ python -u swarm_transformer.py -data MNIST -n_layers 1 -n_hidden 256 -n_iter 10 -wc 480 -lr 0.001 -bs 50
```



If you use the code in your scientific work, we are thankful if you could cite 

Vollgraf, Roland. *“Learning Set-equivariant Functions with SWARM Mappings.”*, arXiv:1906.09400 (2019).

The design of the figures above was inspired by Colah's excellent blog *"Understanding LSTM Networks"* <https://colah.github.io/posts/2015-08-Understanding-LSTMs/>
