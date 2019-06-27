import argparse
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import numpy as np

from clustering_dataset import to_one_hot, greedy_cross_entropy, create_data_loders


# figures shall no be plotted on screen but aggregated to png files
import matplotlib
matplotlib.use('agg') # make sure to import this before traces and pyplot
from matplotlib.pyplot import figure, scatter, gca, subplot, grid, get_cmap

from models import create_model
from traces import Trace



def get_options():
    """
    parse command line options
    :return: argparse option object
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-type',     type=str, choices=['Swarm',
                                                        'SetLinear',
                                                        'SetTransformer',
                                                        'LSTM',
                                                        'LSTMS'],
                        help='type of set-equivariant model to be used')

    parser.add_argument('-n_hidden', type=int, default=128,
                        help='number of hidden units inside the model')
    parser.add_argument('-n_layers', type=int, default=1,
                        help='number of layers for multi-layered models')
    parser.add_argument('-n_iter',   type=int, default=5,
                        help='number of iterations to be done in Swarm layers')

    parser.add_argument('-non_lin',  default='relu', choices=['relu', 'elu', 'lrelu'],
                        help='non-linearity used between different layers')

    parser.add_argument('-n_heads',      type=int, default=4,
                        help='number of attention heads in SetTransfomer')
    parser.add_argument('-n_ind_points', type=int, default=10,
                        help='number of inducing points if SetTransformer')

    parser.add_argument('-dropout',  type=float, default=0.0,
                        help='dropout rate')
    parser.add_argument('-tasks',    type=int, default=10000,
                        help='number of tasks to be created (training and validation together)')


    parser.add_argument('-bs', type=int,    default=100,
                        help='batch size')
    parser.add_argument('-wc', type=float,  default=60,
                        help='allowed wall clock time for training (in minutes)')
    parser.add_argument('-update_interval', type=float, default=10,
                        help='update interval to generate trace and sample plots (in minutes)')

    parser.add_argument('-max_epochs', type=int, default=1000,
                        help='maximum number of epochs in training')

    parser.add_argument('-bt_horizon', type=float, default=0.2,
                        help='backtracking horizon')
    parser.add_argument('-bt_alpha',   type=float, default=0.9,
                        help='backtracking learning rate discount factor')

    parser.add_argument('-lr', type=float, default=0.01,
                        help='learning rate')

    parser.add_argument('-no_cuda', action='store_true',
                        help='dont use CUDA even if it is available')


    parser.add_argument('-dry_run', action='store_true',
                        help='just print out the model name and exit')

    parser.add_argument('-to_stdout', action='store_true',
                        help='log all output to stdout instead of modelname/log')

    parser.add_argument('-run', type=int, default=0,
                        help='additional run id to be appended to the model name,'
                        'has no function otherwise')


    parser.add_argument('-resume', type=str, default=None,
                        help='resume model from modelname/best.pkl')


    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    opt.n_in = 2
    opt.n_out = 10 # 2d, 10 clusters, other dimensions not yet supported

    # remove '-resume' from argument list, because it shall not become part of the model name
    try:
        i = sys.argv.index('-resume')
        sys.argv.remove(sys.argv[i+1])
        sys.argv.remove(sys.argv[i])
    except:
        pass
    # model name will be concatenation of command line options with spaces escaped by '.'
    opt.name = '.'.join(sys.argv[1:]).replace('.-','-')

    return opt


def create_sample_fn(model, dl_val):
    """
    create a function that produces a sample plot of the model during training
    :param model: the model
    :param dl_val: DataLoader to retrieve sample tasks from. The first 3 tasks will be plotted.
    :return: sample function, that can be called without parameters and returns a figure handle
    """
    def sample_fn():
        fig = figure(figsize=(12, 10))
        model.eval()
        X, idx, mask = next(iter(dl_val))
        idx = to_one_hot(idx, 10)
        l = model(X, mask)
        X = X.data.cpu().numpy()
        l = l.data.cpu().numpy()
        idx = idx.data.cpu().numpy()
        mask = mask.data.cpu().numpy()

        for i in range(3):
            j = np.where(mask[i])[0]
            subplot(2, 3, i + 1)
            scatter(X[i, j, 0], X[i, j, 1], c=np.argmax(l[i, j], 1), s=2, cmap=get_cmap('tab10'))
            gca().set_aspect('equal')
            grid(True)
        for i in range(3):
            j = np.where(mask[i])[0]
            subplot(2, 3, 3 + i + 1)
            scatter(X[i, j, 0], X[i, j, 1], c=np.argmax(idx[i, j], 1), s=2, cmap=get_cmap('tab10'))
            gca().set_aspect('equal')
            grid(True)

        return fig

    return sample_fn



def resume(model, optimizer, checkpoint_path, name=None):
    """
    resume model parameters and optimizer state
    :param model: model to be resumed
    :param optimizer: optimizer to be resumed
    :param checkpoint_path: filename of the saved pkl file
    :param name: model name (must be identical to the name used in check point)
    """
    checkpoint = torch.load(checkpoint_path)

    if name is not None:
        assert checkpoint['name'] == name

    try:
        model.load_state_dict(checkpoint['model'])
    except:
        Warning("Could not resume model from {}".format(checkpoint_path))
    try:
        optimizer.load_state_dict(checkpoint['optimizer'])
    except:
        Warning("Could not resume optimizer from {}".format(checkpoint_path))



def train( model, optimizer, traces, opt, dl_train, dl_val, name=None):
    """

    :param model: model
    :param optimizer: optimizer object
    :param traces: instance of Traces class
    :param opt: parsed command line options
    :param dl_train: training DataLoader
    :param dl_val: validation DataLoader
    :param name: model name
    """
    wc = opt.wc

    bt_alpha = opt.bt_alpha
    bt_horizon = opt.bt_horizon
    update_interval = opt.update_interval

    max_epochs = opt.max_epochs

    if name is None:
        name = opt.name
        name_part = name+'.part'



    def validate(model, dl_val):
        """
        run a complete validation epoch
        :param model:
        :param dl_val:
        :return: validation loss
        """
        model.eval()
        val_loss = 0
        for X, idx, mask in dl_val:
            logits = F.log_softmax(model(X, mask), 2)
            idx = to_one_hot(idx, 10)

            loss, _ = greedy_cross_entropy(logits, idx, mask, 10)
            loss = -loss.mean()
            val_loss += loss.item()
        val_loss /= len(dl_val)
        return val_loss


    best_val_loss = math.inf
    val_loss_history = [math.inf]

    t_start = time.time()
    t_update = 0       # timer to count when the next traces update is due
    t_no_training = 0  # time spend generating traces and samples

    for e in range(max_epochs):

        # inform the Traces object that a new epoch has begun
        traces.on_epoch_begin(e)

        for i, (X, idx, mask) in enumerate(dl_train):

            model.train()
            optimizer.zero_grad()

            logits = F.log_softmax(model(X, mask), 2)
            idx = to_one_hot(idx, 10)

            loss, _ = greedy_cross_entropy(logits, idx, mask, 10)
            loss = -loss.mean()

            print(i, "%.4f" % loss.item(), end="\r")

            loss.backward()
            optimizer.step()

            # a dictionary of values and metrics that will be logged by the Traces object
            logs = {'loss': loss.item()}

            time_is_up = time.time() > t_start + 60 * wc + t_no_training  # or i>=250
            if time_is_up:
                print("preparing to complete")

            if i + 1 == len(dl_train) or time_is_up:
                # we are done with the last iteration
                # -> kick off a validation epoch now and add the val_loss to the log
                val_loss = validate(model, dl_val)
                print("%d: val_loss = %.4f" % (e, val_loss))
                logs['val_loss'] = val_loss

            logs['lr'] = [p['lr'] for p in optimizer.param_groups]

            # now actually log the metrics for iteration i
            traces.on_batch_end(i, logs)

            sys.stdout.flush()

            if time_is_up:
                break

        last_worse = np.argwhere(np.array(val_loss_history) > val_loss).max()
        print("last_worse", last_worse)

        if last_worse < min(e * (1.0 - bt_horizon), e - 5) or val_loss > max(val_loss_history):
            # the last validation result that was worse than this lays more than bt_horizon% epochs back
            # or this validation loss is worse than everything before
            # -> we will discard this model and backtrack to the best we had so far

            if not time_is_up:
                # but not if computation time is already up
                checkpoint_path = name_part + "/best.pkl"

                keep_lr = [param_group['lr'] for param_group in optimizer.param_groups]

                resume(model, optimizer, checkpoint_path, name)

                # once we backtracked, we decrease learning rate by factor bt_alpha
                for param_group, lr in zip(optimizer.param_groups, keep_lr):
                    param_group['lr'] = bt_alpha * lr

                val_loss = checkpoint['best_val_loss']
                print("back tracking to {:g}".format(val_loss))

        val_loss_history.append(val_loss)

        if val_loss < best_val_loss:
            # this model is better than every thing before,
            # -> let's save it as a check point
            print("saving best model at val_loss={:g}".format(val_loss))
            checkpoint = {}
            checkpoint['best_val_loss'] = val_loss
            checkpoint['model'] = model.state_dict()
            checkpoint['optimizer'] = optimizer.state_dict()
            checkpoint['name'] = name

            checkpoint_path = name_part + "/best.pkl"
            torch.save(checkpoint, checkpoint_path)

            best_val_loss = val_loss

        if time.time() > t_update + update_interval * 60 or time_is_up or e==max_epochs-1:
            # it's time to plot some learning curves, weight traces, and sample figures
            # this can take some time, so we don't do it all to often
            t_no_training = t_no_training - time.time()

            # this does the actual magic
            traces.on_epoch_end(e)

            # reset the update counter and record how much time we have spent here,
            # this will not account for the training time budget
            t_update = time.time()
            t_no_training = t_no_training + time.time()
        

        if time_is_up:
            break

    print("{}s spent preparing traces and samples".format(t_no_training))




def main() :
    """
    now, finally, let's do something
    """
    opt = get_options()
    # model name will be augmented with '.part' until it has completed training
    name_part = opt.name + '.part'

    try:
        os.mkdir(name_part)
    except:
        pass
        #raise RuntimeError("could not create  directory {}".format(name_part))

    if not opt.to_stdout:
        sys.stdout = open(name_part+'/log', 'w')



    model = create_model(opt)

    device = torch.device('cuda' if opt.cuda else 'cpu')

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    print(model)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("model has {} parameters".format(n_params))

    # create the data loaders with the specified number of tasks,
    # random seed defaults to 0 in create_data_loaders, so experiments will be reproducible
    dl_train, dl_val = create_data_loders(opt.tasks, device=device)

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    sample_fn = create_sample_fn( model, dl_val)

    if opt.resume is not None:
        resume(model, optimizer, opt.resume)
        for param_group in optimizer.param_groups:
            param_group['lr'] = opt.lr

    # create a tracing object, that records training and validation losses and other metrics and records 13 individual
    # weights of every model parameter tensor
    # every now and then it plots learning curves, weight traces and model samples to
    # modelname/[metrics.png,weights.png,samples.png] respectively
    traces = Trace(model, 13, sample_fn, name=name_part, columns=4)

    # now train the model
    train(model, optimizer, traces, opt, dl_train, dl_val)

    # when done, remove the '.part' from the model name
    os.rename(name_part, opt.name)


if __name__== "__main__":
    main()
