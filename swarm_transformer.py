import argparse
import math
import time
import os
import sys

import numpy as np
import torch

import matplotlib

from tools import create_location_features_2d, factors

matplotlib.use('agg') # make sure to import this before traces and pyplot
from matplotlib.pyplot import figure, colorbar, imshow, show, plot


from torch import nn, optim
from torch.distributions import Multinomial
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from swarmlayer import SwarmLayer

from traces import Trace


import traceback
import sys



class SwarmTransformer(nn.Module):

    def __init__(self, cells, n_emb, C, H, W,
                 K = None,
                 learnable_location_features = False,
                 ):
        """
        Create a SwarmTransformer module for generative modeling of images
        :param cells: a list of SwarmConvLSTMCell
        :param n_emb: size of positional embeddings and class conditional embeddings
        :param C: number of image channels
        :param H: image height in pixels
        :param W: image width in pixels
        :param K: number of classes (for class conditional generation)
        :param learnable_location_features: if True, learn the location features otherwise use sinusoids
        """
        super().__init__()

        self.cells = nn.Sequential(*cells)
        self.n_emb = n_emb
        # it has to be multiple of 4 because we have per frequency (sine, cosine)x(vertical,horizontal)
        assert (self.n_emb//4)*4 == self.n_emb

        # RBG/gray value embedding (8bit images hard coded here)
        self.input_embedding = nn.Embedding(256, n_emb)
        self.input_embedding.weight.data.uniform_(-0.1, 0.1)

        if K is not None:
            # class conditional embeddings have the same size as location features (will be added later)
            self.cond_embedding = nn.Embedding(K, n_emb)
            self.cond_embedding.weight.data.uniform_(-0.1, 0.1)
        else:
            self.cond_embedding = None

        self.ce_loss = nn.CrossEntropyLoss()

        if K is not None:
            assert K > 0
        self.cond = K

        self.n_channels = C
        if self.n_channels>1:
            # learnable RGB-channel embedding
            self.channel_embedding = nn.Parameter(torch.zeros((self.n_emb,self.n_channels), dtype=torch.float32))
        else:
            self.channel_embedding = None

        self.learnable_location_features = learnable_location_features
        if self.learnable_location_features:
            self.location_features = nn.Parameter(0.001*torch.randn(self.n_emb, H, W), requires_grad=True)
        else:
            self.location_features = nn.Parameter(create_location_features_2d(H, W, self.n_emb), requires_grad=False)




    def prepare_data(self, X, Y=None):
        """
        Prepare input data to be used for training. In order to use a 2d SwarmLSTMConvCell, the input's W and C
        dimensions are flattened
        :param X: channel-first batch of images, size (N,C,H,W)
        :param Y: batch of labels, size (N)
        :return: X_in, X_out  (X_in: (N,n_emb,H,W*C), X_out: (N,H,W,C))
        """
        N,C,H,W = X.size()

        # 1. compute input embeddings for X
        X_in = self.input_embedding(X) # (N,C,H,W,Demb)
        X_in = X_in.transpose(4,1)     # (N,Demb,H,W,C)

        # 2. shift input by one to enforce causality
        X_in = X_in.contiguous().view((N, self.n_emb, -1))
        X_in = torch.cat( (torch.zeros_like(X_in[:,:, 0:1]),X_in[:,:,:-1]), dim=2)
        X_in = X_in.view((N, self.n_emb, H, W,C)).contiguous()

        # 3. compute location features
        F = self.location_features
        Df = F.size()[0]
        F_in = F.view((1,Df,H,W,1)).expand((1,Df,H,W,C))

        X_in = X_in+F_in

        # 4. compute class conditional  features
        if self.cond_embedding is not None:
            assert Y is not None
            Y_in = self.cond_embedding(Y) # (N,Demb)
            Y_in = Y_in.view( (N, self.n_emb, 1,1,1))
            X_in = X_in+Y_in

        # 5. compute channel embeddings
        if self.channel_embedding is not None:
            assert C == self.n_channels
            X_in = X_in + self.channel_embedding.view((1,self.n_emb,1,1,self.n_channels))

        # 6. flatten W and C channels in order to use a2d SwarmConvLSTMCell
        X_in = X_in.view((N, self.n_emb, H, W*C))

        # output is the raw input with channels last
        X_out = X.transpose(1,2).transpose(2,3)
        return X_in, X_out



    def forward(self, x, y=None):

        N, C, H, W = x.size()
        X_in, X_out = self.prepare_data(x,y)


        logits = self.cells(X_in)
        # note, W and C dimensions are flattened, logits are (N,n_out,H,W*C)
        # reshaping them back now
        logits = logits.view( -1, 256, H,W,C)
        loss = self.ce_loss(logits, X_out)

        return loss, logits






def create_datasets(batch_size,  name='MNIST'):

    ds={}
    ds['MNIST'] = datasets.MNIST
    ds['FashionMNIST'] = datasets.FashionMNIST
    ds['CIFAR10'] = datasets.CIFAR10
    ds['BWCIFAR'] = datasets.CIFAR10
    ds['SMALL'] = datasets.CIFAR10
    ds['CIFAR100'] = datasets.CIFAR100

    if name=='BWCIFAR':
        transform = transforms.Compose([transforms.ToTensor(),
                                         lambda x: (torch.mean(x, dim=0,keepdim=True) * 255).long()
                                        ])
    elif name == 'SMALL':
            transform = transforms.Compose([transforms.ToTensor(),
                                            lambda x: (x*255).long()[:,8:24,8:24]
                                            ])
    else:
        transform = transforms.Compose([ transforms.ToTensor(),
                                         #transforms.Normalize((0.,), (1./255,)),
                                         lambda x: (x*255).long()
                                       ])

    ds_train = ds[name]('./data/'+name, train=True, download=True, transform=transform)
    ds_val   = ds[name]('./data/'+name, train=False, transform=transform)

    dl_train = DataLoader( ds_train, batch_size=batch_size, shuffle=True)
    dl_val   = DataLoader( ds_val,   batch_size=batch_size, shuffle=True)

    return dl_train, dl_val




def create_sample_fn( model, C,H,W, K,  device):
    """
    create a function that produces a sample plot of the model during training
    :param model: the model
    :param C: number of RBG channels
    :param H: height in pixels
    :param W: width in pixels
    :param K: number of classes (or None)
    :param device:
    :return: sample function, that can be called without parameters and returns a figure handle
    """
    def sample_fn():
        fig = figure(figsize=(12,5))
        model.eval()

        if K is None:
            n_samp = 12
            Y = None
        else:
            n_samp = 2*K
            Y = torch.arange(2*K, device=device)%K

        X = torch.zeros( n_samp,C,H,W, device=device).long()

        with torch.no_grad():
            for h in range(H):
                for w in range(W):
                    for c in range(C):
                        _,logits = model(X,Y)

                        m = Multinomial(logits=logits[:,:,h,w,c])
                        X_ = m.sample(torch.Size([]))
                        X[:,c,h,w] = torch.argmax(X_,dim=1)

        X = X.cpu().numpy()
        if C>1:
            X = X.astype('float')/255.0
            _ = imshow(X.reshape(2, n_samp//2, C, H, W).transpose(0, 3, 1, 4, 2).reshape(2 * H, n_samp//2 * W, C))
        else:
            _ = imshow(X.reshape(2, n_samp//2, H, W).transpose(0, 2, 1, 3).reshape(2 * H, n_samp//2 * W))
        colorbar()

        return fig

    return sample_fn



from parse import parse
class ModelName(object):
    #name_template  = "%s-%d-%s-%d-%d-wc%.0f-lr%f"
    # like "CIFAR10-2-relu-12-5-wc60-lr0.01"
    name_template  = "{}-{:d}-{}-{:d}-{:d}-wc{:g}-lr{:g}-bs{:d}-{}"

    def create(self, opt):
        name=ModelName.name_template.format(opt.data, opt.n_layers, opt.non_lin, opt.n_hidden,
                                            opt.n_iter, opt.wc, opt.lr, opt.bs, opt.p)
        return name

    def parse(self, name, opt):
        print(opt)
        res = parse(ModelName.name_template, name)
        if res is None:
            raise ValueError("Could not parse model name {}".format(name))
        (opt.data, opt.n_layers, opt.non_lin, opt.n_hidden,
         opt.n_iter, opt.wc, opt.lr, opt.bs, opt.p) = tuple(res)
        return opt



def validate(model, dl_val, device):
    """
    run a complete validation epoch
    :param model:
    :param dl_val:
    :param device:
    :return: validation loss
    """
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X, Y in dl_val:
            X = X.to(device)
            Y = Y.to(device)
            loss, _ = model(X, Y)
            loss = loss.mean()
            val_loss += loss.item()
    val_loss /= len(dl_val)
    return val_loss


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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', type=str, choices=['MNIST',
                                                    'FashionMNIST',
                                                    'CIFAR10',
                                                    'CIFAR100',
                                                    'BWCIFAR',
                                                    'SMALL'],
                        help='dataset to be used in the experiment')


    parser.add_argument('-n_hidden', type=int, default=128,
                        help='number of hidden units inside the model')
    parser.add_argument('-n_layers', type=int, default=1,
                        help='number of layers for mult-layered models')
    parser.add_argument('-n_iter',   type=int, default=5,
                        help='number of iterations to be done in Swarm layers')

    parser.add_argument('-non_lin',  default='relu', choices=['relu', 'elu', 'lrelu'],
                        help='non-linearity used between different layers')

    parser.add_argument('-bs', type=int,    default=100,
                        help='batch size')
    parser.add_argument('-wc', type=float,  default=60,
                        help='allowed wall clock time for training (in minutes)')
    parser.add_argument('-update_interval', type=float, default=10,
                        help='update interval to generate trace and sample plots (in minutes)')

    parser.add_argument('-lr', type=float, default=0.01,
                        help='learning rate')

    parser.add_argument('-no_cuda', action='store_true',
                        help='dont use CUDA even if it is available')

    parser.add_argument('-name', type=str, default=None,
                        help='you can provide a model name that will be parsed into cmd line options')

    parser.add_argument('-dry_run', action='store_true',
                        help='just print out the model name and exit')

    parser.add_argument('-to_stdout', action='store_true',
                        help='log all output to stdout instead of modelname/log')

    parser.add_argument('-bt_horizon', type=float, default=0.1,
                        help='backtracking horizon')
    parser.add_argument('-bt_alpha',   type=float, default=0.9,
                        help='backtracking learning rate discount factor')

    parser.add_argument('-cond', action='store_true',
                        help='do class conditional modeling')

    parser.add_argument('-resume', type=str, default=None,
                        help='resume model from modelname/best.pkl')

    parser.add_argument('-learn_loc', type=bool, default=False)


    opt = parser.parse_args()

    if opt.name is not None:
        opt = ModelName().parse(opt.name, opt)
    name = ModelName().create(opt)
    assert opt.name is None or name==opt.name


    print(name)
    if opt.dry_run:
        exit()

    import sys

    name_part = name+".part"
    try:
        os.mkdir(name_part)
    except:
        pass

    if not opt.to_stdout:
        sys.stdout = open(name_part+'/log', 'w')


    opt.cuda = not opt.no_cuda

    C,H,W,K = {'MNIST':(1,28,28,10),
             'FashionMNIST':(1,28,28,10),
             'CIFAR10':(3,32,32,10),
             'CIFAR100':(3,32,32,100),
             'BWCIFAR':(1,32,32,10),
             'SMALL': (3,16,16,10),
               } [opt.data]
    n_classes = 256  # not dependent on the dataset so far

    non_linearity = {'elu':nn.ELU(), 'relu':nn.ReLU(), 'lrelu':nn.LeakyReLU()} [opt.non_lin]


    n_in = opt.n_hidden
    n_hidden = opt.n_hidden
    n_layers = opt.n_layers
    n_iter   = opt.n_iter


    # in case the desired batch size does not fit into CUDA memory
    # do batch iteration. Try in a loop the largest batch size nad batch_iter=1 first.
    # Decrease batch_size (increase batch_iter) by common factors until there is a model that does not throw an
    # out-of-memory error
    for batch_iter in factors(opt.bs):

        print(type(opt.bs),type(int(opt.bs//batch_iter)))

        print("trying batch size {} in {} iterations".format(opt.bs//batch_iter ,batch_iter))

        try:
            layers = []
            n_out_last = n_in
            for i in range(n_layers):
                if i<n_layers-1:
                    layers.append( SwarmLayer(n_in=n_out_last, n_out=n_hidden, n_hidden=n_hidden, n_iter=n_iter, pooling='CAUSAL'))
                    layers.append( non_linearity)
                    n_out_last = n_hidden
                else:
                    layers.append( SwarmLayer(n_in=n_out_last, n_out=n_classes, n_hidden=n_hidden, n_iter=n_iter, pooling='CAUSAL'))


            model = SwarmTransformer(layers, C=C, W=W, H=H, K=K, n_emb=n_in,
                                     learnable_location_features=opt.learn_loc)


            device = torch.device('cuda' if opt.cuda else 'cpu')
            if torch.cuda.device_count()>1:
                model = nn.DataParallel(model)

            model.to(device)


            print(model)

            print("backtracking {}% epochs with lr decrease factor {}".format(100*opt.bt_horizon, opt.bt_alpha))

            # create datasets with batch sizes split by batch_iter
            dl_train, dl_val = create_datasets( int(opt.bs//batch_iter), opt.data)

            sample_fn = create_sample_fn( model, C,H,W,K, device)

            optimizer = optim.Adam(model.parameters(), lr=opt.lr)

            if opt.resume is not None:
                resume(model, optimizer, opt.resume)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = opt.lr

            # create a tracing object, that records training and validation losses and other metrics and records 13 individual
            # weights of every model parameter tensor
            # every now and then it plots learning curves, weight traces and model samples to
            # modelname/[metrics.png,weights.png,samples.png] respectively
            traces = Trace(model, 13, sample_fn, name=name_part, columns=4)


            best_val_loss = math.inf
            val_loss_history = [np.inf]

            t_start = time.time()
            t_update = 0      # timer to count when the next traces update is due
            t_no_training = 0 # time spend generating traces and samples
            e = 0             # count the epochs
            while True:
                # inform the Traces object that a new epoch has begun
                traces.on_epoch_begin(e)

                for i, (X, Y) in enumerate(dl_train):
                    X = X.to(device)
                    Y = Y.to(device)
                    model.train()

                    if i%batch_iter==0:
                        optimizer.zero_grad()
                        norm = 0

                    loss, _ = model(X, Y)
                    loss = loss.mean()

                    (loss/batch_iter).backward()

                    if (i+1)%batch_iter==0:
                        # do an optimizer update step only every batch_iter iterations
                        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), math.inf, norm_type=1)
                        optimizer.step()

                    print(i, "%.4f (norm=%.4g)" % (loss.item(), norm), end="\r")

                    # a dictionary of values and metrics that will be logged by the Traces opbject
                    logs = {'loss': loss.item(), 'norm': norm}

                    time_is_up = time.time()>t_start+60*opt.wc + t_no_training #or i>=250
                    if time_is_up:
                        print("preparing to complete")

                    if i+1 == len(dl_train) or time_is_up:
                        # we are done with the last iteration
                        # -> kick off a validation epoch now and add the val_loss to the log
                        val_loss = validate(model, dl_val, device)
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
                if last_worse < min(e * (1.0 - opt.bt_horizon), e - 5) or val_loss > max(val_loss_history):
                    # the last validation result that was worse than this lays more than bt_horizon% epochs back
                    # or this validation loss is worse than everything before
                    # -> we will discard this model and backtrack to the best we had so far

                    if not time_is_up:
                        # but not if computation time is already up
                        checkpoint_path = name_part+"/best.pkl"

                        keep_lr = [param_group['lr']  for param_group in optimizer.param_groups]

                        resume( model, optimizer, checkpoint_path, name)

                        # once we backtracked, we decrease learning rate by factor bt_alpha
                        for param_group,lr in zip(optimizer.param_groups, keep_lr):
                            param_group['lr'] = opt.bt_alpha*lr

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

                    checkpoint_path = name_part+"/best.pkl"
                    torch.save(checkpoint, checkpoint_path)

                    best_val_loss = val_loss


                if time.time() > t_update + opt.update_interval*60 or time_is_up:
                    # it's time to plot some learning curves, weight traces, and sample figures
                    # this can take some time, so we don't do it all to often
                    t_no_training = t_no_training - time.time()

                    # this does the actual magic
                    traces.on_epoch_end(e)

                    # reset the update counter and record how much time we have spent here,
                    # this will not account for the training time budget
                    t_update = time.time()
                    t_no_training = t_no_training + time.time()
                e += 1

                if time_is_up:
                    break

            print("{}s spent preparing traces and samples".format(t_no_training))
            os.rename(name_part, name)

            break # the loop over batch iterations

        except RuntimeError:
            print("failed with batch size {}".format(opt.bs/batch_iter))
            exc_info = sys.exc_info()
            try:
                del model
            except NameError:
                pass
        finally:
            # Display the *original* exception
            traceback.print_exception(*exc_info)
            del exc_info

if __name__== "__main__":
    main()
