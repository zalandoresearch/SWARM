import argparse
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from scipy.stats import dirichlet


import matplotlib
from torch.optim.lr_scheduler import ExponentialLR, StepLR

from set_transformer import PoolingMultiheadAttention, MultiheadAttentionBlock
from swarmlayer import SwarmLayer

matplotlib.use('agg')  # make sure to import this before traces and pyplot
from matplotlib.pyplot import *

from traces import Trace


def get_options():
    parser = argparse.ArgumentParser()


    parser.add_argument('-n_hidden', type=int, default=128)
    parser.add_argument('-n_layers', type=int, default=1)
    parser.add_argument('-n_iter', type=int, default=5)
    parser.add_argument('-non_lin', default='relu', choices=['relu', 'elu', 'lrelu'])

    parser.add_argument('-n_dim', type=int, default=2)
    parser.add_argument('-n_clust', type=int, default=4)

    parser.add_argument('-pma', action='store_true')

    parser.add_argument('-dropout', type=float, default=0.0)

    parser.add_argument('-bs', type=int, default=10)
    parser.add_argument('-wc', type=float, default=300)  # wall clock time limit for training
    parser.add_argument('-update_interval', type=float,
                        default=5)  # update interval to generate trace and sample plots

    parser.add_argument('-max_epochs', type=int, default=100)

    parser.add_argument('-bt_horizon', type=float, default=1.0)
    parser.add_argument('-bt_alpha',   type=float, default=0.8)

    parser.add_argument('-lr', type=float, default=0.01)
    parser.add_argument('-gamma', type=float, default=0.97)

    parser.add_argument('-no_cuda', action='store_true')

    # parser.add_argument('-name', type=str, default=None)

    parser.add_argument('-dry_run', action='store_true')  # just print out the model name and exit

    parser.add_argument('-to_stdout', action='store_true')

    parser.add_argument('-run', type=int, default=0)

    parser.add_argument('-resume', type=str, default=None)

    # parser.add_argument('-epoch', type=int, default=10)
    # parser.add_argument('-batch_size', type=int, default=64)
    #
    # #parser.add_argument('-d_word_vec', type=int, default=512)
    # parser.add_argument('-d_model', type=int, default=512)
    # parser.add_argument('-d_inner_hid', type=int, default=2048)
    # parser.add_argument('-d_k', type=int, default=64)
    # parser.add_argument('-d_v', type=int, default=64)
    #
    # parser.add_argument('-n_head', type=int, default=8)
    # parser.add_argument('-n_layers', type=int, default=6)
    # parser.add_argument('-n_warmup_steps', type=int, default=4000)

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    opt.n_in = 2
    opt.n_out = 10  # 2d, 10 clusters, other dimensions not yet supportrd

    try:
        i = sys.argv.index('-resume')
        sys.argv.remove(sys.argv[i + 1])
        sys.argv.remove(sys.argv[i])
    except:
        pass

    opt.name = '.'.join(sys.argv[1:]).replace('.-', '-')

    return opt


def masked_mean(x,mask=None):
    if mask is None:
        pool = x.mean(dim=1)
    else:
        mask = mask.float()
        pool = torch.sum( x*mask.unsqueeze(2), 1) / torch.sum(mask,1).unsqueeze(1)
    return pool

LOG2PIHALF = np.log(2*np.pi)/2

class gauss_llh(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, mu, si):
        ctx.save_for_backward( X, mu, si)
        l = -0.5 * ((X.unsqueeze(3) - mu)) ** 2 / si - 0.5 * torch.log(si) - LOG2PIHALF
        l = l.sum(2)
        return l

    @staticmethod
    def backward( ctx, delta):
        X, mu, si = ctx.saved_tensors

        #print(X.size(), mu.size(), si.size(),delta.size())
        dmu = (X.unsqueeze(3)-mu)*delta.unsqueeze(2)
        dmu = dmu.mean(dim=1,keepdim=True)

        dsi = si**2*((X.unsqueeze(3)-mu)**2-si)*delta.unsqueeze(2)
        dsi = dsi.mean(dim=1,keepdim=True)

        return None, dmu, dsi

def MOG_llh(X, mu, si, pi, mask=None):

    #l = -0.5*((X.unsqueeze(3)-mu))**2/si - 0.5*torch.log(si) - LOG2PIHALF
    l = -0.5*((X.unsqueeze(3)-mu)/si)**2 - torch.log(si) - LOG2PIHALF

    l = l.sum(2)

    #l = gauss_llh.apply(X,mu,si)

    lpi = l+pi
    lpi_max = torch.max(lpi, dim=2, keepdim=True)[0]
    lpi = lpi-lpi_max

    llh = torch.log(torch.exp(lpi).sum(2))+lpi_max.squeeze(2)
    llh = masked_mean(llh.unsqueeze(2),mask)

    return llh


class GatedPooling(nn.Module):

    def __init__(self, dg, dv):

        super().__init__()

        self.dg = dg
        self.dv = dv

    def forward(self, x):

        N,E,D = x.size()
        i=0
        g = x[:, :, :self.dg].view(N, E, self.dg, 1)
        v = x[:, :, self.dg:].view(N, E, self.dg, self.dv)

        y = torch.sum(torch.softmax(g, dim=1)*v, dim=1) # (N,dg,dv)

        return y

class SwarmMOG(nn.Module):
    def __init__(self, n_in, n_out, n_hidden, n_iter, n_layers,
                 n_clust=4,
                 n_dim=2,
                 non_lin = 'relu',
                 pma = False):
        super().__init__()

        self.n_clust = n_clust
        self.n_dim = n_dim

        assert n_out == n_clust*(2*n_dim+1), f'{n_out}, {n_clust}, {n_dim}'

        self.swarm=nn.ModuleList()
        last_out = n_in
        next_in = n_hidden
        for i in range(n_layers):
            if i == n_layers-1:
                next_in =  n_out if not pma else 128
            swarm = SwarmLayer(last_out, next_in, n_hidden, n_iter=n_iter, n_dim=1, channel_first=False)
            last_out = next_in

            swarm.cell.Wih.bias.data.zero_()
            swarm.ffwd.bias.data.zero_()

            self.swarm.append(swarm)

        nonlinearities = {}
        nonlinearities['relu'] = nn.ReLU()
        nonlinearities['elu'] = nn.ELU()
        nonlinearities['lrelu'] = nn.LeakyReLU()

        #self.non_lin = nonlinearities[non_lin]

        # self.out = nn.Sequential( #nn.Linear(n_hidden, n_hidden),
        #                           #nonlinearities[non_lin],
        #                           #nn.Linear(n_hidden, n_hidden),
        #                           nonlinearities[non_lin],
        #                           nn.Linear(n_hidden, n_out) )

        if pma:
            self.pma = PoolingMultiheadAttention( d=128, k=n_clust, h=4, rff=nn.Linear(128,128))
            self.sab = MultiheadAttentionBlock( d=128,  h=4, rff=nn.Linear(128,128))
            self.out = nn.Linear(n_clust*128,n_out)
        else:
            self.pma = None
        #self.swarm_dec = SwarmLayer(128, n_out//n_clust, 128,  n_iter=5, n_dim=1, channel_first=False)

        #self.gp = GatedPooling(n_clust, 2*n_dim+1)


    def forward(self, X, mask=None):
        # X is (N,E,2)

        last = X
        for i,swarm in enumerate(self.swarm):
            if i==0:
                last = X
            else:
                last = self.non_lin(last)

            last = swarm(last, mask)

        #out_gate = torch.softmax(last[:,:,:self.n_clust], dim=1) # (N,E,n_clust)
        #out_data = last[:,:,self.n_clust:]                      # (N,E,128)

        #pool = torch.matmul( out_gate.transpose(1,2), out_data)
        #print(pool.size())

        #pool = self.pma(last)
        #print(pool.size())
        #out = self.swarm_dec(pool)
        #print(pool.size())


        #pool = masked_mean(last, mask)
        #pool = self.out(pool)



        # out = self.swarm_dec( pool.view(-1,4,32)) # (N,4,5)
        #out = pool.view(-1, self.n_clust, 2*self.n_dim+1).contiguous()

        if self.pma is None:
            out = masked_mean(last, mask)
        else:
            last = self.pma(last)
            last = self.sab(last,last)
            out = self.out(last.view(-1,self.n_clust*128))

        out = out.view(-1, self.n_clust,2*self.n_dim+1)

        mu = out[:, :, :self.n_dim].transpose(1, 2).view(-1, 1, self.n_dim, self.n_clust)
        si = out[:, :, self.n_dim:2*self.n_dim].transpose(1, 2).view(-1, 1, self.n_dim, self.n_clust)
        si = nn.functional.softplus(si)
        pi = out[:, :, 2*self.n_dim].view(-1, 1, self.n_clust)
        pi = torch.log_softmax(pi, dim=2)

        llh = MOG_llh(X, mu, si, pi, mask)

        return llh.mean(), mu, si, pi


def create_model(opt):
    model = SwarmMOG( opt.n_dim, opt.n_clust*(2*opt.n_dim+1), opt.n_hidden, opt.n_iter, opt.n_layers, opt.n_clust, opt.n_dim, opt.non_lin, opt.pma)
    return model


def data_fn(n_tasks=10,
            n_clust=4,
            n_entities=(100, 500),
            n_dim=2,
            device=None,
            return_params=False):
    n_ent = np.random.randint(*n_entities)

    x = np.random.randn(n_tasks, n_ent, n_dim) * 0.3
    mu = np.random.rand(n_tasks, n_dim, n_clust) * 8 - 4
    pi = dirichlet.rvs(np.ones(n_clust), n_tasks)
    idx = np.zeros((n_tasks, n_ent), dtype=np.long)

    for i in range(n_tasks):
        idx[i] = np.random.choice(n_clust, n_ent, p=pi[i])
        x[i] += mu[i, :, idx[i]]

    x = torch.tensor(x).float().to(device)
    idx = torch.tensor(idx).to(device)

    if not return_params:
        return x, idx, None

    si = 0.3 * torch.ones(n_tasks, n_dim, n_clust).float().to(device)
    mu = torch.tensor(mu).float().to(device)
    pi = torch.log(torch.tensor(pi).float()).to(device)
    return x, idx, None, mu, si, pi



def create_sample_fn(model, data_fn, device):
    def sample_fn():

        X, idx, mask = data_fn(device=device, n_tasks=9 )
        _, mu, si, pi = model(X, mask)

        def ellipse(mu, si, *args, **kwargs):
            l = np.linspace(0, 2 * np.pi)
            plot(np.cos(l) * si[0] + mu[0], np.sin(l) * si[1] + mu[1], *args, **kwargs)

        fig = figure(figsize=(20, 20))
        for n in range(9):
            subplot(3, 3, n + 1)
            x = X[n].data.cpu().numpy()
            plot(x[:, 0], x[:, 1], 'k.')

            mu_ = mu[n, 0].data.cpu().numpy()
            si_ = si[n, 0].data.cpu().numpy()
            plot(mu_[0], mu_[1], 'o')

            for j in range(4):
                ellipse(mu_[:, j], si_[:, j], '-')

            gca().set_xlim(-5, 5)
            gca().set_ylim(-5, 5)
            grid(True)
        return fig

    return sample_fn


def resume(model, optimizer, checkpoint_path, name=None):
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


def train(model, optimizer, sched, traces, opt, data_fn, device, name=None):
    wc = opt.wc

    bt_alpha = opt.bt_alpha
    bt_horizon = opt.bt_horizon
    update_interval = opt.update_interval

    max_epochs = opt.max_epochs

    if name is None:
        name = opt.name
        name_part = name + '.part'

    def validate(model, data_fn, device, N=100):
        model.eval()
        val_loss = 0
        # for X, idx, mask in dl_val:
        for i in range(N):
            X, idx, mask = data_fn(device=device)
            loss = -model(X, mask)[0]
            val_loss += loss.item()
        val_loss /= N
        return val_loss

    best_val_loss = math.inf
    val_loss_history = [math.inf]

    t_start = time.time()
    t_update = 0
    t_no_training = 0  # time spend generating traces and samples

    for e in range(max_epochs):
        traces.on_epoch_begin(e)

        for i in range(500):
            model.train()
            optimizer.zero_grad()

            X, idx, mask = data_fn(n_tasks=opt.bs,device=device)
            model.train()
            optimizer.zero_grad()

            loss = -model(X, mask)[0]

            print(i, "%.4f" % loss.item(), end="\r")

            loss.backward()

            #param_backup = [p.data.clone() for p in model.parameters()]
            optimizer.step()

            # with torch.no_grad():
            #     loss_delta = loss + model(X, mask)[0]
            #
            #     if loss_delta.item()<0.:
            #         for p,p_ in zip( model.parameters(), param_backup):
            #             p.data.copy_(p_)


            logs = {'loss': loss.item()} #, 'loss_delta': loss_delta.item()}

            time_is_up = time.time() > t_start + 60 * wc + t_no_training  # or i>=250
            if time_is_up:
                print("preparing to complete")

            if i + 1 == 500 or time_is_up:
                val_loss = validate(model, data_fn, device)
                print("%d: val_loss = %.4f" % (e, val_loss))
                logs['val_loss'] = val_loss

            logs['lr'] = [p['lr'] for p in optimizer.param_groups]

            # print(i,loss.item())
            traces.on_batch_end(i, logs)

            sys.stdout.flush()

            if time_is_up:
                break

        sched.step()

        last_worse = np.argwhere(np.array(val_loss_history) > val_loss).max()
        print("last_worse", last_worse)
        if last_worse < min(e * (1.0 - bt_horizon), e - 5) or val_loss > max(val_loss_history) or np.isnan(val_loss):
            if not time_is_up:
                checkpoint_path = name_part + "/best.pkl"

                keep_lr = [param_group['lr'] for param_group in optimizer.param_groups]

                resume(model, optimizer, checkpoint_path, name)

                for param_group, lr in zip(optimizer.param_groups, keep_lr):
                    param_group['lr'] = bt_alpha * lr

                val_loss = checkpoint['best_val_loss']
                print("back tracking to {:g}".format(val_loss))

        val_loss_history.append(val_loss)

        if val_loss < best_val_loss:
            print("saving best model at val_loss={:g}".format(val_loss))
            checkpoint = {}
            checkpoint['best_val_loss'] = val_loss
            checkpoint['model'] = model.state_dict()
            checkpoint['optimizer'] = optimizer.state_dict()
            checkpoint['name'] = name

            checkpoint_path = name_part + "/best.pkl"
            torch.save(checkpoint, checkpoint_path)

            best_val_loss = val_loss

        if time.time() > t_update + update_interval * 60 or time_is_up or e == max_epochs - 1:
            t_no_training = t_no_training - time.time()

            traces.on_epoch_end(e)

            t_update = time.time()
            t_no_training = t_no_training + time.time()

        if time_is_up:
            break

    checkpoint_path = name_part + "/best.pkl"
    resume(model, optimizer, checkpoint_path, name)

    test_loss = validate(model, data_fn, device, N=1000)
    print("test_loss = %.4f" % test_loss)

    print("{}s spent preparing traces and samples".format(t_no_training))


def main():
    opt = get_options()
    name_part = opt.name + '.part'

    try:
        os.mkdir(name_part)
    except:
        pass
        # raise RuntimeError("could not create  directory {}".format(name_part))

    if not opt.to_stdout:
        sys.stdout = open(name_part + '/log', 'w')

    model = create_model(opt)

    device = torch.device('cuda' if opt.cuda else 'cpu')

    #if torch.cuda.device_count() > 1:
    #    model = nn.DataParallel(model)
    model.to(device)
    print(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("model has {} parameters".format(n_params))


    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    sample_fn = create_sample_fn(model, data_fn, device)
    #sched = ExponentialLR(optimizer, opt.gamma, last_epoch=-1)
    sched = StepLR(optimizer, step_size=70, gamma=0.1)

    if opt.resume is not None:
        resume(model, optimizer, opt.resume)
        for param_group in optimizer.param_groups:
            param_group['lr'] = opt.lr

    traces = Trace(model, 13, sample_fn, name=name_part, columns=4)

    train(model, optimizer, sched,  traces, opt, data_fn, device=device )

    os.rename(name_part, opt.name)


if __name__ == "__main__":
    main()

