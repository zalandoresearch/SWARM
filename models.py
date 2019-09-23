import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.dropout import _DropoutNd

from swarmlayer import SwarmLayer

from set_transformer import InducedSetAttentionBlock, RFF

nonlinearities = {}
nonlinearities['relu']  = nn.ReLU()
nonlinearities['elu']   = nn.ELU()
nonlinearities['lrelu'] = nn.LeakyReLU()


class MaskedSequential(nn.Module):
    """
    Build a sequential module out of modules that take a mask parameter in their forward() method. Known modules that
    don't take a mask argument (e.g. non-linearities) can be seamlessly included.
    """
    def __init__(self, *mods):

        super().__init__()
        self.mods = nn.ModuleList([*mods])

    def forward(self, x, mask):
        for m in self.mods:
            if type(m) in [type(nl) for nl in nonlinearities.values()] + \
                    [nn.Linear] + [Dropout2dChannelsLast] + [nn.LSTM]:
                x = m(x)
            else:
                x = m(x, mask)
        return x




class SetLinear(nn.Module):

    def __init__(self,
                 n_in,
                 n_out,
                 pooling='MEAN'):
        super().__init__()

        self.pooling = pooling
        self.ffwd1 = nn.Linear(n_in, n_out)
        self.ffwd2 = nn.Linear(n_in, n_out)

    def forward(self, x, mask=None):
        # x is (N, n_samp, n_in)
        N, n_samp, n_in = x.size()

        local = self.ffwd1(x)
        glob = self.ffwd2(x)
        if mask is not None:
            mask = mask.unsqueeze(2).float()
            if self.pooling=='MEAN':
                pool = (glob * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True)
            else:
                pool = torch.max(glob+torch.log(mask), dim=1, keepdim=True)[0]
        else:
            if self.pooling=='MEAN':
                pool = glob.mean(dim=1, keepdim=True)
            else:
                pool = torch.max(glob, dim=1, keepdim=True)[0]

        return local + pool






class Dropout2dChannelsLast(_DropoutNd):

    def forward(self, x, input):

        e = torch.ones_like((input[:,:1,:]))
        return input * F.dropout(e, self.p, self.training, self.inplace)






def create_model( opt):


    nonlinearity = nonlinearities[opt.non_lin]

    if opt.type == 'Swarm':

        # uses opt. ...
        # n_layers
        # n_in
        # n_out
        # n_hidden
        # n_iter
        # dropout

        if opt.n_layers == 1:
            model = SwarmLayer( n_in     = opt.n_in,
                                n_out    = opt.n_out,
                                n_hidden = opt.n_hidden,
                                n_iter   = opt.n_iter,
                                dropout  = opt.dropout,
                                n_dim=1, pooling='MEAN', channel_first=False, cache=True)
        else:
            assert opt.n_layers>1

            layers = []
            n_out_last = opt.n_in
            for i in range(opt.n_layers-1):
                layers.append( SwarmLayer(n_in     = n_out_last,
                                          n_out    = opt.n_hidden,
                                          n_hidden = opt.n_hidden,
                                          n_iter   = opt.n_iter,
                                          dropout=opt.dropout,
                                          n_dim=1, pooling='MEAN', channel_first=False, cache=True) )
                layers.append( nonlinearity)
                n_out_last = opt.n_hidden
            layers.append( SwarmLayer(n_in     = n_out_last,
                                      n_out    = opt.n_out,
                                      n_hidden = opt.n_hidden,
                                      n_iter   = opt.n_iter,
                                      dropout=opt.dropout,
                                      n_dim=1, pooling='MEAN', channel_first=False, cache=True) )

            model = MaskedSequential(*layers)

    elif opt.type == 'SetLinear' or opt.type == 'SetLinearMax':

        # uses opt. ...
        # n_layers
        # n_in
        # n_out
        # n_hidden
        pooling = 'MEAN' if opt.type == 'SetLinear' else 'MAX'
        layers = []
        n_out_last = opt.n_in
        for i in range(opt.n_layers - 1):
            layers.append(SetLinear(n_in=n_out_last,
                                    n_out=opt.n_hidden,
                                    pooling=pooling) )
            n_out_last = opt.n_hidden
            layers.append(nonlinearity)
        layers.append(SetLinear(n_in=n_out_last,
                                n_out=opt.n_out,
                                pooling=pooling) )

        model = MaskedSequential(*layers)


    elif opt.type == 'SetTransformer':

        # uses opt. ...
        # n_layers
        # n_in
        # n_out
        # n_hidden
        # n_heads
        # n_ind_points

        d = opt.n_hidden
        h = opt.n_heads
        m = opt.n_ind_points

        layers = [nn.Linear( opt.n_in, opt.n_hidden)]
        for _ in range(opt.n_layers):
            layers.append( InducedSetAttentionBlock(d=d, m=m, h=h,first_rff=RFF(d=d),second_rff=RFF(d=d)) )
            if opt.dropout>0.0:
                layers.append(Dropout2dChannelsLast( p=opt.dropout))
        layers.append(nn.Linear(opt.n_hidden, opt.n_out))

        model = MaskedSequential(*layers)

    elif opt.type == 'LSTM' or opt.type == 'LSTMS':

        # uses opt. ...
        # n_layers
        # n_in
        # n_out
        # n_hidden

        # LSTMS is LSTM made set equivariant by sorting

        class LSTMModel(nn.Module):
            def __init__(self, opt):
                super().__init__()
                self.lstm = nn.LSTM( opt.n_in, opt.n_hidden, opt.n_layers,
                               batch_first=True, bidirectional=True)
                self.lin = nn.Linear(opt.n_hidden*2 , opt.n_out)

            def forward(self, x, mask):

                if opt.type == 'LSTMS':
                    i = torch.argsort(x[:, :, 0], dim=1)
                    ix = i.unsqueeze(2).expand_as(x)
                    x    = torch.gather(x, dim=1, index=ix)
                    #ask = torch.gather(mask, dim=1, index=i)

                tmp,_ = self.lstm(x)
                out = self.lin(tmp)

                if opt.type == 'LSTMS':
                    iout = i.unsqueeze(2).expand_as(out)
                    out = torch.zeros_like(out).scatter_(1, iout, out)

                return out

        model = LSTMModel(opt)


    else:

        raise ValueError("Unknown model type {}".format(opt.type))



    return model
