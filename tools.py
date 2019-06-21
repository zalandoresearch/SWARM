import torch
import torch.nn.functional as F
import numpy as np
import math





def causal_2D_pooling( n_in, H,W, h,w, device=None):

    kernel = np.tile(np.eye(n_in, dtype=np.float32)[:, :, np.newaxis, np.newaxis], (1, 1, h, w))
    kernel[:, :, h - 1, w // 2 + 1:] = 0

    padding = (h-1, w//2)

    kernel = torch.tensor(kernel, device=device)
    scale = F.conv2d(torch.ones( (1, n_in, H, W), device=device).float(),
                     kernel, padding=padding) [:,:,:H,:W]

    return kernel, padding, scale

def sort_by_slice(x, y):

    i = torch.argsort(x[:,0,:], dim=1)
    ix = i.unsqueeze(1).expand_as(x)
    x = torch.gather(x,dim=2,index=ix)
    iy = i
    y = torch.gather(y,dim=1,index=iy)

    return x,y


def create_location_features_1d(W, n_emb, device=None):


    assert n_emb%2 == 0   # must be a multiple of 2

    lW = torch.arange(W, device=device).float()/W

    features = []

    for i in np.logspace(0,math.log10(0.5*W),n_emb//2):
        features.append(torch.sin(lW*i*2*math.pi))
        features.append(torch.cos(lW*i*2*math.pi))

    features =  torch.stack(tuple(features), 0) # (n_emb,W)

    return features


def create_location_features_2d(H, W, n_emb, device=None):

    assert n_emb%4 == 0   # must be a multiple of 4

    lW = torch.arange(W, device=device).view( (1,W)).float()/W
    lH = torch.arange(H, device=device).view( (H,1)).float()/W

    features = []

    for i in np.logspace(0,math.log10(0.5*max(W,H)),n_emb//4):
        features.append(torch.sin(lW*i*2*math.pi).expand( H, W))
        features.append(torch.sin(lH*i*2*math.pi).expand( H, W))
        features.append(torch.cos(lW*i*2*math.pi).expand( H, W))
        features.append(torch.cos(lH*i*2*math.pi).expand( H, W))

    features =  torch.stack(tuple(features), 0) #(Nemb,H,W)

    return features

from itertools import chain, combinations
def factors(n):
    def prime_factors(n):
        i = 2
        pf = []
        while i * i <= n:
            if n % i:
                i += 1
            else:
                n //= i
                pf.append(i)
        if n > 1:
            pf.append(n)
        return pf
    pf = [1] + prime_factors(n)
    return np.unique([np.prod(l) for l in chain(*[list(combinations(pf,i)) for i in range(1,len(pf))])])
