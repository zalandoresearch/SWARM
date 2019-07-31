import math

import numpy as np
import torch
from scipy.stats import invwishart, multivariate_normal, multinomial
from scipy.optimize import linear_sum_assignment


def create_dataset(n_dim, n_clust, n_tasks, n_entities, seed=None):
    """
    Create the amortised clustering dataset
    :param n_dim: number of dimensions
    :param n_clust: pair (lo,hi) number of clusters uniformly in the range(lo,hi)
    :param n_tasks: number of tasks
    :param n_entities: pair (lo,hi) number of entities uniformly in the range(lo,hi)
    :param seed: random seed
    :return: data set
    """
    if seed is not None:
        np.random.seed(seed)

    tasks = []
    for i in range(n_tasks):

        n_clust_ = np.random.randint(*n_clust)
        Si = np.zeros((n_clust_, n_dim, n_dim))
        mu = np.zeros((n_clust_, n_dim))
        x = []
        idx = []

        n_ent = np.random.randint(*n_entities)

        for j, n in enumerate(*multinomial.rvs(n_ent, np.ones(n_clust_) / n_clust_, 1)):
            Si[j] = invwishart.rvs(4, 0.05 * np.eye(n_dim))
            mu[j] = np.random.randn(n_dim)
            x.append(multivariate_normal.rvs(mu[j], Si[j], size=n).astype(np.float32))
            idx.append(j * np.ones(n, dtype=np.long))

        j = np.random.permutation(n_ent)
        x = np.concatenate(x, 0)[j]
        idx = np.concatenate(idx, 0)[j]

        tasks.append((x, idx, mu, Si))

    return tasks


def collate_fn(batch, device=None):
    """
    Concatenate list of tasks to a valid mini batch. Since tasks can have different numbers of entities, they are zero
    padded to the largest common length. A mask array indicated the padding (mask==0) vs. valid entities (mask==1)
    :param batch: list of tasks
    :param device: pytorch device
    :return: entity coordinates, cluster ID's, mask array
    """
    n_ent = max([x.shape[0] for x, _, _, _ in batch])

    def subsamp(x, idx, mu, Si, n_ent):
        n_pad = n_ent - x.shape[0]
        if n_pad > 0:
            x = np.pad(x, ((0, n_pad), (0, 0)), mode='constant', constant_values=0)
            idx = np.pad(idx, (0, n_pad), mode='constant', constant_values=0)
        mask = np.array([1] * (n_ent - n_pad) + [0] * n_pad, dtype=np.uint8)
        return x, idx, mask

    x, idx, mask = zip(*[subsamp(*task, n_ent) for task in batch])

    x = torch.tensor(np.stack(x, 0), device=device)
    idx = torch.tensor(np.stack(idx, 0), device=device)
    mask = torch.tensor(np.stack(mask, 0), device=device)

    return x, idx, mask


import random


class BalancedSampler(object):
    """
    A Sampler object that produces mini batches of tasks of similar length. That way the padding required to equalize the
    length of all tasks in the minibatch is minimized. The mini batches themselves can be sampled in random order.
    """
    def __init__(self, dataset, sort_fn, batch_size, drop_last, shuffle):

        self.shuffle = shuffle
        self.idx = list(range(len(dataset)))
        self.idx.sort(key=lambda i: sort_fn(dataset[i]))

        def chunks(l, n):
            """Yield successive n-sized chunks from l."""

            N = len(l) - n + 1 if drop_last else len(l)
            for i in range(0, N, n):
                yield l[i:i + n]

        self.batches = list(chunks(self.idx, batch_size))
        if self.shuffle:
            random.shuffle(self.batches)

    def __iter__(self):
        for b in self.batches:
            yield b
        if self.shuffle:
            random.shuffle(self.batches)

    def __len__(self):
        return len(self.batches)


def to_one_hot(x, n_classes):
    x_1h = torch.zeros(x.size() + (n_classes,), device=x.device)

    x_1h.scatter_(x.dim(), x.unsqueeze(-1), 1)
    return x_1h


def from_one_hot(x, n_classes):
    return torch.sum(1 - x.cumsum(dim=-1), dim=-1)




def greedy_cross_entropy(logits, labels, mask, C):
    """
    compute cross entropy under permutation ambiguity under a greedy matching approach
    N: batch size, E: number of entities, C: number of classes
    :param logits: (N,E,C) tensor of unnormalized logits
    :param labels: (N,E,C) tensor of 1-hot encoded class labels
    :param mask: (N,E) {0,1}-tensor masking not present entities
    :param C: number of classes
    :return: greedy cross entropy loss, pivot matching elements
    """
    # logits (N,E,C)

    m_max = 0
    idx = []
    logits = logits *mask.float().unsqueeze(-1)
    M = torch.matmul(labels.transpose(1, 2), logits) / torch.sum(mask.float(), 1).view(-1, 1, 1)
    N, _, _ = M.size()

    for _ in range(C):
        # M is (N,C,C)
        m_max_, ij = torch.max(M.view(N, -1), 1)
        m_max += m_max_
        i = ij // C
        j = ij % C

        idx.append(torch.stack([i, j], 1))

        M = M.scatter(1, i.view(-1, 1, 1).expand(N, 1, C), -math.inf)
        M = M.scatter(2, j.view(-1, 1, 1).expand(N, C, 1), -math.inf)

    idx = torch.stack(idx, 1)
    return m_max, idx




def hungarian_cross_entropy(logits, labels, mask, C):
    # logits (N,E,C)

    m_max = []
    idx = []
    logits = logits *mask.float().unsqueeze(-1)
    M = torch.matmul(labels.transpose(1, 2), logits) / torch.sum(mask.float(), 1).view(-1, 1, 1)

    for m in M:
        i, j = linear_sum_assignment(m.data.cpu().numpy())
        m_max.append(m[i,j].sum())

        idx.append( torch.tensor([i,j]))

    m_max = torch.stack(m_max,0)
    idx = torch.stack(idx, 0)
    return m_max, idx


greedy_cross_entropy = hungarian_cross_entropy



from torch.utils.data import DataLoader


def create_data_loders(n_tasks, seed=0, device=None):

    ds = create_dataset( n_dim=2, n_clust=(3,11), n_tasks=n_tasks, n_entities=(100,1001), seed=seed)

    split = (n_tasks*9)//10
    ds_train = ds[:split]
    ds_val = ds[split:]

    bs_train = BalancedSampler(ds_train, lambda b: b[0].shape[0], batch_size=100, drop_last=False, shuffle=True)
    bs_val   = BalancedSampler(ds_val,   lambda b: b[0].shape[0], batch_size=100, drop_last=False, shuffle=True)

    dl_train = DataLoader( ds_train, collate_fn = lambda batch: collate_fn(batch,device), batch_sampler=bs_train)
    dl_val   = DataLoader( ds_val,   collate_fn = lambda batch: collate_fn(batch,device), batch_sampler=bs_val)


    return dl_train, dl_val
