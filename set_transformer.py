import torch
import torch.nn as nn

class Attention(nn.Module):
    """Scaled Dot-Product Attention."""

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)

    def forward(self, queries, keys, values, mask_query=None, mask_key=None):
        """
        It is equivariant to permutations
        of the batch dimension (`b`).

        It is equivariant to permutations of the
        second dimension of the queries (`n`).

        It is invariant to permutations of the
        second dimension of keys and values (`m`).

        Arguments:
            queries: a float tensor with shape [b, n, d].
            keys: a float tensor with shape [b, m, d].
            values: a float tensor with shape [b, m, d'].
        Returns:
            a float tensor with shape [b, n, d'].
        """



        attention = torch.bmm(queries, keys.transpose(1, 2))

        if mask_query is not None:
            #print("query",attention.size(),mask_query.size())
            attention = attention * mask_query.unsqueeze(2).float()

        if mask_key is not None:
            #print("key",attention.size(),mask_key.size())
            attention = attention + torch.log(mask_key.unsqueeze(1).float())


        attention = self.softmax(attention / self.temperature)
        # it has shape [b, n, m]

        return torch.bmm(attention, values)



class MultiheadAttention(nn.Module):

    def __init__(self, d, h):
        """
        Arguments:
            d: an integer, dimension of queries and values.
                It is assumed that input and
                output dimensions are the same.
            h: an integer, number of heads.
        """
        super().__init__()

        assert d % h == 0
        self.d = d
        self.h = h

        # everything is projected to this dimension
        p = d // h

        self.project_queries = nn.Linear(d, d)
        self.project_keys = nn.Linear(d, d)
        self.project_values = nn.Linear(d, d)
        self.concatenation = nn.Linear(d, d)
        self.attention = Attention(temperature=p**0.5)

        def weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

        self.apply(weights_init)

    def forward(self, queries, keys, values, mask_query=None, mask_key=None):
        """
        Arguments:
            queries: a float tensor with shape [b, n, d].
            keys: a float tensor with shape [b, m, d].
            values: a float tensor with shape [b, m, d].
        Returns:
            a float tensor with shape [b, n, d].
        """

        h = self.h
        b, n, d = queries.size()
        _, m, _ = keys.size()
        p = d // h

        if mask_query is not None:
            queries = queries * mask_query.unsqueeze(2).float()

        if mask_key is not None:
            keys = keys * mask_key.unsqueeze(2).float()
            values = values * mask_key.unsqueeze(2).float()

        queries = self.project_queries(queries)  # shape [b, n, d]
        keys = self.project_keys(keys)  # shape [b, m, d]
        values = self.project_values(values)  # shape [b, m, d]

        if mask_query is not None:
            queries = queries * mask_query.unsqueeze(2).float()

        if mask_key is not None:
            keys = keys * mask_key.unsqueeze(2).float()
            values = values * mask_key.unsqueeze(2).float()


        queries = queries.view(b, n, h, p)
        keys = keys.view(b, m, h, p)
        values = values.view(b, m, h, p)

        queries = queries.permute(2, 0, 1, 3).contiguous().view(h*b, n, p)
        keys = keys.permute(2, 0, 1, 3).contiguous().view(h*b, m, p)
        values = values.permute(2, 0, 1, 3).contiguous().view(h*b, m, p)

        if mask_query is not None:
            mask_query_ = mask_query.repeat([h,1])
        else:
            mask_query_ = None

        if mask_key is not None:
            mask_key_ = mask_key.repeat([h,1])
        else:
            mask_key_ = None

        output = self.attention(queries, keys, values, mask_query_, mask_key_)  # shape [h*b, n, p]
        output = output.view(h, b, n, p)
        output = output.permute(1, 2, 0, 3).contiguous().view(b, n, d)
        output = self.concatenation(output)  # shape [b, n, d]

        if mask_query is not None:
            output = output * mask_query.unsqueeze(2).float()


        return output



class MultiheadAttentionBlock(nn.Module):

    def __init__(self, d, h, rff):
        """
        Arguments:
            d: an integer, input dimension.
            h: an integer, number of heads.
            rff: a module, row-wise feedforward layer.
                It takes a float tensor with shape [b, n, d] and
                returns a float tensor with the same shape.
        """
        super().__init__()

        self.multihead = MultiheadAttention(d, h)
        self.layer_norm1 = nn.LayerNorm(d)
        self.layer_norm2 = nn.LayerNorm(d)
        self.rff = rff

    def forward(self, x, y, mask_x=None, mask_y=None):
        """
        It is equivariant to permutations of the
        second dimension of tensor x (`n`).

        It is invariant to permutations of the
        second dimension of tensor y (`m`).

        Arguments:
            x: float tensors with shape [b, n, d].
            y: float tensors with shape [b, m, d].
        Returns:
            a float tensor with shape [b, n, d].
        """
        h = self.layer_norm1(x + self.multihead(x, y, y, mask_query=mask_x, mask_key=mask_y))
        return self.layer_norm2(h + self.rff(h))


class PoolingMultiheadAttention(nn.Module):

    def __init__(self, d, k, h, rff):
        super().__init__()
        self.mab = MultiheadAttentionBlock(d, h, rff)
        self.seed_vectors = nn.Parameter(torch.randn(k, d))

    def forward(self, z):
        """
        Arguments:
            z: a float tensor with shape [b, n, d].
        Returns:
            a float tensor with shape [b, k, d].
        """
        b = z.size(0)
        s = self.seed_vectors.unsqueeze(0).repeat([b, 1, 1])  # shape [b, k, d]
        return self.mab(s, z)
        # note that in the original paper
        # they return mab(s, rff(z))


class InducedSetAttentionBlock(nn.Module):

    def __init__(self, d, m, h, first_rff, second_rff):
        super().__init__()
        self.mab1 = MultiheadAttentionBlock(d, h, first_rff)
        self.mab2 = MultiheadAttentionBlock(d, h, second_rff)
        self.inducing_points = nn.Parameter(torch.randn(m, d))

    def forward(self, x, mask=None):
        """
        Arguments:
            x: a float tensor with shape [b, n, d].
        Returns:
            a float tensor with shape [b, n, d].
        """
        b = x.size()[0]
        i = self.inducing_points.unsqueeze(0).repeat([b, 1, 1])  # shape [b, m, d]
        h = self.mab1(i, x, mask_y=mask)  # shape [b, m, d]
        return self.mab2(x, h, mask_x=mask)

class RFF(nn.Module):

    def __init__(self, d):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(),
            nn.Linear(d, d), nn.ReLU(),
            nn.Linear(d, d), nn.ReLU(),
            nn.Linear(d, d), nn.ReLU()
        )

        def weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

        self.apply(weights_init)

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, n, d].
        Returns:
            a float tensor with shape [b, n, d].
        """
        return self.layers(x)

