import math
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, input):
        input_coal = input.coalesce()
        drop_val = F.dropout(input_coal._values(), self.p, self.training)
        return torch.sparse.FloatTensor(input_coal._indices(), drop_val, input.shape)


class MixedDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.dense_dropout = nn.Dropout(p)
        self.sparse_dropout = SparseDropout(p)

    def forward(self, input):
        if input.is_sparse:
            return self.sparse_dropout(input)
        else:
            return self.dense_dropout(input)


class MixedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # Our fan_in is interpreted by PyTorch as fan_out (swapped dimensions)
        nn.init.kaiming_uniform_(self.weight, mode='fan_out', a=math.sqrt(5))
        if self.bias is not None:
            _, fan_out = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_out)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if self.bias is None:
            if input.is_sparse:
                res = torch.sparse.mm(input, self.weight)
            else:
                res = input.matmul(self.weight)
        else:
            if input.is_sparse:
                res = torch.sparse.addmm(self.bias.expand(input.shape[0], -1), input, self.weight)
            else:
                res = torch.addmm(self.bias, input, self.weight)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
                self.in_features, self.out_features, self.bias is not None)


def sparse_matrix_to_torch(X):
    coo = X.tocoo()
    indices = np.array([coo.row, coo.col])
    return torch.sparse.FloatTensor(
            torch.LongTensor(indices),
            torch.FloatTensor(coo.data),
            coo.shape)


def sparse_mat_to_padded_tensors(sp_matrix):
    nnz_per_row = torch.LongTensor(sp_matrix.indptr[1:] - sp_matrix.indptr[:-1])
    padded_inds = np.zeros([sp_matrix.shape[0], max(nnz_per_row)], dtype=np.int64)
    padded_vals = np.zeros([sp_matrix.shape[0], max(nnz_per_row)], dtype=np.float32)
    for i in range(sp_matrix.shape[0]):
        padded_inds[i, :nnz_per_row[i]] = sp_matrix.indices[sp_matrix.indptr[i]:sp_matrix.indptr[i + 1]]
        padded_vals[i, :nnz_per_row[i]] = sp_matrix[i].data
    return torch.FloatTensor(padded_vals), torch.LongTensor(padded_inds), nnz_per_row


def matrix_to_torch(X):
    if sp.issparse(X):
        return sparse_matrix_to_torch(X)
    else:
        return torch.FloatTensor(X)


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def logcosh(x):
    return x + F.softplus(-2. * x) - math.log(2.)


def trace_batch(mat):
    return torch.diagonal(mat, dim1=-2, dim2=-1).sum(-1)


def max_finite(tensor, dim=None):
    mask = torch.isinf(tensor)
    tensor_neginf = torch.masked_fill(tensor, mask, -math.inf)
    if dim is None:
        return tensor_neginf.max()
    else:
        return tensor_neginf.max(dim=dim)


@torch.jit.script
def sum_finite(tensor: torch.Tensor, dim: int):
    mask = ~torch.isfinite(tensor)
    tensor_finite = torch.masked_fill(tensor, mask, 0)
    return torch.sum(tensor_finite, dim)


def sinkhorn_normalization(mat: torch.FloatTensor, niter: int, mean_one: bool = False, mask: torch.BoolTensor = None):
    batch_size, nrows, ncols = mat.shape
    assert nrows == ncols

    if mean_one:
        target_sum = nrows
    else:
        target_sum = 1

    # Mask for padded tensors
    if mask is not None:
        mat_u = mat.clone().transpose(1, 2)
        mat_u[:, :, 0] += mask

        mat_v = mat.clone()
        mat_v[:, :, 0] += mask

    u = mat.new_ones([batch_size, nrows])
    for _ in range(niter):
        v = torch.clamp(target_sum / torch.einsum("bij, bj -> bi", mat_u, u), max=1e10)
        u = torch.clamp(target_sum / torch.einsum("bij, bj -> bi", mat_v, v), max=1e10)

    return u, v


def logsinkhorn_normalization(mat: torch.FloatTensor, niter: int, mean_one: bool):
    batch_size, nrows, ncols = mat.shape
    assert nrows == ncols

    if mean_one:
        target_sum = math.log(nrows)
    else:
        target_sum = 0

    def M(u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = -c_{ij} + u_i + v_j$"
        # clamp to prevent NaN for inf - inf
        if u is None:
            return v[:, None, :] + mat
        elif v is None:
            return u[:, :, None] + mat
        else:
            return u[:, :, None] + v[:, None, :] + mat

    u = mat.new_zeros(batch_size, nrows)
    v = mat.new_zeros(batch_size, nrows)
    for _ in range(niter):
        u = target_sum - torch.logsumexp(M(None, v), dim=-1)
        v = target_sum - torch.logsumexp(M(u, None), dim=-2)

    return u, v
