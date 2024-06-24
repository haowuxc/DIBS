import numpy as np
import torch
import math

from itertools import product
from torch import log, exp
import torch.nn.functional as F


device = "cuda" if torch.cuda.is_available() else "cpu"


def compute_all_costs(
    z_features,
    x_features,
    gamma_xz,
    drop_cost_type,
    keep_percentile,
    l2_normalize=False,
    given_baseline_logits=None,
    return_baseline=False,
):
    """This function computes pairwise match and individual drop costs used in Drop-DTW

    Parameters
    __________

    sample: dict
        sample dictionary
    distractor: torch.tensor of size [d] or None
        Background class prototype. Only used if the drop cost is learnable.
    drop_cost_type: str
        The type of drop cost definition, i.g., learnable or logits percentile.
    keep_percentile: float in [0, 1]
        if drop_cost_type == 'logit', defines drop (keep) cost threshold as logits percentile
    l2_normalize: bool
        wheather to normalize clip and step features before computing the costs
    """

    if l2_normalize:
        x_features = F.normalize(x_features, p=2, dim=1)
        z_features = F.normalize(z_features, p=2, dim=1)

    sim = z_features @ x_features.T

    if drop_cost_type == "logit":
        if keep_percentile > 1:
            baseline_logit = sim.min().detach() - 1
        else:
            k = max([1, int(torch.numel(sim) * keep_percentile)])
            baseline_logit = torch.topk(sim.reshape([-1]), k).values[-1].detach()
        baseline_logits = baseline_logit.repeat([1, sim.shape[1]])  # making it of shape [1, N]
        sims_ext = torch.cat([sim, baseline_logits], dim=0)
    else:
        assert False, f"No such drop mode {drop_cost_type}"

    softmax_sims = torch.nn.functional.softmax(sims_ext / gamma_xz, dim=0)
    matching_probs, drop_probs = softmax_sims[:-1], softmax_sims[-1]
    zx_costs = -torch.log(matching_probs + 1e-5)
    drop_costs = -torch.log(drop_probs + 1e-5)
    return zx_costs, drop_costs, drop_probs


def compute_double_costs(
    z_features,
    x_features,
    gamma_xz,
    drop_cost_type,
    keep_percentile,
    l2_normalize=False,
    return_baseline=False,
):
    """This function computes pairwise match and individual drop costs used in Drop-DTW

    Parameters
    __________

    sample: dict
        sample dictionary
    distractor: torch.tensor of size [d] or None
        Background class prototype. Only used if the drop cost is learnable.
    drop_cost_type: str
        The type of drop cost definition, i.g., learnable or logits percentile.
    keep_percentile: float in [0, 1]
        if drop_cost_type == 'logit', defines drop (keep) cost threshold as logits percentile
    l2_normalize: bool
        wheather to normalize clip and step features before computing the costs
    """

    z_features, frame_features = z_features, x_features
    if l2_normalize:
        x_features = F.normalize(frame_features, p=2, dim=1)
        z_features = F.normalize(z_features, p=2, dim=1)
    sim = z_features @ x_features.T

    if drop_cost_type == "logit":
        k = max([1, int(torch.numel(sim) * keep_percentile)])
        baseline_logit = torch.topk(sim.reshape([-1]), k).values[-1].detach()
    else:
        assert False, f"No such drop mode {drop_cost_type}"
    sim_ext = F.pad(sim, (0, 1, 0, 1), value=baseline_logit)

    softmax_sims = torch.nn.functional.softmax(sim_ext.reshape(-1) / gamma_xz, dim=0).reshape(sim_ext.shape)
    matching_probs, x_drop_probs, z_drop_probs = softmax_sims[:-1, :-1], softmax_sims[-1, :-1], softmax_sims[:-1, -1]
    zx_costs = -torch.log(matching_probs + 1e-5)
    x_drop_costs = -torch.log(x_drop_probs + 1e-5)
    z_drop_costs = -torch.log(z_drop_probs + 1e-5)
    return zx_costs, x_drop_costs, z_drop_costs


class VarTable:
    def __init__(self, dims, dtype=torch.float, device=device):
        self.dims = dims
        d1, d2, d_rest = dims[0], dims[1], dims[2:]

        self.vars = []
        for i in range(d1):
            self.vars.append([])
            for j in range(d2):
                var = torch.zeros(d_rest).to(dtype).to(device)
                self.vars[i].append(var)

    def __getitem__(self, pos):
        i, j = pos
        return self.vars[i][j]

    def __setitem__(self, pos, new_val):
        i, j = pos
        if self.vars[i][j].sum() != 0:
            assert False, "This cell has already been assigned. There must be a bug somwhere."
        else:
            self.vars[i][j] = self.vars[i][j] + new_val

    def show(self):
        device, dtype = self[0, 0].device, self[0, 0].dtype
        mat = torch.zeros((self.d1, self.d2, self.d3)).to().to(dtype).to(device)
        for dims in product([range(d) for d in self.dims]):
            i, j, rest = dims[0], dims[1], dims[2:]
            mat[dims] = self[i, j][rest]
        return mat


def minGamma(inputs, gamma=1, keepdim=True):
    """continuous relaxation of min defined in the D3TW paper"""
    if type(inputs) == list:
        if inputs[0].shape[0] == 1:
            inputs = torch.cat(inputs)
        else:
            inputs = torch.stack(inputs, dim=0)

    if gamma == 0:
        minG = inputs.min(dim=0, keepdim=keepdim)
    else:
        # log-sum-exp stabilization trick
        zi = -inputs / gamma
        max_zi = zi.max()
        log_sum_G = max_zi + log(exp(zi - max_zi).sum(dim=0, keepdim=keepdim) + 1e-5)
        minG = -gamma * log_sum_G
    return minG


def minProb(inputs, gamma=1, keepdim=True):
    if type(inputs) == list:
        if inputs[0].shape[0] == 1:
            inputs = torch.cat(inputs)
        else:
            inputs = torch.stack(inputs, dim=0)

    if gamma == 0:
        minP = inputs.min(dim=0, keepdim=keepdim)
    else:
        probs = F.softmax(-inputs / gamma, dim=0)
        minP = (probs * inputs).sum(dim=0, keepdim=keepdim)
    return minP


def prob_min(values, gamma_min, logits=None):
    logits = values if logits is None else logits
    assert len(logits) == len(values), "Values and prob logits are of different length"

    if len(values) > 1:
        values = torch.cat(values, dim=-1)
        logits = torch.cat(logits, dim=-1)
    else:
        values = values[0]
        logits = logits[0]

    if gamma_min > 0:
        probs = F.softmax(-logits / gamma_min, dim=-1)
    else:
        probs = F.one_hot(logits.argmin(), logits.size(-1))

    if values.dim() > probs.dim():
        probs = probs[..., None, :]

    out = (values * probs).sum(-1).to(values.dtype)
    return out


def list_min(values, keys=None):
    keys = values if keys is None else keys
    assert len(keys) == len(values), "Values and prob logits are of different length"

    if values[0].dim() == keys[0].dim() + 1:
        dim = -2
    else:
        dim = -1

    if len(values) > 1:
        values = torch.cat(values, dim=dim)
        keys = torch.cat(keys, dim=-1)
    else:
        values = values[0]
        keys = keys[0]

    onehot = F.one_hot(keys.argmin(-1), keys.size(-1))
    if values.dim() > keys.dim():
        onehot = onehot[..., None]
    out = (values * onehot).sum(dim).to(values.dtype)
    return out


def traceback(D):
    i, j = np.array(D.shape) - 2
    p, q = [i], [j]
    while (i > 0) or (j > 0):
        tb = np.argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:  # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return np.array(p), np.array(q)


def diag_to_mat(diags, K, N):
    mat = np.zeros([K, N]) - 123
    for d in range(len(diags)):
        for r, v in enumerate(diags[d]):
            j = min(d, N - 1) - r
            i = d - j
            mat[i, j] = v if v < 1e8 else np.inf
    return mat


def pad_costs(zx_costs_list, drop_costs_list):
    B = len(zx_costs_list)
    Ns, Ks = [], []
    for i in range(B):
        Ki, Ni = zx_costs_list[i].shape
        if Ki >= Ni:
            # in case the number of steps is greater than the number of frames,
            # duplicate every frame and let the drops do the job.
            mult = math.ceil(Ki / Ni)
            zx_costs_list[i] = torch.stack([zx_costs_list[i]] * mult, dim=-1).reshape([Ki, -1])
            drop_costs_list[i] = torch.stack([drop_costs_list[i]] * mult, dim=-1).reshape([-1])
            Ni *= mult
        Ns.append(Ni)
        Ks.append(Ki)
    N, K = max(Ns), max(Ks)

    # preparing padded tables
    padded_cum_drop_costs, padded_drop_costs, padded_zx_costs = [], [], []
    for i in range(B):
        zx_costs = zx_costs_list[i]
        drop_costs = drop_costs_list[i]
        cum_drop_costs = torch.cumsum(drop_costs, dim=0)

        # padding everything to the size of the largest N and K
        row_pad = torch.zeros([N - Ns[i]]).to(zx_costs.device)
        padded_cum_drop_costs.append(torch.cat([cum_drop_costs, row_pad]))
        padded_drop_costs.append(torch.cat([drop_costs, row_pad]))
        multirow_pad = torch.stack([row_pad + 9999999999] * Ks[i], dim=0)
        padded_table = torch.cat([zx_costs, multirow_pad], dim=1)
        rest_pad = torch.zeros([K - Ks[i], N]).to(zx_costs.device) + 9999999999
        padded_table = torch.cat([padded_table, rest_pad], dim=0)
        padded_zx_costs.append(padded_table)
    return padded_cum_drop_costs, padded_drop_costs, padded_zx_costs, Ns, Ks


def get_diag_coord_grid(B, d_len, num_states, d_idx):
    """
    B - batch size
    d - num_elements in the diagonal
    num_states - number of states in DP table
    d_idx - idx of the diagonal , used for marking
    """
    r = torch.arange(d_len)
    s = torch.arange(num_states)
    d = torch.ones(d_len, num_states) * d_idx
    mg = torch.stack([d, *torch.meshgrid(r, s)], dim=-1)[None, ...].repeat([B, 1, 1, 1])
    return mg


def diag_traceback(pointer, N, paths):
    # getting rid of unnecessary elements in the batch
    pointer = [int(l.item()) for l in pointer]
    d, r, s = pointer
    traceback = [pointer]
    while d > 0:
        new_pointer = [int(l.item()) for l in paths[d][r, s]]
        traceback.append(new_pointer)
        d, r, s = new_pointer

    # transform to rectangular coordinates
    rectangular_traceback = []
    for d, r, s in traceback:
        i = r + max(0, d - N + 1)
        j = d - i
        if i > 0 and j > 0:
            rectangular_traceback.append((i, j, s))

    return traceback, rectangular_traceback


def nw_diag_traceback(d, r, N, paths):
    d, r = int(d.item()), int(r.item())
    traceback = []
    while d > 0:
        d_1, s_1, s = [int(l.item()) for l in paths[d][r, 0]]
        traceback.append((d, r, s))
        d, r = d_1, s_1

    # transform to rectangular coordinates
    rectangular_traceback = []
    for d, r, s in traceback:
        i = r + max(0, d - N + 1)
        j = d - i
        if i > 0 and j > 0:
            rectangular_traceback.append((i, j, s))

    return traceback, rectangular_traceback


def compute_symmetric_cost(sim, keep_percentile=0.3):
    k = max([1, int(torch.numel(sim) * keep_percentile)])
    baseline_logit = torch.topk(sim.reshape([-1]), k).values[-1].detach()
    baseline_logits = baseline_logit.repeat([1, sim.shape[1]])  # making it of shape [1, N]
    zx_costs = -sim
    x_drop_costs = -baseline_logits.squeeze()
    z_drop_costs = -baseline_logit.repeat([1, sim.shape[0]]).squeeze()
    return zx_costs, x_drop_costs, z_drop_costs


#============ Hach from model_utilis.py in StepFormer ============#



def unique_softmax(sim, labels, gamma=1, dim=0):
    assert sim.shape[0] == labels.shape[0]
    labels = labels.detach().cpu().numpy()
    _, unique_index, unique_inverse_index = np.unique(labels, return_index=True, return_inverse=True)
    unique_sim = sim[unique_index]
    unique_softmax_sim = torch.nn.functional.softmax(unique_sim / gamma, dim=dim)
    softmax_sim = unique_softmax_sim[unique_inverse_index]
    return softmax_sim

def compute_masked_sims(z, x, z_pad_mask, x_pad_mask, l2_normalize=False, softmax_dim=None, gamma=None):
    # z ~ [B, K, d], x ~ [B, N, d]
    if l2_normalize:
        z, x = F.normalize(z, dim=-1), F.normalize(x, dim=-1)
    pad_sims = torch.einsum("bkd,bnd->bkn", z, x)
    masked_sims = []
    for i in range(x.shape[0]):
        masked_sim = pad_sims[i]
        masked_sim = masked_sim if z_pad_mask is None else masked_sim[~z_pad_mask[i], :]
        masked_sim = masked_sim if x_pad_mask is None else masked_sim[:, ~x_pad_mask[i]]
        if softmax_dim is not None:
            masked_sim = F.softmax(masked_sim / gamma, dim=softmax_dim)
        masked_sims.append(masked_sim)
    return masked_sims  

def compute_sim(z, x, l2_norm):
    if l2_norm:
        return F.normalize(z, dim=1) @ F.normalize(x, dim=1).T
    else:
        return z @ x.T


def cosine_sim(x, z):
    cos_sim_fn = torch.nn.CosineSimilarity(dim=1)
    return cos_sim_fn(x[..., None], z.T[None, ...])


def cos_dist(x, z):
    cos_sim_fn = torch.nn.CosineSimilarity(dim=1)
    return (1 - cos_sim_fn(x[..., None], z.T[None, ...])) / 2


def l2_dist(x, z):
    dist_squared = (x**2).sum() + (z**2).sum() - 2 * x @ z.T
    return torch.clamp(dist_squared, min=0).sqrt()


def cos_loglikelihood(x, z, gamma=0.1, z_dim=1):
    cos_sim = cosine_sim(x, z)
    probs = F.softmax(cos_sim / gamma, dim=z_dim)
    return torch.log(probs)