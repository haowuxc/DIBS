import numpy as np
import torch
import math
from torch import log, exp
import torch.nn.functional as F
from copy import copy

from pdvc.dp.dp_utils import VarTable, minGamma, minProb, pad_costs, prob_min, unique_softmax, cosine_sim


device = "cuda" if torch.cuda.is_available() else "cpu"


def softDTW(
    step_features,
    frame_features,
    labels,
    dist_type="inner",
    softning="prob",
    gamma_min=0.1,
    gamma_xz=0.1,
    step_normalize=True,
):
    """function to obtain a soft (differentiable) version of DTW
    embs1, embs2: embedding of size N*D and M*D (N and M : number of video frames
    and D: dimensionality of of the embedding vector)
    """
    # defining the function
    _min_fn = minProb if softning == "prob" else minGamma
    min_fn = lambda x: _min_fn(x, gamma=gamma_min)

    # first get a pairwise distance matrix
    if dist_type == "inner":
        dist = step_features @ frame_features.T
    else:
        dist = cosine_sim(step_features, frame_features)
    if step_normalize:
        if labels is not None:
            norm_dist = unique_softmax(dist, labels, gamma_xz)
        else:
            norm_dist = torch.softmax(dist / gamma_xz, 0)
        dist = -log(norm_dist)

    # initialize soft-DTW table
    nrows, ncols = dist.shape
    # sdtw = torch.zeros((nrows+1,ncols+1)).to(torch.float).to(device)
    sdtw = VarTable((nrows + 1, ncols + 1))
    for i in range(1, nrows + 1):
        sdtw[i, 0] = 9999999999
    for j in range(1, ncols + 1):
        sdtw[0, j] = 9999999999

    # obtain dtw table using min_gamma or softMin relaxation
    for i in range(1, nrows + 1):
        for j in range(1, ncols + 1):
            neighbors = torch.stack([sdtw[i, j - 1], sdtw[i - 1, j - 1], sdtw[i - 1, j]])
            di, dj = i - 1, j - 1  # in the distance matrix indices are shifted by one
            new_val = dist[di, dj] + min_fn(neighbors)
            sdtw[i, j] = torch.squeeze(new_val, 0)
    sdtw_loss = sdtw[nrows, ncols] / step_features.shape[0]
    return sdtw_loss, sdtw, dist


def dropDTW(zx_costs, drop_costs, softning="prob", exclusive=True, contiguous=True, gamma_min=1):
    """function to obtain a soft (differentiable version of DTW)
    embs1, embs2: embedding of size N*D and M*D (N and M : number of video frames
    and D: dimensionality of of the embedding vector)
    """
    # defining the min function
    min_fn = minProb if softning == "prob" else minGamma
    inf = 9999999999
    K, N = zx_costs.shape
    exclusive = exclusive if K <= N else False
    cum_drop_costs = torch.cumsum(drop_costs, dim=0)

    # Creating and initializing DP tables
    D = VarTable((K + 1, N + 1, 3))  # This corresponds to B 3-dim DP tables
    for zi in range(1, K + 1):
        D[zi, 0] = torch.zeros_like(D[zi, 0]) + inf
    for xi in range(1, N + 1):
        D[0, xi] = torch.zeros_like(D[0, xi]) + cum_drop_costs[xi - 1]

    # obtain dtw table using min_gamma or softMin relaxation
    for zi in range(1, K + 1):
        for xi in range(1, N + 1):
            z_cost_ind, x_cost_ind = zi - 1, xi - 1  # indexind in costs is shifted by 1

            d_diag, d_left = D[zi - 1, xi - 1][0:1], D[zi, xi - 1][0:1]
            dp_left, dp_up = D[zi, xi - 1][2:3], D[zi - 1, xi][2:3]

            # positive transition, i.e. matching x_i to z_j
            if contiguous:
                pos_neighbors = [d_diag, dp_left]
            else:
                pos_neighbors = [d_diag, d_left]
            if not exclusive:
                pos_neighbors.append(dp_up)

            Dp = min_fn(pos_neighbors, gamma=gamma_min) + zx_costs[z_cost_ind, x_cost_ind]

            # negative transition, i.e. dropping xi
            Dm = d_left + drop_costs[x_cost_ind]

            # update final solution matrix
            D_final = min_fn([Dm, Dp], gamma=gamma_min)
            D[zi, xi] = torch.cat([D_final, Dm, Dp], dim=0)

    # Computing the final min cost for the whole batch
    min_cost = D[K, N][0]
    return min_cost, D


def batch_dropDTW(
    zx_costs_list, drop_costs_list, softning="prob", exclusive=True, contiguous=True, drop_mode="DropDTW", gamma_min=1
):
    """function to obtain a soft (differentiable version of DTW)
    embs1, embs2: embedding of size N*D and M*D (N and M : number of video frames
    and D: dimensionality of of the embedding vector)
    """
    # defining the min function
    min_fn = minProb if softning == "prob" else minGamma
    inf = 9999999999

    # pre-processing
    B = len(zx_costs_list)
    padded_cum_drop_costs, padded_drop_costs, padded_zx_costs, Ns, Ks = pad_costs(zx_costs_list, drop_costs_list)
    all_zx_costs = torch.stack(padded_zx_costs, dim=-1)
    all_cum_drop_costs = torch.stack(padded_cum_drop_costs, dim=-1)
    all_drop_costs = torch.stack(padded_drop_costs, dim=-1)
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
        multirow_pad = torch.stack([row_pad + inf] * Ks[i], dim=0)
        padded_table = torch.cat([zx_costs, multirow_pad], dim=1)
        rest_pad = torch.zeros([K - Ks[i], N]).to(zx_costs.device) + inf
        padded_table = torch.cat([padded_table, rest_pad], dim=0)
        padded_zx_costs.append(padded_table)

    all_zx_costs = torch.stack(padded_zx_costs, dim=-1)
    all_cum_drop_costs = torch.stack(padded_cum_drop_costs, dim=-1)
    all_drop_costs = torch.stack(padded_drop_costs, dim=-1)

    # Creating and initializing DP tables
    D = VarTable((K + 1, N + 1, 3, B))  # This corresponds to B 3-dim DP tables
    for zi in range(1, K + 1):
        D[zi, 0] = torch.zeros_like(D[zi, 0]) + inf
    for xi in range(1, N + 1):
        if drop_mode == "DropDTW":
            D[0, xi] = torch.zeros_like(D[0, xi]) + all_cum_drop_costs[(xi - 1) : xi]
        elif drop_mode == "OTAM":
            D[0, xi] = torch.zeros_like(D[0, xi])
        else:  # drop_mode == 'DTW'
            D[0, xi] = torch.zeros_like(D[0, xi]) + inf

    # obtain dtw table using min_gamma or softMin relaxation
    for zi in range(1, K + 1):
        for xi in range(1, N + 1):
            z_cost_ind, x_cost_ind = zi - 1, xi - 1  # indexind in costs is shifted by 1

            d_diag, d_left = D[zi - 1, xi - 1][0:1], D[zi, xi - 1][0:1]
            dp_left, dp_up = D[zi, xi - 1][2:3], D[zi - 1, xi][2:3]

            if drop_mode == "DropDTW":
                # positive transition, i.e. matching x_i to z_j
                if contiguous:
                    pos_neighbors = [d_diag, dp_left]
                else:
                    pos_neighbors = [d_diag, d_left]
                if not exclusive:
                    pos_neighbors.append(dp_up)

                Dp = min_fn(pos_neighbors, gamma=gamma_min) + all_zx_costs[z_cost_ind, x_cost_ind]

                # negative transition, i.e. dropping xi
                Dm = d_left + all_drop_costs[x_cost_ind]

                # update final solution matrix
                D_final = min_fn([Dm, Dp], gamma=gamma_min)
            else:
                d_right = D[zi - 1, xi][0:1]
                D_final = Dm = Dp = (
                    min_fn([d_diag, d_left, d_right], gamma=gamma_min) + all_zx_costs[z_cost_ind, x_cost_ind]
                )
            D[zi, xi] = torch.cat([D_final, Dm, Dp], dim=0)

    # Computing the final min cost for the whole batch
    min_costs = []
    for i in range(B):
        Ni, Ki = Ns[i], Ks[i]
        min_cost_i = D[Ki, Ni][0, i]
        min_costs.append(min_cost_i / Ni)

    return min_costs, D


def batch_double_dropDTW(zx_costs_list, drop_costs_list, gamma_min=1):
    """function to obtain a soft (differentiable version of DTW)
    embs1, embs2: embedding of size N*D and M*D (N and M : number of video frames
    and D: dimensionality of of the embedding vector)
    """
    min_fn = lambda x: minProb(x, gamma=gamma_min)
    dev, dtype = zx_costs_list[0].device, zx_costs_list[0].dtype

    # assuming sequences are the same length
    B = len(zx_costs_list)
    padded_cum_drop_costs, padded_drop_costs, padded_zx_costs, Ns, Ks = pad_costs(zx_costs_list, drop_costs_list)
    all_zx_costs = torch.stack(padded_zx_costs, dim=-1)
    all_cum_drop_costs = torch.stack(padded_cum_drop_costs, dim=-1)
    all_drop_costs = torch.stack(padded_drop_costs, dim=-1)
    N, K = max(Ns), max(Ks)

    # Creating and initializing DP tables
    D = VarTable((K + 1, N + 1, 4, B), dtype, dev)  # This corresponds to B 4-dim DP tables
    for zi in range(1, K + 1):
        D[zi, 0] = torch.zeros_like(D[zi, 0]) + all_cum_drop_costs[(zi - 1) : zi]
    for xi in range(1, N + 1):
        D[0, xi] = torch.zeros_like(D[0, xi]) + all_cum_drop_costs[(xi - 1) : xi]

    for zi in range(1, K + 1):
        for xi in range(1, N + 1):
            # define frequently met neighbors here
            diag_neigh_states = [0, 1, 2, 3]  # zx, z-, -x, --
            diag_neigh_costs = [D[zi - 1, xi - 1][s] for s in diag_neigh_states]

            left_neigh_states = [0, 1]  # zx and z-
            left_neigh_costs = [D[zi, xi - 1][s] for s in left_neigh_states]

            upper_neigh_states = [0, 2]  # zx and -x
            upper_neigh_costs = [D[zi - 1, xi][s] for s in upper_neigh_states]

            z_cost_ind, x_cost_ind = zi - 1, xi - 1  # indexind in costs is shifted by 1

            # DP 0: coming to zx
            neigh_costs_zx = diag_neigh_costs + upper_neigh_costs + left_neigh_costs
            D0 = min_fn(neigh_costs_zx) + all_zx_costs[z_cost_ind, x_cost_ind]

            # DP 1: coming to z-
            neigh_costs_z_ = left_neigh_costs
            D1 = min_fn(neigh_costs_z_) + all_drop_costs[x_cost_ind]

            # DP 2: coming to -x
            neigh_costs__x = upper_neigh_costs
            D2 = min_fn(neigh_costs__x) + all_drop_costs[z_cost_ind]

            # DP 3: coming to --
            costs___ = [d + all_drop_costs[z_cost_ind] * 2 for d in diag_neigh_costs] + [
                D[zi, xi - 1][3] + all_drop_costs[x_cost_ind],
                D[zi - 1, xi][3] + all_drop_costs[z_cost_ind],
            ]
            D3 = min_fn(costs___)

            D[zi, xi] = torch.cat([D0, D1, D2, D3], dim=0)

    # Computing the final min cost for the whole batch
    min_costs = []
    for i in range(B):
        min_cost_i = min_fn(D[K, N][:, i])
        min_costs.append(min_cost_i / N)
    return min_costs, D


def drop_dtw_machine(zx_costs, drop_costs, gamma_min=1, exclusive=True, contiguous=True):
    K, N = zx_costs.shape
    dev = zx_costs.device
    flipped_costs = torch.flip(zx_costs, [0])  # flip the cost matrix upside down
    cum_drop_costs = torch.cumsum(drop_costs, dim=-1)

    # initialize first two contr diagonals
    inf = torch.tensor([9999999999], device=dev, dtype=zx_costs.dtype)
    diag_pp = torch.zeros([1, 2], device=dev)  # diag at i-2
    diag_p_col = torch.ones([1, 2], device=dev) * inf
    diag_p_row = torch.stack([inf, cum_drop_costs[[0]]], -1)
    diag_p = torch.cat([diag_p_row, diag_p_col], 0)  # diag at i-1

    for i in range(K + N - 1):
        size = diag_p.size(0) - 1
        pp_start = max(0, diag_pp.size(0) - diag_p.size(0))
        neigh_up, neigh_left, neigh_diag = diag_p[:-1], diag_p[1:], diag_pp[pp_start : (pp_start + size)]
        neigh_up_pos, neigh_left_pos = neigh_up[:, [0]], neigh_left[:, [0]]

        # define match and drop cost vectors
        match_costs_diag = torch.flip(torch.diag(flipped_costs, i + 1 - K), [-1])
        d_start, d_end = max(1 - K + i, 0), min(i, N - 1) + 1
        drop_costs_diag = torch.flip(drop_costs[d_start:d_end], [-1])

        # update positive and negative tables -> compute new diagonal
        pos_neighbors = [neigh_diag, neigh_left_pos] if contiguous else [neigh_diag, neigh_left]
        if not exclusive:
            pos_neighbors.append(neigh_up_pos)
        diag_pos = prob_min(pos_neighbors, gamma_min) + match_costs_diag
        diag_neg = prob_min([neigh_left], gamma_min) + drop_costs_diag
        diag = torch.stack([diag_pos, diag_neg], -1)

        # add the initialization values on the ends of diagonal if needed
        if i < N - 1:
            # fill in 0th row with [drop_cost, inf]
            pad = torch.stack([inf, cum_drop_costs[[i + 1]]], -1)
            diag = torch.cat([pad, diag])
        if i < K - 1:
            # fill in 0th col with [inf, inf]
            pad = torch.stack([inf, inf], -1)
            diag = torch.cat([diag, pad])

        diag_pp = diag_p
        diag_p = diag
    assert (diag.size(0) == 1) and (diag.size(1) == 2), f"Last diag shape is {diag.shape} instead of [1, 2]"

    cost = prob_min(diag, gamma_min)
    return cost


def batch_drop_dtw_machine(zx_costs_list, drop_costs_list, gamma_min=1, exclusive=True, contiguous=True):
    dev, dtype = zx_costs_list[0].device, zx_costs_list[0].dtype
    inf = torch.tensor([9999999999], device=dev, dtype=dtype)
    B = len(zx_costs_list)

    # For samples where K > N, exclusive computation is not possible
    shapes = [t.shape for t in zx_costs_list]
    Ks, Ns = [s[0] for s in shapes], [s[1] for s in shapes]
    N, K = max(Ns), max(Ks)
    persample_exclusive = torch.tensor([Ni >= Ki for Ki, Ni in shapes]).to(dev)

    # transform endpoints into diagonal coordinates
    Ds, Rs = torch.zeros(B).to(dev).to(int), torch.zeros(B).to(dev).to(int)
    for i, (Ki, Ni) in enumerate(zip(Ks, Ns)):
        Ds[i] = Ki + Ni - 2
        Rs[i] = min(Ds[i] + 2, N) - Ni
    Ds_orig, Rs_orig = copy(Ds), copy(Rs)

    # define costs in tensors
    all_zx_costs = [F.pad(c, [0, N - c.shape[1], 0, K - c.shape[0]]) for c in zx_costs_list]
    all_zx_costs = torch.stack(all_zx_costs, 0)

    all_drop_costs = torch.stack([F.pad(c, [0, N - c.shape[0]], value=inf.item()) for c in drop_costs_list], 0)
    all_cum_drop_costs = torch.stack(
        [F.pad(torch.cumsum(c, -1), [0, N - c.shape[0]], value=inf.item()) for c in drop_costs_list], 0
    )
    flipped_costs = torch.flip(all_zx_costs, [1])  # flip the cost matrix upside down

    """Rules for the diagonals:
        dim1: batch dimension
        dim2: the diagonal itself. The first element along this dim corresponds
              to the top right element on the diagonal. The movement is from top right
              to bottom left, like that /
        dim3: Keep and Drop dimensions of the DP table. Here, 0 is keep, 1 is drop.
     """
    # initialize first two contr diagonals
    batch_inf, batch_ones = torch.stack([inf] * B, 0), torch.ones([B, 1], device=dev, dtype=dtype)
    diag_pp = torch.zeros([B, 1, 2], device=dev)  # diag at i-2
    diag_p_col = torch.ones([B, 1, 2], device=dev) * batch_inf[..., None]
    diag_p_row = torch.stack([batch_inf, all_cum_drop_costs[:, [0]]], -1)
    diag_p = torch.cat([diag_p_row, diag_p_col], 1)  # diag at i-1

    # The pathlength path is also a diagonal representation that carries the optimal pathlength to each point
    with torch.no_grad():
        path_pp = torch.zeros([B, 1, 2], device=dev, dtype=dtype)
        path_p = torch.ones([B, 2, 2], device=dev, dtype=dtype)

    min_costs = torch.zeros(B).to(dtype=dtype).to(device=dev)
    path_lens = torch.zeros(B).to(dtype=dtype).to(device=dev)
    for d in range(K + N - 1):
        size = diag_p.size(1) - 1
        pp_start = 0 if d < N else 1
        neigh_up, neigh_left, neigh_diag = diag_p[:, :-1], diag_p[:, 1:], diag_pp[:, pp_start : (pp_start + size)]
        neigh_up_pos, neigh_left_pos = neigh_up[..., [0]], neigh_left[..., [0]]

        neigh_path_up, neigh_path_left, neigh_path_diag = (
            path_p[:, :-1],
            path_p[:, 1:],
            path_pp[:, pp_start : (pp_start + size)],
        )
        neigh_path_up_pos, neigh_path_left_pos = neigh_path_up[..., [0]], neigh_path_left[..., [0]]

        # define match and drop cost vectors
        match_costs_diag = torch.stack(
            [torch.flip(torch.diag(flipped_costs[j], d + 1 - K), [-1]) for j in range(flipped_costs.size(0))], 0
        )

        d_start, d_end = max(1 - K + d, 0), min(d, N - 1) + 1
        drop_costs_diag = torch.flip(all_drop_costs[:, d_start:d_end], [-1])

        # update positive and negative tables -> compute new diagonal
        pos_neighbors = [neigh_diag, neigh_left_pos] if contiguous else [neigh_diag, neigh_left]
        pos_path_neighbors = (
            [neigh_path_diag, neigh_path_left_pos] if contiguous else [neigh_path_diag, neigh_path_left]
        )
        if exclusive and (~persample_exclusive).any():
            # apply non-exclusive rule for some batch elements, via masing out the exclusive elements with inf
            masked_neigh_up_pos = neigh_up_pos + persample_exclusive[:, None, None] * batch_inf[:, None]
            pos_neighbors.append(masked_neigh_up_pos)

            pos_path_neighbors.append(neigh_path_up_pos * (~persample_exclusive[:, None, None]))
        elif not exclusive:
            # apply standard non-exclusive rule to all batch elements
            pos_neighbors.append(neigh_up_pos)
            pos_path_neighbors.append(neigh_path_up_pos)

        # DP Table update
        diag_pos = prob_min(pos_neighbors, gamma_min) + match_costs_diag
        diag_neg = prob_min([neigh_left], gamma_min) + drop_costs_diag
        diag = torch.stack([diag_pos, diag_neg], -1)

        # Path Table Update
        with torch.no_grad():
            path_pos = prob_min(pos_path_neighbors, gamma_min, pos_neighbors) + 1
            path_neg = prob_min([neigh_path_left], gamma_min, [neigh_left]) + 1
            path = torch.stack([path_pos, path_neg], -1)

        # add the initialization values on the ends of diagonal if needed
        if d < N - 1:
            # fill in DP table's 0th row with [drop_cost, inf]
            pad_d = torch.stack([batch_inf, all_cum_drop_costs[:, [d + 1]]], -1)
            diag = torch.cat([pad_d, diag], 1)

            # fill in Path table's 0th row with [d, inf]
            pad_p = torch.stack([batch_inf, torch.zeros_like(batch_inf) + d], -1)
            path = torch.cat([pad_p, path], 1)

        if d < K - 1:
            # fill in DP table's 0th col with [inf, inf]
            pad_d = torch.stack([batch_inf, batch_inf], -1)
            diag = torch.cat([diag, pad_d], 1)

            # fill in Path table's 0th row with [d, inf]
            pad_p = pad_d
            path = torch.cat([path, pad_p], 1)

        diag_pp = diag_p
        diag_p = diag

        path_pp = path_p
        path_p = path

        # process answers
        if (Ds == d).any():
            mask, orig_mask = Ds == d, Ds_orig == d
            bs, rs = torch.nonzero(mask, as_tuple=False)[:, 0], Rs[mask]
            min_costs[orig_mask] = min_costs[orig_mask] + prob_min([diag[bs, rs]], gamma_min)
            path_lens[orig_mask] = path_lens[orig_mask] + prob_min([path[bs, rs]], gamma_min, [diag[bs, rs]])

            diag, diag_p, diag_pp, path, path_p, path_pp, Ds, Rs, flipped_costs = [
                t[~mask] for t in [diag, diag_p, diag_pp, path, path_p, path_pp, Ds, Rs, flipped_costs]
            ]
            all_drop_costs, all_cum_drop_costs, batch_inf, persample_exclusive = [
                t[~mask] for t in [all_drop_costs, all_cum_drop_costs, batch_inf, persample_exclusive]
            ]
            if torch.numel(Ds) == 0:
                break

    # costs = prob_min([diag], gamma_min)
    costs_norm = min_costs / path_lens
    return min_costs, path_lens


def batch_double_drop_dtw_machine(
    zx_costs_list, x_drop_costs_list, z_drop_costs_list, gamma_min=1, exclusive=True, contiguous=True
):
    dev, dtype = zx_costs_list[0].device, zx_costs_list[0].dtype
    inf = torch.tensor([9999999999], device=dev, dtype=dtype)
    B = len(zx_costs_list)

    Ns, Ks = [], []
    for i in range(B):
        Ki, Ni = zx_costs_list[i].shape
        if exclusive and Ki >= Ni:
            # in case the number of steps is greater than the number of frames,
            # duplicate every frame and let the drops do the job.
            mult = math.ceil(Ki / Ni)
            zx_costs_list[i] = torch.stack([zx_costs_list[i]] * mult, dim=-1).reshape([Ki, -1])
            x_drop_costs_list[i] = torch.stack([x_drop_costs_list[i]] * mult, dim=-1).reshape([-1])
            Ni *= mult
        Ns.append(Ni)
        Ks.append(Ki)
    N, K = max(Ns), max(Ks)

    # transform endpoints into diagonal coordinates
    Ds, Rs = torch.zeros(B).to(dev).to(int), torch.zeros(B).to(dev).to(int)
    for i, (Ki, Ni) in enumerate(zip(Ks, Ns)):
        Ds[i] = Ki + Ni - 2
        Rs[i] = min(Ds[i] + 2, N) - Ni
    Ds_orig, Rs_orig = copy(Ds), copy(Rs)

    # special padding of costs to ensure that the path goest through the endpoint
    all_zx_costs = []
    for i, c in enumerate(zx_costs_list):
        c_inf_frame = F.pad(c, [0, 1, 0, 1], value=inf.item())
        mask = torch.ones_like(c_inf_frame)
        mask[-1, -1] = 0
        c_pad = F.pad(c_inf_frame * mask, [0, N - c.shape[1] - 1, 0, K - c.shape[0] - 1])
        all_zx_costs.append(c_pad)
    all_zx_costs = torch.stack(all_zx_costs, 0)

    all_x_drop_costs = torch.stack([F.pad(c, [0, N - c.shape[0]], value=inf.item()) for c in x_drop_costs_list], 0)
    all_cum_x_drop_costs = torch.stack(
        [F.pad(torch.cumsum(c, -1), [0, N - c.shape[0]], value=inf.item()) for c in x_drop_costs_list], 0
    )
    all_z_drop_costs = torch.stack([F.pad(c, [0, K - c.shape[0]], value=inf.item()) for c in z_drop_costs_list], 0)
    all_cum_z_drop_costs = torch.stack(
        [F.pad(torch.cumsum(c, -1), [0, K - c.shape[0]], value=inf.item()) for c in z_drop_costs_list], 0
    )
    flipped_costs = torch.flip(all_zx_costs, [1])  # flip the cost matrix upside down

    """Rules for the diagonals:
        dim1: batch dimension
        dim2: the diagonal itself. The first element along this dim corresponds
              to the top right element on the diagonal. The movement is from top right
              to bottom left, like that /
        dim3: Keep and Drop dimensions of the DP table. The dimensions are as follows:
              {0: zx, 1: z-, 2: -x, 3: --}
     """
    # initialize first two contr diagonals
    batch_inf = torch.stack([inf] * B, 0)
    diag_pp = torch.zeros([B, 1, 4], device=dev)  # diag at i-2
    x1_dropcost, z1_dropcost = all_cum_x_drop_costs[:, [0]], all_cum_z_drop_costs[:, [0]]
    diag_p_row = torch.stack([batch_inf, x1_dropcost, batch_inf, x1_dropcost], -1)
    diag_p_col = torch.stack([batch_inf, batch_inf, z1_dropcost, z1_dropcost], -1)
    diag_p = torch.cat([diag_p_row, diag_p_col], 1)  # diag at i-1

    min_costs = torch.zeros(B).to(dtype=dtype).to(device=dev)  # for storing the solution for each element
    for d in range(K + N - 1):
        size = diag_p.size(1) - 1
        pp_start = 0 if d < N else 1
        neigh_up, neigh_left, neigh_diag = diag_p[:, :-1], diag_p[:, 1:], diag_pp[:, pp_start : (pp_start + size)]
        neigh_left_pos, neigh_left_neg = neigh_left[..., [0, 1]], neigh_left[..., [2, 3]]
        neigh_up_pos, neigh_up_neg = neigh_up[..., [0, 2]], neigh_up[..., [1, 3]]

        # define match and drop cost vectors
        match_costs_diag = torch.stack(
            [torch.flip(torch.diag(flipped_costs[j], d + 1 - K), [-1]) for j in range(flipped_costs.size(0))], 0
        )

        x_d_start, x_d_end = max(d + 1 - K, 0), min(d, N - 1) + 1
        x_drop_costs_diag = torch.flip(all_x_drop_costs[:, x_d_start:x_d_end], [-1])
        z_d_start, z_d_end = max(d + 1 - N, 0), min(d, K - 1) + 1
        z_drop_costs_diag = all_z_drop_costs[:, z_d_start:z_d_end]

        # update positive and negative tables -> compute new diagonal

        # DP 0: coming to zx
        neighbors_zx = [neigh_diag, neigh_left_pos[..., [0]]] if contiguous else [neigh_diag, neigh_left_pos]
        if not exclusive:
            neighbors_zx.append(neigh_up_pos)
        diag_zx = prob_min(neighbors_zx, gamma_min) + match_costs_diag

        # DP 1: coming to z-
        neighbors_z_ = [neigh_left_pos]
        diag_z_ = prob_min(neighbors_z_, gamma_min) + x_drop_costs_diag

        # DP 2: coming to -x
        neighbors__x = [neigh_up_pos]
        diag__x = prob_min(neighbors__x, gamma_min) + z_drop_costs_diag

        # DP 3: coming to --
        neighbors___ = [neigh_left_neg + x_drop_costs_diag[..., None], neigh_up_neg + z_drop_costs_diag[..., None]]
        diag___ = prob_min(neighbors___, gamma_min)

        # Aggregating all the dimensions of DP together
        diag = torch.stack([diag_zx, diag_z_, diag__x, diag___], -1)

        # Haven't done below
        # add the initialization values on the ends of diagonal if needed
        if d < N - 1:
            # fill in 0th row with [drop_cost, inf]
            x_drop_cost = all_cum_x_drop_costs[:, [d + 1]]
            pad = torch.stack([batch_inf, x_drop_cost, batch_inf, x_drop_cost], -1)
            diag = torch.cat([pad, diag], 1)
        if d < K - 1:
            # fill in 0th col with [inf, inf]
            z_drop_cost = all_cum_z_drop_costs[:, [d + 1]]
            pad = torch.stack([batch_inf, batch_inf, z_drop_cost, z_drop_cost], -1)
            diag = torch.cat([diag, pad], 1)

        diag_pp = diag_p
        diag_p = diag

        # process answers
        if (Ds == d).any():
            mask, orig_mask = Ds == d, Ds_orig == d
            bs, rs = torch.nonzero(mask, as_tuple=False)[:, 0], Rs[mask]
            min_costs[orig_mask] = min_costs[orig_mask] + prob_min([diag[bs, rs]], gamma_min)

            # filtering out already processed elements
            diag, diag_p, diag_pp, Ds, Rs, flipped_costs = [
                t[~mask] for t in [diag, diag_p, diag_pp, Ds, Rs, flipped_costs]
            ]
            all_x_drop_costs, all_z_drop_costs, all_cum_x_drop_costs, all_cum_z_drop_costs = [
                t[~mask] for t in [all_x_drop_costs, all_z_drop_costs, all_cum_x_drop_costs, all_cum_z_drop_costs]
            ]

            if torch.numel(Ds) == 0:
                break

    costs_norm = min_costs / torch.tensor(Ns).to(dev)
    return costs_norm


if __name__ == "__main__":
    from exact_dp import double_drop_dtw

    K, N = 7, 15
    zx_costs = torch.normal(torch.ones([K, N]))
    x_drop_costs = zx_costs.mean(0)
    z_drop_costs = zx_costs.mean(1)

    min_cost, *_ = double_drop_dtw(zx_costs.numpy(), x_drop_costs.numpy(), z_drop_costs.numpy())
    my_costs = batch_double_drop_dtw_machine([zx_costs], [x_drop_costs], [z_drop_costs], gamma_min=0)
    print(my_costs * N, min_cost)
