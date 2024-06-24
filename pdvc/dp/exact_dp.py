import torch
import numpy as np
import torch.nn.functional as F
from functools import partial
from copy import copy

# from dp.dp_utils import get_diag_coord_grid, diag_traceback, nw_diag_traceback, list_min
from pdvc.dp.dp_utils import get_diag_coord_grid, diag_traceback, nw_diag_traceback, list_min


def crosstask_dp(cost_matrix, exactly_one=True, bg_cost=0):
    "Algorithm used in Cross-Task to calculate Recall"

    def get_step(k):
        return 0 if k % 2 == 0 else int((k + 1) / 2)

    T = cost_matrix.shape[0]
    K = cost_matrix.shape[1]
    K_ext = int(2 * K + 1)

    L = -np.ones([T + 1, K_ext], dtype=float)
    P = -np.ones([T + 1, K_ext], dtype=float)
    L[0, 0] = 0
    P[0, 0] = 0

    for t in range(1, T + 1):
        Lt = L[t - 1, :]
        Pt = P[t - 1, :]
        for k in range(K_ext):
            s = get_step(k)
            opt_label = -1

            j = k
            if (opt_label == -1 or opt_value > Lt[j]) and Pt[j] != -1 and (s == 0 or not exactly_one):
                opt_label = j
                opt_value = Lt[j]

            j = k - 1
            if j >= 0 and (opt_label == -1 or opt_value > Lt[j]) and Pt[j] != -1:
                opt_label = j
                opt_value = L[t - 1][j]

            if s != 0:
                j = k - 2
                if j >= 0 and (opt_label == -1 or opt_value > Lt[j]) and Pt[j] != -1:
                    opt_label = j
                    opt_value = Lt[j]

            if s != 0:
                L[t, k] = opt_value + cost_matrix[t - 1][s - 1]
            else:
                L[t, k] = opt_value + bg_cost
            P[t, k] = opt_label

    labels = np.zeros_like(cost_matrix)
    if L[T, K_ext - 1] < L[T, K_ext - 2] or (P[T, K_ext - 2] == -1):
        k = K_ext - 1
    else:
        k = K_ext - 2
    for t in range(T, 0, -1):
        s = get_step(k)
        if s > 0:
            labels[t - 1, s - 1] = 1
        k = P[t, k].astype(int)
    return labels


def iou_based_matching(pred_seg, gt_seg, pred_step_ids, gt_step_ids, ignore_class=True):
    """Performs the matching of predicted and gt sequence segments"""
    pred_segments = torch.stack([pred_seg == idx for idx in pred_step_ids], 0)  # [N_pred, T]
    gt_segments = torch.stack([gt_seg == idx for idx in gt_step_ids], 0)  # [N_gt, T]
    intersection = (
        torch.logical_and(pred_segments.unsqueeze(1), gt_segments.unsqueeze(0)).to(int).sum(-1)
    )  # [N_pred, N_gt]
    union = torch.logical_or(pred_segments.unsqueeze(1), gt_segments.unsqueeze(0)).to(int).sum(-1)  # [N_pred, N_gt]
    iou = intersection / (union + 1e-5)  # [N_pred, N_gt]

    C = -iou.detach().cpu().numpy().T  # [N_gt, N_pred]
    if not ignore_class:
        print("Not ignoring class")
        is_same_step_id = pred_step_ids.unsqueeze(1) == gt_step_ids.unsqueeze(0)  # [N_pred, N_gt]
        if is_same_step_id.shape == (1, 1):
            C[0, 0] += 9999 * (~is_same_step_id[0, 0])
        else:
            C[~is_same_step_id] = 9999

    x_drop, z_drop = np.zeros(C.shape[1]), np.zeros(C.shape[0])
    labels = double_drop_dtw(C, x_drop, z_drop, one_to_many=False, many_to_one=False, return_labels=True) - 1
    indices = (np.arange(len(labels))[labels > -1], labels[labels > -1])
    return [torch.as_tensor(i, dtype=torch.int64) for i in indices]


def drop_dtw(zx_costs, drop_costs, exclusive=True, contiguous=True, one_to_one=False, return_labels=False):
    """Drop-DTW algorithm that allows drop only from one (video) side. See Algorithm 1 in the paper.

    Parameters
    ----------
    zx_costs: np.ndarray [K, N]
        pairwise match costs between K steps and N video clips
    drop_costs: np.ndarray [N]
        drop costs for each clip
    exclusive: bool
        If True any clip can be matched with only one step, not many.
    contiguous: bool
        if True, can only match a contiguous sequence of clips to a step
        (i.e. no drops in between the clips)
    return_label: bool
        if True, returns output directly useful for segmentation computation (made for convenience)
    """
    K, N = zx_costs.shape

    # D: the dynamic programming table, which records the intermediate costs
    # P: the path tracking table, which records the previous location and state (zi, xi, prev_state)

    # initialize solutin matrices
    D = np.zeros([K + 1, N + 1, 2])  # the 2 last dimensions correspond to different states.
    # State (dim) 0 - x is matched; State 1 - x is dropped
    D[1:, 0, :] = np.inf  # no drops in z in any state
    D[0, 1:, 0] = np.inf  # no drops in x in state 0, i.e. state where x is matched
    D[0, 1:, 1] = np.cumsum(drop_costs)  # drop costs initizlization in state 1

    # initialize path tracking info for each state
    P = np.zeros([K + 1, N + 1, 2, 3], dtype=int) # the last dimension records the previous location and state (zi, xi, prev_state)
    for xi in range(1, N + 1):
        P[0, xi, 1] = 0, xi - 1, 1
    # filling in the dynamic tables
    for zi in range(1, K + 1):
        for xi in range(1, N + 1):
            # define frequently met neighbors here
            diag_neigh_states = [0, 1]
            diag_neigh_coords = [(zi - 1, xi - 1) for _ in diag_neigh_states]
            diag_neigh_costs = [D[zi - 1, xi - 1, s] for s in diag_neigh_states]

            left_neigh_states = [0, 1]
            left_neigh_coords = [(zi, xi - 1) for _ in left_neigh_states]
            left_neigh_costs = [D[zi, xi - 1, s] for s in left_neigh_states]

            left_pos_neigh_states = [0] if contiguous else left_neigh_states
            left_pos_neigh_coords = [(zi, xi - 1) for _ in left_pos_neigh_states]
            left_pos_neigh_costs = [D[zi, xi - 1, s] for s in left_pos_neigh_states] # Drop between clips is not allowed when setting `contiguous==True` (one step to sparse clips is not allowed)

            top_pos_neigh_states = [0]
            top_pos_neigh_coords = [(zi - 1, xi) for _ in top_pos_neigh_states]
            top_pos_neigh_costs = [D[zi - 1, xi, s] for s in top_pos_neigh_states]

            z_cost_ind, x_cost_ind = zi - 1, xi - 1  # indexind in costs is shifted by 1

            # state 0: matching x to z
            neigh_states_pos = diag_neigh_states
            neigh_coords_pos = diag_neigh_coords
            neigh_costs_pos = diag_neigh_costs
            if not one_to_one:
                neigh_states_pos = neigh_states_pos + left_pos_neigh_states
                neigh_coords_pos = neigh_coords_pos + left_pos_neigh_coords
                neigh_costs_pos = neigh_costs_pos + left_pos_neigh_costs
            if not exclusive: # exclusive=True indicates any clip can be matched with only one step, that is, path from top is not allowed
                neigh_states_pos = neigh_states_pos + top_pos_neigh_states
                neigh_coords_pos = neigh_coords_pos + top_pos_neigh_coords
                neigh_costs_pos = neigh_costs_pos + left_pos_neigh_costs + top_pos_neigh_costs

            costs_pos = np.array(neigh_costs_pos) + zx_costs[z_cost_ind, x_cost_ind] # calculate cumulative cost in current step
            opt_ind_pos = np.argmin(costs_pos)
            P[zi, xi, 0] = *neigh_coords_pos[opt_ind_pos], neigh_states_pos[opt_ind_pos] # Records the last step's position (zi,xi) and state (0 or 1)
            D[zi, xi, 0] = costs_pos[opt_ind_pos]  # Update the minimal cumulative cost of selected path

            # state 1: x is dropped
            costs_neg = np.array(left_neigh_costs) + drop_costs[x_cost_ind]
            opt_ind_neg = np.argmin(costs_neg)
            P[zi, xi, 1] = *left_neigh_coords[opt_ind_neg], left_neigh_states[opt_ind_neg]
            D[zi, xi, 1] = costs_neg[opt_ind_neg]

    cur_state = D[K, N, :].argmin()
    min_cost = D[K, N, cur_state]
    #breakpoint()

    # backtracking the solution
    zi, xi = K, N
    path, labels = [], np.zeros(N)
    x_dropped = [] if cur_state == 1 else [N]
    while not (zi == 0 and xi == 0):
        path.append((zi, xi))
        zi_prev, xi_prev, prev_state = P[zi, xi, cur_state]
        if xi > 0:
            labels[xi - 1] = zi * (cur_state == 0)  # either zi or 0
        if prev_state == 1:
            x_dropped.append(xi_prev)
        zi, xi, cur_state = zi_prev, xi_prev, prev_state

    if not return_labels:
        return min_cost, D, path, x_dropped
    else:
        return labels


def double_drop_dtw(
    pairwise_zx_costs,
    x_drop_costs,
    z_drop_costs,
    contiguous=True,
    one_to_many=True,
    many_to_one=True,
    return_labels=False,
):
    """Drop-DTW algorithm that allows drops from both sequences. See Algorithm 1 in Appendix.

    Parameters
    ----------
    pairwise_zx_costs: np.ndarray [K, N]
        pairwise match costs between K steps and N video clips
    x_drop_costs: np.ndarray [N]
        drop costs for each clip
    z_drop_costs: np.ndarray [N]
        drop costs for each step
    contiguous: bool
        if True, can only match a contiguous sequence of clips to a step
        (i.e. no drops in between the clips)
    """
    K, N = pairwise_zx_costs.shape

    # initialize solution matrices
    D = np.zeros([K + 1, N + 1, 4])  # the 4 dimensions are the following states: zx, z-, -x, --
    # no drops allowed in zx DP. Setting the same for all DPs to change later here.
    D[1:, 0, :] = 99999999
    D[0, 1:, :] = 99999999
    D[0, 0, 1:] = 99999999
    # Allow to drop x in z- and --
    D[0, 1:, 1], D[0, 1:, 3] = np.cumsum(x_drop_costs), np.cumsum(x_drop_costs)
    # Allow to drop z in -x and --
    D[1:, 0, 2], D[1:, 0, 3] = np.cumsum(z_drop_costs), np.cumsum(z_drop_costs)

    # initialize path tracking info for each of the 4 DP tables:
    P = np.zeros([K + 1, N + 1, 4, 3], dtype=int)  # (zi, xi, prev_state)
    for zi in range(1, K + 1):
        P[zi, 0, 2], P[zi, 0, 3] = (zi - 1, 0, 2), (zi - 1, 0, 3)
    for xi in range(1, N + 1):
        P[0, xi, 1], P[0, xi, 3] = (0, xi - 1, 1), (0, xi - 1, 3)

    # filling in the dynamic tables
    for zi in range(1, K + 1):
        for xi in range(1, N + 1):
            # define frequently met neighbors here
            diag_neigh_states = [0, 1, 2, 3]  # zx, z-, -x, --
            diag_neigh_coords = [(zi - 1, xi - 1) for _ in diag_neigh_states]
            diag_neigh_costs = [D[zi - 1, xi - 1, s] for s in diag_neigh_states]

            left_pos_neigh_states = [0, 1]  # zx and z-
            left_pos_neigh_coords = [(zi, xi - 1) for _ in left_pos_neigh_states]
            left_pos_neigh_costs = [D[zi, xi - 1, s] for s in left_pos_neigh_states]

            top_pos_neigh_states = [0, 2]  # zx and -x
            top_pos_neigh_coords = [(zi - 1, xi) for _ in top_pos_neigh_states]
            top_pos_neigh_costs = [D[zi - 1, xi, s] for s in top_pos_neigh_states]

            left_neg_neigh_states = [2, 3]  # -x and --
            left_neg_neigh_coords = [(zi, xi - 1) for _ in left_neg_neigh_states]
            left_neg_neigh_costs = [D[zi, xi - 1, s] for s in left_neg_neigh_states]

            top_neg_neigh_states = [1, 3]  # z- and --
            top_neg_neigh_coords = [(zi - 1, xi) for _ in top_neg_neigh_states]
            top_neg_neigh_costs = [D[zi - 1, xi, s] for s in top_neg_neigh_states]

            z_cost_ind, x_cost_ind = zi - 1, xi - 1  # indexind in costs is shifted by 1

            # DP 0: coming to zx
            neigh_states_zx = diag_neigh_states
            neigh_coords_zx = diag_neigh_coords
            neigh_costs_zx = diag_neigh_costs
            if one_to_many:
                if contiguous:
                    neigh_states_zx.extend(left_pos_neigh_states[0:1])
                    neigh_coords_zx.extend(left_pos_neigh_coords[0:1])
                    neigh_costs_zx.extend(left_pos_neigh_costs[0:1])
                else:
                    neigh_states_zx.extend(left_pos_neigh_states)
                    neigh_coords_zx.extend(left_pos_neigh_coords)
                    neigh_costs_zx.extend(left_pos_neigh_costs)
            if many_to_one:
                neigh_states_zx.extend(top_pos_neigh_states)
                neigh_coords_zx.extend(top_pos_neigh_coords)
                neigh_costs_zx.extend(top_pos_neigh_costs)

            costs_zx = np.array(neigh_costs_zx) + pairwise_zx_costs[z_cost_ind, x_cost_ind]
            opt_ind_zx = np.argmin(costs_zx)
            P[zi, xi, 0] = *neigh_coords_zx[opt_ind_zx], neigh_states_zx[opt_ind_zx]
            D[zi, xi, 0] = costs_zx[opt_ind_zx]

            # DP 1: coming to z-
            neigh_states_z_ = left_pos_neigh_states
            neigh_coords_z_ = left_pos_neigh_coords
            neigh_costs_z_ = left_pos_neigh_costs
            costs_z_ = np.array(neigh_costs_z_) + x_drop_costs[x_cost_ind]
            opt_ind_z_ = np.argmin(costs_z_)
            P[zi, xi, 1] = *neigh_coords_z_[opt_ind_z_], neigh_states_z_[opt_ind_z_]
            D[zi, xi, 1] = costs_z_[opt_ind_z_]

            # DP 2: coming to -x
            neigh_states__x = top_pos_neigh_states
            neigh_coords__x = top_pos_neigh_coords
            neigh_costs__x = top_pos_neigh_costs
            costs__x = np.array(neigh_costs__x) + z_drop_costs[z_cost_ind]
            opt_ind__x = np.argmin(costs__x)
            P[zi, xi, 2] = *neigh_coords__x[opt_ind__x], neigh_states__x[opt_ind__x]
            D[zi, xi, 2] = costs__x[opt_ind__x]

            # DP 3: coming to --
            neigh_states___ = np.array(left_neg_neigh_states + top_neg_neigh_states)
            # neigh_states___ = np.array(left_neg_neigh_states + top_neg_neigh_states + diag_neigh_states)
            # adding negative left and top neighbors
            neigh_coords___ = np.array(left_neg_neigh_coords + top_neg_neigh_coords)
            # neigh_coords___ = np.array(left_neg_neigh_coords + top_neg_neigh_coords + diag_neigh_coords)
            costs___ = np.concatenate(
                [
                    left_neg_neigh_costs + x_drop_costs[x_cost_ind],
                    top_neg_neigh_costs + z_drop_costs[z_cost_ind],
                    # diag_neigh_costs + z_drop_costs[z_cost_ind] + x_drop_costs[x_cost_ind],
                ],
                0,
            )

            opt_ind___ = costs___.argmin()
            P[zi, xi, 3] = *neigh_coords___[opt_ind___], neigh_states___[opt_ind___]
            D[zi, xi, 3] = costs___[opt_ind___]

    cur_state = D[K, N, :].argmin()
    min_cost = D[K, N, cur_state]

    # unroll path
    path = []
    zi, xi = K, N
    x_dropped = [N] if cur_state in [1, 3] else []
    z_dropped = [K] if cur_state in [2, 3] else []
    while not (zi == 0 and xi == 0):
        path.append((zi, xi))
        zi_prev, xi_prev, prev_state = P[zi, xi, cur_state]
        if prev_state in [1, 3]:
            x_dropped.append(xi_prev)
        if prev_state in [2, 3]:
            z_dropped.append(zi_prev)
        zi, xi, cur_state = zi_prev, xi_prev, prev_state

    if return_labels:
        labels = np.zeros(N)
        for zi, xi in path:
            if zi not in z_dropped and xi not in x_dropped:
                labels[xi - 1] = zi
        return labels
    else:
        return min_cost, path, x_dropped, z_dropped


def batch_double_drop_dtw_machine(
    zx_costs_list, x_drop_costs_list, z_drop_costs_list, many_to_one=False, one_to_many=False, contiguous=True
):
    # many_to_one is the same as not exclusive, i.e. multiple z match to one x
    # one_to_many was always true by default before, i.e. multiple x match to one z
    dev, dtype = zx_costs_list[0].device, zx_costs_list[0].dtype
    inf = torch.tensor([9999999999], device=dev, dtype=dtype)
    B = len(zx_costs_list)

    shapes = [t.shape for t in zx_costs_list]
    Ks, Ns = [s[0] for s in shapes], [s[1] for s in shapes]
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

    # The path is also a diagonal representation that carries the optimal pathlength to each point
    path_pp = torch.zeros([B, 1, 4, 3], device=dev, dtype=int)
    path_p = torch.zeros([B, 2, 4, 3], device=dev, dtype=int)
    all_paths = [path_pp, path_p]  # going to store all the intermediate paths diagonals for the backtrack

    # Coords is also a diagonal representation that carries the current coordinates in [d, r] for each point
    # the last dimension is 3 because it's [d, r, s], where d is a diagonal, r is element's order in the diagonal
    # and s is statet (one of the 4)
    coord_pp = get_diag_coord_grid(B, 1, 4, 0).to(dev)
    coord_p = get_diag_coord_grid(B, 2, 4, 1).to(dev)

    min_costs = torch.zeros(B).to(dtype=dtype).to(device=dev)  # for storing the solution for each element
    tracebacks = [None for _ in range(B)]  # going to store all the intermediate paths diagonals for the backtrack

    for d in range(K + N - 1):
        size = diag_p.size(1) - 1
        pp_start = 0 if d < N else 1
        neigh_up, neigh_left, neigh_diag = diag_p[:, :-1], diag_p[:, 1:], diag_pp[:, pp_start : (pp_start + size)]
        neigh_left_pos, neigh_left_neg = neigh_left[..., [0, 1]], neigh_left[..., [2, 3]]
        neigh_up_pos, neigh_up_neg = neigh_up[..., [0, 2]], neigh_up[..., [1, 3]]

        coord_up, coord_left, coord_diag = coord_p[:, :-1], coord_p[:, 1:], coord_pp[:, pp_start : (pp_start + size)]
        coord_left_pos, coord_left_neg = coord_left[..., [0, 1], :], coord_left[..., [2, 3], :]
        coord_up_pos, coord_up_neg = coord_up[..., [0, 2], :], coord_up[..., [1, 3], :]

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
        neighbors_zx = [neigh_diag]
        coordinates_zx = [coord_diag]
        if one_to_many:
            neighbors_zx.append(neigh_left_pos[..., [0]] if contiguous else neigh_left)
            coordinates_zx.append(coord_left_pos[..., [0], :] if contiguous else coord_left)
        if many_to_one:
            neighbors_zx.append(neigh_up_pos)
            coordinates_zx.append(coord_up_pos)
        diag_zx = list_min(neighbors_zx) + match_costs_diag
        path_zx = list_min(coordinates_zx, keys=neighbors_zx)

        # DP 1: coming to z-
        neighbors_z_ = [neigh_left_pos]
        coordinates_z_ = [coord_left_pos]
        diag_z_ = list_min(neighbors_z_) + x_drop_costs_diag
        path_z_ = list_min(coordinates_z_, keys=neighbors_z_)

        # DP 2: coming to -x
        neighbors__x = [neigh_up_pos]
        coordinates__x = [coord_up_pos]
        diag__x = list_min(neighbors__x) + z_drop_costs_diag
        path__x = list_min(coordinates__x, keys=neighbors__x)

        # DP 3: coming to --
        neighbors___ = [neigh_left_neg + x_drop_costs_diag[..., None], neigh_up_neg + z_drop_costs_diag[..., None]]
        coordinates___ = [coord_left_neg, coord_up_neg]
        diag___ = list_min(neighbors___)
        path___ = list_min(coordinates___, neighbors___)

        # Aggregating all the dimensions of DP together
        diag = torch.stack([diag_zx, diag_z_, diag__x, diag___], -1)
        path = torch.stack([path_zx, path_z_, path__x, path___], -2)

        # Haven't done below
        # add the initialization values on the ends of diagonal if needed
        effective_d = d + 2  # effective count of d is actually d + 2, since started with 2
        if d < N - 1:
            # fill in 0th row of cost matrix with [inf, x_drop_cost, inf, x_drop_cost]
            x_drop_cost = all_cum_x_drop_costs[:, [d + 1]]
            cost_pad = torch.stack([batch_inf, x_drop_cost, batch_inf, x_drop_cost], -1)
            diag = torch.cat([cost_pad, diag], dim=1)

            # fill in 0th row of path matrix with the right pointers
            left_pointer = torch.stack(
                [torch.ones(4) * (effective_d - 1), torch.zeros(4), torch.arange(4)], dim=-1
            )  # [4, 3]
            left_pointer = (
                left_pointer[None, None, ...].repeat([diag.size(0), 1, 1, 1]).to(dev).to(dtype)
            )  # [B, 1, 4, 3]
            path = torch.cat([left_pointer, path], 1)
        if d < K - 1:
            # fill in 0th col of cost matrix with [inf, inf, z_drop_cost, z_drop_cost]
            z_drop_cost = all_cum_z_drop_costs[:, [d + 1]]
            pad = torch.stack([batch_inf, batch_inf, z_drop_cost, z_drop_cost], -1)
            diag = torch.cat([diag, pad], dim=1)

            # fill in 0th col of path matrix with the right pointers

            # the number of elements in the prev diagonal. Refers to 0th element of the column
            last_r_p = diag_p.size(1)
            up_pointer = torch.stack(
                [torch.ones(4) * (effective_d - 1), torch.ones(4) * (last_r_p - 1), torch.arange(4)],
                dim=-1,
            )  # [4, 3]
            up_pointer = up_pointer[None, None, ...].repeat([diag.size(0), 1, 1, 1]).to(dev).to(dtype)  # [B, 1, 4, 3]
            path = torch.cat([path, up_pointer], dim=1)

        all_paths.append(path)

        diag_pp = diag_p
        diag_p = diag

        coord_pp = coord_p
        coord_p = get_diag_coord_grid(diag.size(0), diag.size(1), 4, effective_d).to(dev)

        # process answers
        if (Ds == d).any():
            mask, orig_mask = Ds == d, Ds_orig == d
            original_bs = torch.nonzero(orig_mask, as_tuple=False)[:, 0]
            bs, rs = torch.nonzero(mask, as_tuple=False)[:, 0], Rs[mask]
            min_costs[orig_mask] = min_costs[orig_mask] + list_min([diag[bs, rs]])
            for orig_b, b, r in zip(original_bs, bs, rs):
                # min_costs[orig_b] = min_costs[orig_b] + list_min([diag[b, r]])
                best_pointer = list_min([coord_p[b, r]], keys=[diag[b, r]])
                this_paths = [p[b.item()] for p in all_paths]
                # current_N = Ns[orig_b.item()] + 1
                current_N = N + 1
                tracebacks[orig_b.item()] = diag_traceback(best_pointer, current_N, this_paths)[1]

            # filtering out already processed elements
            diag, diag_p, diag_pp, coord_p, coord_pp, path, Ds, Rs, flipped_costs = [
                t[~mask] for t in [diag, diag_p, diag_pp, coord_p, coord_pp, path, Ds, Rs, flipped_costs]
            ]
            all_x_drop_costs, all_z_drop_costs, all_cum_x_drop_costs, all_cum_z_drop_costs, batch_inf = [
                t[~mask]
                for t in [all_x_drop_costs, all_z_drop_costs, all_cum_x_drop_costs, all_cum_z_drop_costs, batch_inf]
            ]
            all_paths = [p[~mask] for p in all_paths]

            if torch.numel(Ds) == 0:
                break

    return min_costs, tracebacks


def batch_NW_machine(zx_costs_list, x_drop_costs_list, z_drop_costs_list):
    # many_to_one is the same as not exclusive, i.e. multiple z match to one x
    # one_to_many was always true by default before, i.e. multiple x match to one z
    dev, dtype = zx_costs_list[0].device, zx_costs_list[0].dtype
    inf = torch.tensor([9999999999], device=dev, dtype=dtype)
    B = len(zx_costs_list)

    shapes = [t.shape for t in zx_costs_list]
    Ks, Ns = [s[0] for s in shapes], [s[1] for s in shapes]
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
    diag_pp = torch.zeros([B, 1, 1], device=dev)  # diag at i-2
    x1_dropcost, z1_dropcost = all_cum_x_drop_costs[:, [0]], all_cum_z_drop_costs[:, [0]]
    diag_p_row = x1_dropcost[..., None]
    diag_p_col = z1_dropcost[..., None]
    diag_p = torch.cat([diag_p_row, diag_p_col], 1)  # diag at i-1

    # The path is also a diagonal representation that carries the optimal pathlength to each point
    path_pp = torch.zeros([B, 1, 1, 3], device=dev, dtype=int)
    path_p = torch.zeros([B, 2, 1, 3], device=dev, dtype=int)
    all_paths = [path_pp, path_p]  # going to store all the intermediate paths diagonals for the backtrack

    # Coords is also a diagonal representation that carries the current coordinates in [d, r] for each point
    # the last dimension is 3 because it's [d, r, s], where d is a diagonal, r is element's order in the diagonal
    # and s is statet (one of the 4)
    coord_pp = get_diag_coord_grid(B, 1, 1, 0).to(dev)
    coord_p = get_diag_coord_grid(B, 2, 1, 1).to(dev)

    min_costs = torch.zeros(B).to(dtype=dtype).to(device=dev)  # for storing the solution for each element
    tracebacks = [None for _ in range(B)]  # going to store all the intermediate paths diagonals for the backtrack

    for d in range(K + N - 1):
        size = diag_p.size(1) - 1
        pp_start = 0 if d < N else 1
        neigh_up, neigh_left, neigh_diag = diag_p[:, :-1], diag_p[:, 1:], diag_pp[:, pp_start : (pp_start + size)]

        coord_up, coord_left, coord_diag = (
            coord_p[:, :-1].clone(),
            coord_p[:, 1:].clone(),
            coord_pp[:, pp_start : (pp_start + size)].clone(),
        )
        # assign the right state to coordinates
        coord_diag[..., 2] = 0
        coord_left[..., 2] = 1
        coord_up[..., 2] = 2

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
        neighbors = [
            neigh_diag + match_costs_diag[..., None],
            neigh_left + x_drop_costs_diag[..., None],
            neigh_up + z_drop_costs_diag[..., None],
        ]
        coordinates = [coord_diag, coord_left, coord_up]
        diag = list_min(neighbors)[..., None]
        path = (list_min(coordinates, keys=neighbors))[..., None, :]

        # Haven't done below
        # add the initialization values on the ends of diagonal if needed
        effective_d = d + 2  # effective count of d is actually d + 2, since started with 2
        if d < N - 1:
            # fill in 0th row of cost matrix with [inf, x_drop_cost, inf, x_drop_cost]
            x_drop_cost = all_cum_x_drop_costs[:, [d + 1]]
            cost_pad = x_drop_cost[..., None]
            diag = torch.cat([cost_pad, diag], dim=1)

            # fill in 0th row of path matrix with the right pointers
            left_pointer = torch.stack(
                [torch.ones(1) * (effective_d - 1), torch.zeros(1), torch.ones(1) * 1], dim=-1
            )  # [1, 3]
            left_pointer = (
                left_pointer[None, None, ...].repeat([diag.size(0), 1, 1, 1]).to(dev).to(dtype)
            )  # [B, 1, 1, 3]
            path = torch.cat([left_pointer, path], 1)
        if d < K - 1:
            # fill in 0th col of cost matrix with [inf, inf, z_drop_cost, z_drop_cost]
            z_drop_cost = all_cum_z_drop_costs[:, [d + 1]]
            pad = z_drop_cost[..., None]
            diag = torch.cat([diag, pad], dim=1)

            # fill in 0th col of path matrix with the right pointers

            # the number of elements in the prev diagonal. Refers to 0th element of the column
            last_r_p = diag_p.size(1)
            up_pointer = torch.stack(
                [torch.ones(1) * (effective_d - 1), torch.ones(1) * (last_r_p - 1), torch.ones(1) * 2],
                dim=-1,
            )  # [1, 3]
            up_pointer = up_pointer[None, None, ...].repeat([diag.size(0), 1, 1, 1]).to(dev).to(dtype)  # [B, 1, 1, 3]
            path = torch.cat([path, up_pointer], dim=1)

        all_paths.append(path)

        diag_pp = diag_p
        diag_p = diag

        coord_pp = coord_p
        coord_p = get_diag_coord_grid(diag.size(0), diag.size(1), 1, effective_d).to(dev)

        # process answers
        if (Ds == d).any():
            mask, orig_mask = Ds == d, Ds_orig == d
            original_bs = torch.nonzero(orig_mask, as_tuple=False)[:, 0]
            bs, rs = torch.nonzero(mask, as_tuple=False)[:, 0], Rs[mask]
            min_costs[orig_mask] = min_costs[orig_mask] + list_min([diag[bs, rs]])
            for orig_b, b, r in zip(original_bs, bs, rs):
                this_paths = [p[b.item()] for p in all_paths]
                current_N = N + 1
                dc, rc, _ = coord_p[b, r][0]
                tracebacks[orig_b.item()] = nw_diag_traceback(dc, rc, current_N, this_paths)[1]

            # filtering out already processed elements
            diag, diag_p, diag_pp, coord_p, coord_pp, path, Ds, Rs, flipped_costs = [
                t[~mask] for t in [diag, diag_p, diag_pp, coord_p, coord_pp, path, Ds, Rs, flipped_costs]
            ]
            all_x_drop_costs, all_z_drop_costs, all_cum_x_drop_costs, all_cum_z_drop_costs, batch_inf = [
                t[~mask]
                for t in [all_x_drop_costs, all_z_drop_costs, all_cum_x_drop_costs, all_cum_z_drop_costs, batch_inf]
            ]
            all_paths = [p[~mask] for p in all_paths]

            if torch.numel(Ds) == 0:
                break

    return min_costs, tracebacks


def batch_drop_dtw_machine(zx_costs_list, x_drop_costs_list, many_to_one=False, one_to_many=False, contiguous=True):
    # many_to_one is the same as not exclusive, i.e. multiple z match to one x
    # one_to_many was always true by default before, i.e. multiple x match to one z
    dev, dtype = zx_costs_list[0].device, zx_costs_list[0].dtype
    inf = torch.tensor([9999999999], device=dev, dtype=dtype)
    B = len(zx_costs_list)

    shapes = [t.shape for t in zx_costs_list]
    Ks, Ns = [s[0] for s in shapes], [s[1] for s in shapes]
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
    diag_pp = torch.zeros([B, 1, 2], device=dev)  # diag at i-2
    x1_dropcost = all_cum_x_drop_costs[:, [0]]
    diag_p_row = torch.stack([batch_inf, x1_dropcost], -1)
    diag_p_col = torch.stack([batch_inf, batch_inf], -1)
    diag_p = torch.cat([diag_p_row, diag_p_col], 1)  # diag at i-1

    # The path is also a diagonal representation that carries the optimal pathlength to each point
    path_pp = torch.zeros([B, 1, 2, 3], device=dev, dtype=int)
    path_p = torch.zeros([B, 2, 2, 3], device=dev, dtype=int)
    all_paths = [path_pp, path_p]  # going to store all the intermediate paths diagonals for the backtrack

    # Coords is also a diagonal representation that carries the current coordinates in [d, r] for each point
    # the last dimension is 3 because it's [d, r, s], where d is a diagonal, r is element's order in the diagonal
    # and s is statet (one of the 4)
    coord_pp = get_diag_coord_grid(B, 1, 2, 0).to(dev)
    coord_p = get_diag_coord_grid(B, 2, 2, 1).to(dev)

    min_costs = torch.zeros(B).to(dtype=dtype).to(device=dev)  # for storing the solution for each element
    tracebacks = [None for _ in range(B)]  # going to store all the intermediate paths diagonals for the backtrack

    for d in range(K + N - 1):
        size = diag_p.size(1) - 1
        pp_start = 0 if d < N else 1
        neigh_up, neigh_left, neigh_diag = diag_p[:, :-1], diag_p[:, 1:], diag_pp[:, pp_start : (pp_start + size)]
        neigh_up_pos, neigh_left_pos = neigh_up[..., [0]], neigh_left[..., [0]]

        coord_up, coord_left, coord_diag = coord_p[:, :-1], coord_p[:, 1:], coord_pp[:, pp_start : (pp_start + size)]
        coord_up_pos, coord_left_pos = coord_up[..., [0], :], coord_left[..., [0], :]

        # define match and drop cost vectors
        match_costs_diag = torch.stack(
            [torch.flip(torch.diag(flipped_costs[j], d + 1 - K), [-1]) for j in range(flipped_costs.size(0))], 0
        )

        x_d_start, x_d_end = max(d + 1 - K, 0), min(d, N - 1) + 1
        x_drop_costs_diag = torch.flip(all_x_drop_costs[:, x_d_start:x_d_end], [-1])

        # update positive and negative tables -> compute new diagonal

        # DP 0: coming to zx
        pos_neighbors = [neigh_diag]
        pos_coordinates = [coord_diag]
        if one_to_many:
            pos_neighbors.append(neigh_left_pos if contiguous else neigh_left)
            pos_coordinates.append(coord_left_pos if contiguous else coord_left)
        if many_to_one:
            pos_neighbors.append(neigh_up)
            pos_coordinates.append(coord_up)
        diag_pos = list_min(pos_neighbors) + match_costs_diag
        path_pos = list_min(pos_coordinates, keys=pos_neighbors)

        neg_neighbors = [neigh_left]
        neg_coordinates = [coord_left]
        diag_neg = list_min(neg_neighbors) + x_drop_costs_diag
        path_neg = list_min(neg_coordinates, keys=neg_neighbors)

        diag = torch.stack([diag_pos, diag_neg], -1)
        path = torch.stack([path_pos, path_neg], -2)

        # Haven't done below
        # add the initialization values on the ends of diagonal if needed
        effective_d = d + 2  # effective count of d is actually d + 2, since started with 2
        if d < N - 1:
            # fill in 0th row of cost matrix with [inf, x_drop_cost, inf, x_drop_cost]
            x_drop_cost = all_cum_x_drop_costs[:, [d + 1]]
            cost_pad = torch.stack([batch_inf, x_drop_cost], -1)
            diag = torch.cat([cost_pad, diag], dim=1)

            # fill in 0th row of path matrix with the right pointers
            left_pointer = torch.stack(
                [torch.ones(2) * (effective_d - 1), torch.zeros(2), torch.arange(2)], dim=-1
            )  # [2, 3]
            left_pointer = (
                left_pointer[None, None, ...].repeat([diag.size(0), 1, 1, 1]).to(dev).to(dtype)
            )  # [B, 1, 2, 3]
            path = torch.cat([left_pointer, path], 1)
        if d < K - 1:
            # fill in 0th col of cost matrix with [inf, inf, z_drop_cost, z_drop_cost]
            pad = torch.stack([batch_inf, batch_inf], -1)
            diag = torch.cat([diag, pad], dim=1)

            # fill in 0th col of path matrix with the right pointers

            # the number of elements in the prev diagonal. Refers to 0th element of the column
            last_r_p = diag_p.size(1)
            up_pointer = torch.stack(
                [torch.ones(2) * (effective_d - 1), torch.ones(2) * (last_r_p - 1), torch.arange(2)],
                dim=-1,
            )  # [2, 3]
            up_pointer = up_pointer[None, None, ...].repeat([diag.size(0), 1, 1, 1]).to(dev).to(dtype)  # [B, 1, 4, 3]
            path = torch.cat([path, up_pointer], dim=1)

        all_paths.append(path)

        diag_pp = diag_p
        diag_p = diag

        coord_pp = coord_p
        coord_p = get_diag_coord_grid(diag.size(0), diag.size(1), 2, effective_d).to(dev)

        # process answers
        if (Ds == d).any():
            mask, orig_mask = Ds == d, Ds_orig == d
            original_bs = torch.nonzero(orig_mask, as_tuple=False)[:, 0]
            bs, rs = torch.nonzero(mask, as_tuple=False)[:, 0], Rs[mask]
            min_costs[orig_mask] = min_costs[orig_mask] + list_min([diag[bs, rs]])
            for orig_b, b, r in zip(original_bs, bs, rs):
                best_pointer = list_min([coord_p[b, r]], keys=[diag[b, r]])
                this_paths = [p[b.item()] for p in all_paths]
                current_N = N + 1
                tracebacks[orig_b.item()] = diag_traceback(best_pointer, current_N, this_paths)[1]

            # filtering out already processed elements
            diag, diag_p, diag_pp, coord_p, coord_pp, path, Ds, Rs, flipped_costs = [
                t[~mask] for t in [diag, diag_p, diag_pp, coord_p, coord_pp, path, Ds, Rs, flipped_costs]
            ]
            all_x_drop_costs, all_cum_x_drop_costs, batch_inf = [
                t[~mask] for t in [all_x_drop_costs, all_cum_x_drop_costs, batch_inf]
            ]
            all_paths = [p[~mask] for p in all_paths]

            if torch.numel(Ds) == 0:
                break

    return min_costs, tracebacks


def fast_batch_double_drop_dtw_machine(
    zx_costs_list, x_drop_costs_list, z_drop_costs_list, many_to_one=False, one_to_many=False, contiguous=True
):
    # many_to_one is the same as not exclusive, i.e. multiple z match to one x
    # one_to_many was always true by default before, i.e. multiple x match to one z
    dev, dtype = zx_costs_list[0].device, zx_costs_list[0].dtype
    inf = torch.tensor([9999999999], device=dev, dtype=dtype)
    B = len(zx_costs_list)

    shapes = [t.shape for t in zx_costs_list]
    Ks, Ns = [s[0] for s in shapes], [s[1] for s in shapes]
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

    # create routing masks for selection
    # 4x3 corresponds to 4 states (zx, z-, -x, --) and 3 neighbors (l, d, u)
    zx_mask = torch.zeros((4, 3))
    zx_mask[:, 1] = 1
    if one_to_many:
        zx_mask[0, 0] = 1
        if not contiguous:
            zx_mask[1, 0] = 1
    if many_to_one:
        zx_mask[[0, 2], 2] = 1

    z__mask = torch.zeros((4, 3))
    z__mask[[0, 1], 0] = 1

    _x_mask = torch.zeros((4, 3))
    _x_mask[[0, 2], 2] = 1

    ___mask = torch.zeros((4, 3))
    ___mask[[2, 3], 0] = 1
    ___mask[[1, 3], 2] = 1

    mask = torch.stack([zx_mask, z__mask, _x_mask, ___mask], dim=-1).to(dev).to(dtype)  # [4, 3, 4]

    def transition(
        neigh_left, neigh_diag, neigh_up, coord_left, coord_diag, coord_up, match_costs, x_drop_costs, z_drop_costs
    ):
        all_neigh = torch.stack([neigh_left, neigh_diag, neigh_up], dim=-1)  # [B, d, 4, 3]
        all_coords = torch.stack([coord_left, coord_diag, coord_up], dim=-1).permute(
            [0, 1, 3, 2, 4]
        )  # [B, d, 3, 4, 3], the first 3 is the spatial dimension of coordinates
        additions_zx = match_costs[..., None].repeat([1, 1, 3])  # [B, d, 3]
        additions_z_ = x_drop_costs[..., None].repeat([1, 1, 3])
        additions__x = z_drop_costs[..., None].repeat([1, 1, 3])
        additions___ = torch.stack([x_drop_costs, match_costs, z_drop_costs], dim=-1)
        additions = torch.stack([additions_zx, additions_z_, additions__x, additions___], dim=-1)  # [B, d, 3, 4]

        inverse_mask = (~(mask[None, None, ...].to(bool))).to(dtype)
        filtered_costs = all_neigh[..., None] * mask[None, None, ...] + inverse_mask * inf[0]  #  [B, d, 4, 3, 4]
        full_costs = filtered_costs + additions[:, :, None, :, :] * mask[None, None, ...]
        B, d = full_costs.shape[:2]
        the_min = full_costs.reshape([B, d, -1, 4]).min(dim=2)
        new_diag = the_min.values

        all_coords = all_coords[..., None].repeat([1, 1, 1, 1, 1, 4]).reshape([B, d, 3, -1, 4])
        argmins = the_min.indices[:, :, None, None, :].repeat([1, 1, 3, 1, 1])
        pointers = torch.gather(all_coords, index=argmins, dim=-2)
        pointers = pointers[:, :, :, 0, :].permute([0, 1, 3, 2])
        return new_diag, pointers

    # initialize first two contr diagonals
    batch_inf = torch.stack([inf] * B, 0)
    diag_pp = torch.zeros([B, 1, 4], device=dev)  # diag at i-2
    x1_dropcost, z1_dropcost = all_cum_x_drop_costs[:, [0]], all_cum_z_drop_costs[:, [0]]
    diag_p_row = torch.stack([batch_inf, x1_dropcost, batch_inf, x1_dropcost], -1)
    diag_p_col = torch.stack([batch_inf, batch_inf, z1_dropcost, z1_dropcost], -1)
    diag_p = torch.cat([diag_p_row, diag_p_col], 1)  # diag at i-1

    # The path is also a diagonal representation that carries the optimal pathlength to each point
    path_pp = torch.zeros([B, 1, 4, 3], device=dev, dtype=int)
    path_p = torch.zeros([B, 2, 4, 3], device=dev, dtype=int)
    all_paths = [path_pp, path_p]  # going to store all the intermediate paths diagonals for the backtrack

    # Coords is also a diagonal representation that carries the current coordinates in [d, r] for each point
    # the last dimension is 3 because it's [d, r, s], where d is a diagonal, r is element's order in the diagonal
    # and s is statet (one of the 4)
    coord_pp = get_diag_coord_grid(B, 1, 4, 0).to(dev)
    coord_p = get_diag_coord_grid(B, 2, 4, 1).to(dev)

    min_costs = torch.zeros(B).to(dtype=dtype).to(device=dev)  # for storing the solution for each element
    tracebacks = [None for _ in range(B)]  # going to store all the intermediate paths diagonals for the backtrack

    for d in range(K + N - 1):
        size = diag_p.size(1) - 1
        pp_start = 0 if d < N else 1
        neigh_up, neigh_left, neigh_diag = diag_p[:, :-1], diag_p[:, 1:], diag_pp[:, pp_start : (pp_start + size)]
        coord_up, coord_left, coord_diag = coord_p[:, :-1], coord_p[:, 1:], coord_pp[:, pp_start : (pp_start + size)]

        # define match and drop cost vectors
        match_costs_diag = torch.stack(
            [torch.flip(torch.diag(flipped_costs[j], d + 1 - K), [-1]) for j in range(flipped_costs.size(0))], 0
        )

        x_d_start, x_d_end = max(d + 1 - K, 0), min(d, N - 1) + 1
        x_drop_costs_diag = torch.flip(all_x_drop_costs[:, x_d_start:x_d_end], [-1])
        z_d_start, z_d_end = max(d + 1 - N, 0), min(d, K - 1) + 1
        z_drop_costs_diag = all_z_drop_costs[:, z_d_start:z_d_end]

        # update positive and negative tables -> compute new diagonal

        diag, path = transition(
            neigh_left,
            neigh_diag,
            neigh_up,
            coord_left,
            coord_diag,
            coord_up,
            match_costs_diag,
            x_drop_costs_diag,
            z_drop_costs_diag,
        )

        # Haven't done below
        # add the initialization values on the ends of diagonal if needed
        effective_d = d + 2  # effective count of d is actually d + 2, since started with 2
        if d < N - 1:
            # fill in 0th row of cost matrix with [inf, x_drop_cost, inf, x_drop_cost]
            x_drop_cost = all_cum_x_drop_costs[:, [d + 1]]
            cost_pad = torch.stack([batch_inf, x_drop_cost, batch_inf, x_drop_cost], -1)
            diag = torch.cat([cost_pad, diag], dim=1)

            # fill in 0th row of path matrix with the right pointers
            left_pointer = torch.stack(
                [torch.ones(4) * (effective_d - 1), torch.zeros(4), torch.arange(4)], dim=-1
            )  # [4, 3]
            left_pointer = (
                left_pointer[None, None, ...].repeat([diag.size(0), 1, 1, 1]).to(dev).to(dtype)
            )  # [B, 1, 4, 3]
            path = torch.cat([left_pointer, path], 1)
        if d < K - 1:
            # fill in 0th col of cost matrix with [inf, inf, z_drop_cost, z_drop_cost]
            z_drop_cost = all_cum_z_drop_costs[:, [d + 1]]
            pad = torch.stack([batch_inf, batch_inf, z_drop_cost, z_drop_cost], -1)
            diag = torch.cat([diag, pad], dim=1)

            # fill in 0th col of path matrix with the right pointers

            # the number of elements in the prev diagonal. Refers to 0th element of the column
            last_r_p = diag_p.size(1)
            up_pointer = torch.stack(
                [torch.ones(4) * (effective_d - 1), torch.ones(4) * (last_r_p - 1), torch.arange(4)],
                dim=-1,
            )  # [4, 3]
            up_pointer = up_pointer[None, None, ...].repeat([diag.size(0), 1, 1, 1]).to(dev).to(dtype)  # [B, 1, 4, 3]
            path = torch.cat([path, up_pointer], dim=1)

        all_paths.append(path)

        diag_pp = diag_p
        diag_p = diag

        coord_pp = coord_p
        coord_p = get_diag_coord_grid(diag.size(0), diag.size(1), 4, effective_d).to(dev)

        # process answers
        if (Ds == d).any():
            local_mask, orig_mask = Ds == d, Ds_orig == d
            original_bs = torch.nonzero(orig_mask, as_tuple=False)[:, 0]
            bs, rs = torch.nonzero(local_mask, as_tuple=False)[:, 0], Rs[local_mask]
            min_costs[orig_mask] = min_costs[orig_mask] + list_min([diag[bs, rs]])
            for orig_b, b, r in zip(original_bs, bs, rs):
                # min_costs[orig_b] = min_costs[orig_b] + list_min([diag[b, r]])
                best_pointer = list_min([coord_p[b, r]], keys=[diag[b, r]])
                this_paths = [p[b.item()] for p in all_paths]
                # current_N = Ns[orig_b.item()] + 1
                current_N = N + 1
                tracebacks[orig_b.item()] = diag_traceback(best_pointer, current_N, this_paths)[1]

            # filtering out already processed elements
            diag, diag_p, diag_pp, coord_p, coord_pp, path, Ds, Rs, flipped_costs = [
                t[~local_mask] for t in [diag, diag_p, diag_pp, coord_p, coord_pp, path, Ds, Rs, flipped_costs]
            ]
            all_x_drop_costs, all_z_drop_costs, all_cum_x_drop_costs, all_cum_z_drop_costs, batch_inf = [
                t[~local_mask]
                for t in [all_x_drop_costs, all_z_drop_costs, all_cum_x_drop_costs, all_cum_z_drop_costs, batch_inf]
            ]
            all_paths = [p[~local_mask] for p in all_paths]

            if torch.numel(Ds) == 0:
                break

    return min_costs, tracebacks


if __name__ == '__main__':
    zx_costs = np.random.rand(3, 4) # K=3 steps, N=4 clips
    # zx_costs = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
    drop_costs = np.random.rand(4)
    align = drop_dtw(zx_costs, drop_costs)
    #breakpoint()
