# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn.functional as F
from torch import log, exp
import numpy as np
from torch import nn
from scipy.optimize import linear_sum_assignment
# from misc.detr_utils.box_ops import box_cl_to_xy, generalized_box_iou

# For matcher_align
from dp.soft_dp import batch_drop_dtw_machine, batch_double_drop_dtw_machine
from dp.exact_dp import batch_double_drop_dtw_machine as exact_batch_double_drop_dtw_machine
from dp.exact_dp import batch_drop_dtw_machine as exact_batch_drop_dtw_machine
from dp.exact_dp import fast_batch_double_drop_dtw_machine, batch_NW_machine
# from dp.gpu_nw import gpu_nw
from dp.dp_utils import compute_all_costs, compute_double_costs


def compute_sim(z, x, l2_norm):
    if l2_norm:
        return F.normalize(z, dim=1) @ F.normalize(x, dim=1).T
    else:
        return z @ x.T

class DTWMatcher(nn.Module):
    '''
    Drop_z: if True, then we drop both the x axis (query) and z axis (text)
    One_to_many: multiple x match to one z
    Many_to_one: multiple z match to one x 
    '''
    def __init__(self,
                keep_percentile,
                top_band_size=0,
                given_droplines=None,
                drop_z=False,
                one_to_many=False,
                many_to_one=False,
                contiguous=False):
        super().__init__()
        self.keep_percentile = keep_percentile
        self.top_band_size = top_band_size
        self.given_droplines = given_droplines
        self.drop_z = drop_z
        self.one_to_many = one_to_many
        self.many_to_one = many_to_one
        self.contiguous = contiguous

    def forward(self, text_embed, event_embed):
        # computing alignments (without gradients)
        orig_device = event_embed.device
        # embarisingly, this is faster on CPU than on GPU!
        sims = compute_sim(text_embed, event_embed, l2_norm=True)
        #sims = [s.cpu() for s in sims]
        sims = [sims.cpu()]
        self.given_droplines = None if self.given_droplines is None else [s.cpu() for s in self.given_droplines]
        with torch.no_grad():
            zx_costs_list = []
            x_drop_costs_list = []
            z_drop_costs_list = []
            for i, sim in enumerate(sims):
                # computing the baseline logit
                top_sim = sim
                if self.given_droplines is None:
                    if self.top_band_size > 0 and self.top_band_size < sim.shape[1]:
                        top_sim = sim.topk(self.top_band_size, dim=1).values

                    if self.keep_percentile > 1:
                        dropline = top_sim.min() - 5
                    else:
                        k = max([1, int(torch.numel(top_sim) * self.keep_percentile)])
                        dropline = torch.topk(top_sim.reshape([-1]), k).values[-1].detach()
                else:
                    dropline = self.given_droplines[i]

                # shift the costs by the drop logits, so I can set drop costs to 0 instead
                zx_costs_list.append(dropline.reshape([1, 1]) - sim)
                z_drop_cost = torch.zeros([sim.size(0)]).to(sim.device)
                x_drop_cost = torch.zeros([sim.size(1)]).to(sim.device)
                z_drop_costs_list.append(z_drop_cost)
                x_drop_costs_list.append(x_drop_cost)

            # TODO figure out if one_to_many and many_to_one should be on
            align_paths, corresp_mats = None, None
            if self.drop_z:
                if not (self.one_to_many or self.many_to_one):
                    _, align_paths = batch_NW_machine(zx_costs_list, x_drop_costs_list, z_drop_costs_list)
                    # corresp_mats = gpu_nw(zx_costs_list, x_drop_costs_list, z_drop_costs_list)
                else:
                    _, align_paths = exact_batch_double_drop_dtw_machine(
                        # _, align_paths = fast_batch_double_drop_dtw_machine(
                        zx_costs_list,
                        x_drop_costs_list,
                        z_drop_costs_list,
                        one_to_many=self.one_to_many,
                        many_to_one=self.many_to_one,
                        contiguous=self.contiguous,
                    )
            else:
                _, align_paths = exact_batch_drop_dtw_machine(
                    zx_costs_list,
                    x_drop_costs_list,
                    one_to_many=self.one_to_many,
                    many_to_one=self.many_to_one,
                    contiguous=self.contiguous,
                )

            if corresp_mats is None:
                corresp_matrices = []
                for b_id, sim in enumerate(sims):
                    corresp_matrix = torch.zeros_like(sim)
                    for i, j, s in align_paths[b_id]:
                        if s == 0:
                            corresp_matrix[i - 1, j - 1] = 1
                    corresp_matrices.append(corresp_matrix.to(orig_device))
                    # corresp_matrices.append(corresp_matrix)
            text_indices = torch.stack([(torch.as_tensor(i-1, dtype=torch.int64)) for i, _, k in align_paths[-1] if k == 0])
            query_indices = torch.stack([(torch.as_tensor(j-1, dtype=torch.int64)) for _, j, k in align_paths[-1] if k == 0])
            text_indices, rearrange = torch.sort(text_indices)
            query_indices = query_indices[rearrange]
            indices = [(query_indices, text_indices)]
        #return align_paths, corresp_matrices
        return indices, _

def build_matcher(args):
    return DTWMatcher(keep_percentile=args.align_keep_percentile,
                    top_band_size=args.align_top_band_size,
                    given_droplines=None,
                    drop_z=args.align_drop_z,
                    one_to_many=args.align_one_to_many,
                    many_to_one=args.align_many_to_one,
                    contiguous=args.align_contiguous)


def build_matcher_simple():
    return DTWMatcher(keep_percentile=0.5)

if __name__ == '__main__':
    text_embed = torch.rand(5, 128)
    event_embed = torch.rand(15, 128)
    #sim = torch.eye(3, 4)
    aligner = build_matcher_simple()
    indices, matrices = aligner(text_embed, event_embed)
    breakpoint()
