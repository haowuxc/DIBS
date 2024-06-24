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
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F
from torch import log, exp
import numpy as np

from misc.detr_utils.box_ops import box_cl_to_xy, generalized_box_iou

# For matcher_align
from pdvc.dp.soft_dp import batch_drop_dtw_machine, batch_double_drop_dtw_machine
from pdvc.dp.exact_dp import batch_double_drop_dtw_machine as exact_batch_double_drop_dtw_machine
from pdvc.dp.exact_dp import batch_drop_dtw_machine as exact_batch_drop_dtw_machine
from pdvc.dp.exact_dp import fast_batch_double_drop_dtw_machine, batch_NW_machine
# from dp.gpu_nw import gpu_nw
from pdvc.dp.dp_utils import compute_all_costs, compute_double_costs


def compute_sim(z, x, l2_norm):
    if l2_norm:
        return F.normalize(z, dim=1) @ F.normalize(x, dim=1).T
    else:
        return z @ x.T

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self,
                 cost_class: float = 1,
                 cost_bbox: float = 1,
                 cost_giou: float = 1,
                 cost_alpha = 0.25,
                 cost_gamma = 2,
                 use_pseudo_box = False):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        # self.cost_caption = cost_caption
        self.cost_alpha = cost_alpha
        self.cost_gamma = cost_gamma
        self.use_pseudo_box = use_pseudo_box

        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0 # or cost_caption!=0, "all costs cant be 0"
        # breakpoint()

    def forward(self, outputs, targets, verbose=False, many_to_one=False):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]
            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

            # Also concat the target labels and boxes
            tgt_ids = torch.cat([v["labels"] for v in targets])
            if self.use_pseudo_box and self.training:
                # print('use pseudo box')
                tgt_bbox = torch.cat([v["boxes_pseudo"] for v in targets])
            else:
                tgt_bbox = torch.cat([v["boxes"] for v in targets])
                # print('use gt box')

            # Compute the classification cost.
            # alpha = 0.25
            alpha = self.cost_alpha
            gamma = self.cost_gamma
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
            # breakpoint()

            # Compute the giou cost betwen boxes
            try:
                cost_giou = -generalized_box_iou(box_cl_to_xy(out_bbox),
                                        box_cl_to_xy(tgt_bbox))
            except:
                print('out_bbox', out_bbox)
                print('tgt_bbox', tgt_bbox)
                breakpoint()

            # cost_caption = outputs['caption_costs'].flatten(0, 1)

            # Final cost matrix
            # breakpoint()
            try: # [100, 10], [100, 11], [100, 10]
                C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            except:
                breakpoint()

            costs = {'cost_bbox': cost_bbox,
                     'cost_class': cost_class,
                     'cost_giou': cost_giou,
                     # 'cost_caption': cost_caption,
                     'out_bbox': out_bbox[:, 0::2]}

            if verbose:
                print('\n')
                print(self.cost_bbox, cost_bbox.var(dim=0), cost_bbox.max(dim=0)[0] - cost_bbox.min(dim=0)[0])
                print(self.cost_class, cost_class.var(dim=0), cost_class.max(dim=0)[0] - cost_class.min(dim=0)[0])
                print(self.cost_giou, cost_giou.var(dim=0), cost_giou.max(dim=0)[0] - cost_giou.min(dim=0)[0])
                # print(self.cost_caption, cost_caption.var(dim=0), cost_caption.max(dim=0)[0] - cost_caption.min(dim=0)[0])

            C = C.view(bs, num_queries, -1).cpu()

        
            sizes = [len(v["boxes_pseudo"]) for v in targets] if self.use_pseudo_box else [len(v["boxes"]) for v in targets]
            # pdb.set_trace()
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            m2o_rate = 4
            rl_indices = [linear_sum_assignment(torch.cat([c[i]]*m2o_rate, -1)) for i, c in enumerate(C.split(sizes, -1))]
            rl_indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j%sizes[ii], dtype=torch.int64)) for ii,(i, j) in
                       enumerate(rl_indices)]

            indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

            if verbose:
                print('------matching results:')
                print(indices)
                for indice in indices:
                    for i, j in zip(*indice):
                        print(out_bbox[i][0::2], tgt_bbox[j][0::2])
                print('-----topK scores:')
                topk_indices = out_prob.topk(10, dim=0)
                print(topk_indices)
                for i,(v,ids) in enumerate(zip(*topk_indices)):
                    print('top {}'.format(i))
                    s= ''
                    for name,cost in costs.items():
                        s += name + ':{} '.format(cost[ids])
                    print(s)

            return indices, rl_indices

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
                drop_z=True,
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

    def forward(self, ouputs, targets, text_embed, event_embed):
        # computing alignments (without gradients)
        orig_device = event_embed[0].device
        # embarisingly, this is faster on CPU than on GPU!
        sims = compute_sim(text_embed, event_embed, l2_norm=True)
        #sims = [s.cpu() for s in sims]
        sims = [sims.cpu()]
        # TODO: Add the classification cost the the alignment cost
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
        return indices, []

class SimMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    based on the similarity bewteen text embedding and query embedding
    """
    def __init__(self,
                 cost_class: float = 1,
                 cost_sim: float = 1,
                 cost_bbox: float = 1,
                 cost_giou: float = 1,
                 cost_alpha = 0.25,
                 cost_gamma = 2,
                 use_pseudo_box = False):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_sim = cost_sim
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        # self.cost_caption = cost_caption
        self.cost_alpha = cost_alpha
        self.cost_gamma = cost_gamma
        self.use_pseudo_box = use_pseudo_box

        assert cost_class != 0 or cost_sim!=0, "all costs cannot be 0"
        # breakpoint()

    def forward(self, outputs, targets, text_embed, event_embed, verbose=False, many_to_one=False):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

            tgt_ids = torch.cat([v["labels"] for v in targets])
            alpha = self.cost_alpha
            gamma = self.cost_gamma
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

            # Also concat the target labels and boxes
            # breakpoint()
            if self.use_pseudo_box:
                tgt_bbox = torch.cat([v["boxes_pseudo"] for v in targets])
                # Compute the L1 cost between boxes
                cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

                # Compute the giou cost betwen boxes
                cost_giou = -generalized_box_iou(box_cl_to_xy(out_bbox),
                                                box_cl_to_xy(tgt_bbox))
            else:
                cost_bbox = torch.zeros_like(cost_class)
                cost_giou = torch.zeros_like(cost_class)

            # Compute the classification cost.
            # alpha = 0.25
            alpha = self.cost_alpha
            gamma = self.cost_gamma
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
            # breakpoint()
            # Compute the similarity cost
            cost_sim = compute_sim(text_embed, event_embed, l2_norm=True).permute(1,0)
            cost_sim = torch.ones_like(cost_sim) - cost_sim
            # breakpoint()

            # cost_caption = outputs['caption_costs'].flatten(0, 1)

            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou + self.cost_sim * cost_sim

            costs = {'cost_bbox': cost_bbox,
                     'cost_class': cost_class,
                     'cost_giou': cost_giou,
                     'cost_sim': cost_sim,
                     # 'cost_caption': cost_caption,
                     'out_bbox': out_bbox[:, 0::2],
                     }

            if verbose:
                print('\n')
                print(self.cost_bbox, cost_bbox.var(dim=0), cost_bbox.max(dim=0)[0] - cost_bbox.min(dim=0)[0])
                print(self.cost_class, cost_class.var(dim=0), cost_class.max(dim=0)[0] - cost_class.min(dim=0)[0])
                print(self.cost_giou, cost_giou.var(dim=0), cost_giou.max(dim=0)[0] - cost_giou.min(dim=0)[0])
                print(self.cost_sim, cost_sim.var(dim=0), cost_sim.max(dim=0)[0] - cost_sim.min(dim=0)[0])
                # print(self.cost_caption, cost_caption.var(dim=0), cost_caption.max(dim=0)[0] - cost_caption.min(dim=0)[0])

            C = C.view(bs, num_queries, -1).cpu()

            sizes = [text_embed.size(0)]
            # pdb.set_trace()
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            m2o_rate = 4
            rl_indices = [linear_sum_assignment(torch.cat([c[i]]*m2o_rate, -1)) for i, c in enumerate(C.split(sizes, -1))]
            rl_indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j%sizes[ii], dtype=torch.int64)) for ii,(i, j) in
                       enumerate(rl_indices)]

            indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


            return indices, rl_indices    

def build_matcher(args):
    if args.matcher_type == 'DTW':
        return DTWMatcher(keep_percentile=args.align_keep_percentile,
                    top_band_size=args.align_top_band_size,
                    given_droplines=None,
                    drop_z=args.align_drop_z,
                    one_to_many=args.align_one_to_many,
                    many_to_one=args.align_many_to_one,
                    contiguous=args.align_contiguous)
    elif args.matcher_type == 'Sim':
        return SimMatcher(cost_class=args.set_cost_class,
                                cost_sim=args.set_cost_sim,
                                cost_bbox=args.set_cost_bbox,
                                cost_giou=args.set_cost_giou,
                                cost_alpha = args.cost_alpha,
                                cost_gamma = args.cost_gamma,
                                use_pseudo_box = args.use_pseudo_box
                                )
    else:
        return HungarianMatcher(cost_class=args.set_cost_class,
                                cost_bbox=args.set_cost_bbox,
                                cost_giou=args.set_cost_giou,
                                cost_alpha = args.cost_alpha,
                                cost_gamma = args.cost_gamma,
                                use_pseudo_box = args.use_pseudo_box
                                )


def build_matcher_simple():
    #return DTWMatcher(keep_percentile=0.5)
    return SimMatcher()

if __name__ == '__main__':
    text_embed = torch.rand(5, 128)
    event_embed = torch.rand(15, 128)
    #sim = torch.eye(3, 4)
    aligner = build_matcher_simple()
    indices, matrices = aligner(text_embed, event_embed)
    breakpoint()