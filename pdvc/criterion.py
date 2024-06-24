# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import copy
import torch
import torch.nn.functional as F
from torch import nn

from misc.detr_utils import box_ops
from misc.detr_utils.misc import (accuracy, get_world_size,
                         is_dist_avail_and_initialized)

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25, focal_gamma=2, opt={}):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.opt = opt
        self.pseudo_box_aug = opt.pseudo_box_aug
        self.refine_pseudo_box = opt.refine_pseudo_box
        if ('Tasty' in opt.visual_feature_folder[0]) or ('tasty' in opt.visual_feature_folder[0]):
            counter_class_rate  =[0.0, 0.012703673018503175, 0.04915769124551229, 0.06489919911626622, 0.0740127036730185, 0.07346037006351837, 0.08064070698702017,
            0.07069870201601768, 0.07870753935376967, 0.07097486882076774, 0.06766086716376692, 0.0579950289975145, 0.05247169290251312, 0.03783485225075946,
            0.03534935100800884, 0.03203534935100801, 0.026788180060756697, 0.02236951118475559, 0.01988400994200497, 0.016570008285004142, 0.013256006628003313,
            0.00856117094725214, 0.006904170118751726, 0.005523336095001381, 0.004694835680751174, 0.0038663352665009665, 0.0027616680475006906, 0.0027616680475006906,
            0.0016570008285004142, 0.0016570008285004142, 0.0005523336095001381, 0.0008285004142502071, 0.0, 0.00027616680475006904, 0.0, 0.0, 0.00027616680475006904,
            0.0011046672190002762, 0.0, 0.0005523336095001381, 0.0, 0.0, 0.0005523336095001381]
        else:
            counter_class_rate = [0.00000000e+00, 0.00000000e+00, 1.93425917e-01, 4.12129084e-01,
       1.88929963e-01, 7.81296833e-02, 5.09541413e-02, 3.12718553e-02,
       1.84833650e-02, 8.39244680e-03, 6.59406534e-03, 4.49595364e-03,
       2.19802178e-03, 1.79838146e-03, 5.99460486e-04, 4.99550405e-04,
       4.99550405e-04, 1.99820162e-04, 2.99730243e-04, 3.99640324e-04,
       2.99730243e-04, 0.00000000e+00, 1.99820162e-04, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 9.99100809e-05, 9.99100809e-05]
        self.counter_class_rate = torch.tensor(counter_class_rate)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        indices, many2one_indices = indices
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=self.focal_gamma) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}
        pred_count = outputs['pred_count']
        max_length = pred_count.shape[1] - 1
        counter_target = [len(target['boxes']) if len(target['boxes']) < max_length  else max_length for target in targets]
        counter_target = torch.tensor(counter_target, device=src_logits.device, dtype=torch.long)
        counter_target_onehot = torch.zeros_like(pred_count)
        counter_target_onehot.scatter_(1, counter_target.unsqueeze(-1), 1)
        weight = self.counter_class_rate[:max_length + 1].to(src_logits.device)

        counter_loss = cross_entropy_with_gaussian_mask(pred_count, counter_target_onehot, self.opt, weight)
        losses['loss_counter'] = counter_loss

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 2]
           The target boxes are expected in format (center, length), normalized by the image size.
        """
        indices, many2one_indices = indices
        N = len(indices[-1][0])
        assert 'pred_boxes' in outputs
        idx, idx2 = self._get_src_permutation_idx2(indices)
        src_boxes = outputs['pred_boxes'][idx]
        if self.opt.use_pseudo_box and self.training:
            # print('use pseudo box')
            target_boxes = torch.cat([t['boxes_pseudo'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        else:
            # print('use gt box')
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cl_to_xy(src_boxes),
            box_ops.box_cl_to_xy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        # print(src_boxes)
        self_iou = torch.triu(box_ops.box_iou(box_ops.box_cl_to_xy(src_boxes),
                                              box_ops.box_cl_to_xy(src_boxes))[0], diagonal=1)
        sizes = [len(v[0]) for v in indices]
        if sizes == [1]:
            losses['loss_self_iou'] = self_iou
            return losses
        self_iou_split = 0
        for i, c in enumerate(self_iou.split(sizes, -1)):
            cc = c.split(sizes, -2)[i]
            self_iou_split += cc.sum() / (0.5 * (sizes[i]) * (sizes[i]-1))
        has_nan = False if torch.all(~torch.isnan(self_iou_split)) else True  
        has_inf = False if torch.all(torch.isfinite(self_iou_split)) else True
        if has_nan or has_inf:
            breakpoint()
        losses['loss_self_iou'] = self_iou_split

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_src_permutation_idx2(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        src_idx2 = torch.cat([src for (_, src) in indices])
        return (batch_idx, src_idx), src_idx2

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
    


    def get_jittered_box(self, box, box_jitter, box_aug_num=5, mode='random'):
        # breakpoint()
        box = box.unsqueeze(0) # (1,2)
        if mode == 'random':
            scale_c = torch.empty((1000, 1), dtype=box.dtype, device=box.device).uniform_(1-box_jitter, 1+box_jitter)
            scale_d = torch.empty((1000, 1), dtype=box.dtype, device=box.device).uniform_(1-box_jitter, 1+box_jitter)
            scale = torch.cat([scale_c, scale_d], dim=1)
            scale_box = box * scale
            scale_box = scale_box.clamp(min=0., max=1.)
            iou, _ = box_ops.box_iou(box_ops.box_cl_to_xy(scale_box), box_ops.box_cl_to_xy(box))
            keep_idx = torch.where(iou.reshape(-1) > 0.1)[0]
            min_keep_cnt = (box_aug_num-1) if (box_aug_num-1) < keep_idx.numel() else keep_idx.numel()
            box_repeat = box.repeat(box_aug_num, 1)
            box_repeat[:min_keep_cnt] = scale_box[keep_idx[:min_keep_cnt]]
        elif mode == 'random_new':
            scale_c = torch.empty((1000, 1), dtype=box.dtype, device=box.device).uniform_(1-box_jitter, 1+box_jitter)
            scale_d = torch.empty((1000, 1), dtype=box.dtype, device=box.device).uniform_(1-box_jitter, 1+box_jitter)
            scale = torch.cat([scale_c, scale_d], dim=1)
            scale_box = box * scale
            scale_box = scale_box.clamp(min=0., max=1.)
            iou, _ = box_ops.box_iou(box_ops.box_cl_to_xy(scale_box), box_ops.box_cl_to_xy(box))
            keep_idx = torch.where(iou.reshape(-1) > 0.1)[0]
            min_keep_cnt = (box_aug_num-1) if (box_aug_num-1) < keep_idx.numel() else keep_idx.numel()
            box_repeat = box.repeat(box_aug_num, 1)
            box_repeat[:min_keep_cnt] = scale_box[keep_idx[:min_keep_cnt]]
        elif mode == 'uniform':
            ratio_c = box_jitter
            ratio_d = 0.048 / 2
            scale_c = torch.tensor([-ratio_c, -ratio_c/2, -ratio_c/4, ratio_c/4, ratio_c/2, ratio_c])
            scale_d = torch.tensor([-ratio_d, -ratio_d/2, ratio_d/2, ratio_d])
            scale = torch.cartesian_prod(scale_c, scale_d).to(device=box.device)
            breakpoint()
            scale_box = box + scale
            scale_box = scale_box.clamp(min=0., max=1.)
            iou, _ = box_ops.box_iou(box_ops.box_cl_to_xy(scale_box), box_ops.box_cl_to_xy(box))
            keep_idx = torch.where(iou.reshape(-1) > 0.1)[0]
            unkeep_idx = torch.where(iou.reshape(-1) <= 0.1)[0]
            if keep_idx.numel() < (box_aug_num-1):
                box_repeat = box.repeat(box_aug_num, 1)
                box_repeat[:keep_idx.numel()] = scale_box[keep_idx]
                random_indices = torch.randperm(unkeep_idx.size(0))[:(box_aug_num-1-keep_idx.numel())]
                box_repeat[keep_idx.numel():(box_aug_num-1)] = scale_box[unkeep_idx[random_indices]]
            else:
                box_repeat = box.repeat(box_aug_num, 1)
                random_indices = torch.randperm(keep_idx.numel())[:(box_aug_num-1)]
                box_repeat[:box_aug_num-1] = scale_box[keep_idx[random_indices]]
        elif mode == 'uniform_old':
            # Conduct augment using pre-defined ratio
            ratio_c = box_jitter
            ratio_d = box_jitter
            scale_c = torch.linspace(1-ratio_c, 1+ratio_c, 4)
            scale_d = torch.linspace(1-ratio_d, 1+ratio_d, 2)
            scale = torch.cartesian_prod(scale_c, scale_d).to(device=box.device) # 16 augmented boxes in total
            scale_box = box * scale
            scale_box = scale_box.clamp(min=0., max=1.)
            iou, _ = box_ops.box_iou(box_ops.box_cl_to_xy(scale_box), box_ops.box_cl_to_xy(box))
            # keep_idx = torch.where(iou.reshape(-1) > 0.1)[0]
            box_repeat = box.repeat(box_aug_num, 1)
            random_indices = torch.randperm(scale_box.size(0))[:(box_aug_num-1)]
            box_repeat[:(box_aug_num-1)] = scale_box[random_indices]
        elif mode == 'random_range':
            def batch_randomize_boxes(boxes, max_vary_range, num_samples=1):
                # Get the centers and widths from the input boxes
                centers = boxes[:, 0]
                widths = boxes[:, 1]
                # breakpoint()
                # Generate random values for the left and right boundaries for each box

                left_boundaries = centers - (widths / 2) - torch.empty(centers.size(0), num_samples, device=boxes.device).uniform_(0, max_vary_range)
                right_boundaries = centers + (widths / 2) + torch.empty(centers.size(0), num_samples, device=boxes.device).uniform_(0, max_vary_range)

                # Ensure that the boundaries stay within the [0, 1] range
                left_boundaries = left_boundaries.clamp(0, 1)
                right_boundaries = right_boundaries.clamp(0, 1)


                # Calculate the new centers and widths
                new_centers = (left_boundaries + right_boundaries) / 2
                new_widths = right_boundaries - left_boundaries

                # Ensure that the widths are non-negative and revert to the original boxes if needed
                is_negative = new_widths <= 0
                new_widths = torch.where(is_negative, widths, new_widths)
                new_centers = torch.where(is_negative, centers, new_centers)

                # Create and return the new boxes tensor
                new_boxes = torch.stack((new_centers, new_widths), dim=2)
                return new_boxes.squeeze(0)
            box_repeat = batch_randomize_boxes(box, box_jitter, box_aug_num)
            if torch.isnan(box_repeat).any():
                breakpoint()
        elif mode == 'augment_width': # original width is 0.5 \sigma range
            import random
            def augment_boxes_with_scale(boxes, scale, num_augments):
                augmented_boxes = []
                for _ in range(num_augments):
                    center, width = boxes[0]
                    # Generate a random scale factor with a more uniform distribution
                    random_scale = scale ** random.uniform(-1, 1)
                    new_width = width * random_scale
                    if center + new_width / 2 > 1 or center - new_width / 2 < 0:
                        new_width = width
                    augmented_boxes.append([center, new_width])
                augmented_boxes = torch.tensor(augmented_boxes, device=boxes.device)
                return augmented_boxes
            box_repeat = augment_boxes_with_scale(box, box_jitter, box_aug_num)
            # breakpoint()

        else:
            raise NotImplementedError('Not support box augmentation mode: {}'.format(mode))      
        return box_repeat

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, others=None, aug_num=None, aug_ratio=None):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}
        if self.training and self.pseudo_box_aug:
            targets_cp = copy.deepcopy(targets)
            assert self.opt.use_pseudo_box
            for i in range((len(targets_cp))):
                boxes_aug = []
                for j in range(len(targets_cp[i]['labels'])):
                    try: 
                        pseudo_box = targets_cp[i]['boxes_pseudo'][j]
                    except:
                        breakpoint()
                    peseudo_box_aug = self.get_jittered_box(pseudo_box, aug_ratio, aug_num, self.opt.pseudo_box_aug_mode)
                    boxes_aug.append(peseudo_box_aug)
                targets_cp[i]['boxes_pseudo'] = torch.cat(boxes_aug, dim=0)
                targets_cp[i]['labels'] = targets_cp[i]['labels'].unsqueeze(dim=1).repeat(1, aug_num).reshape(-1,)
                targets[i]['box_pseudo_aug'] = torch.cat(boxes_aug, dim=0)
        # Retrieve the matching between the outputs of the last layer and the targets
            last_indices = self.matcher(outputs_without_aux, targets_cp)
        else:
            targets_cp = targets
            last_indices = self.matcher(outputs_without_aux, targets)
        outputs['matched_indices'] = last_indices

        num_boxes = sum(len(t["labels"]) for t in targets_cp)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets_cp, last_indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            aux_indices = []
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets_cp)
                aux_indices.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets_cp, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

            return losses, last_indices, aux_indices
        return losses, last_indices

class AlignCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute DTW assignment between ground truth captions and the outputs object queries
        2) we supervise each pair of matched ground-truth / prediction (supervise class)
    """
    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25, focal_gamma=2, opt={}):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.opt = opt
        counter_class_rate = [0.00000000e+00, 0.00000000e+00, 1.93425917e-01, 4.12129084e-01,
       1.88929963e-01, 7.81296833e-02, 5.09541413e-02, 3.12718553e-02,
       1.84833650e-02, 8.39244680e-03, 6.59406534e-03, 4.49595364e-03,
       2.19802178e-03, 1.79838146e-03, 5.99460486e-04, 4.99550405e-04,
       4.99550405e-04, 1.99820162e-04, 2.99730243e-04, 3.99640324e-04,
       2.99730243e-04, 0.00000000e+00, 1.99820162e-04, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 9.99100809e-05, 9.99100809e-05]
        self.counter_class_rate = torch.tensor(counter_class_rate)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        Compute the classification loss and counter loss
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        indices, many2one_indices = indices
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=self.focal_gamma) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        pred_count = outputs['pred_count']
        max_length = pred_count.shape[1] - 1
        counter_target = [len(target['boxes']) if len(target['boxes']) < max_length  else max_length for target in targets]
        counter_target = torch.tensor(counter_target, device=src_logits.device, dtype=torch.long)
        counter_target_onehot = torch.zeros_like(pred_count)
        counter_target_onehot.scatter_(1, counter_target.unsqueeze(-1), 1)
        weight = self.counter_class_rate[:max_length + 1].to(src_logits.device)
        # breakpoint()
        counter_loss = cross_entropy_with_gaussian_mask(pred_count, counter_target_onehot, self.opt, weight)
        losses['loss_counter'] = counter_loss

        return losses
    
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        # Compute temporal IOU loss among given predicted N temporal boundaries, which encourages the temporal boundaries to be more diverse and no overlap
        # outputs: (bsz, num_query, 2)
        # breakpoint()
        # breakpoint()
        indices, many2one_indices = indices
        idx, idx2 = self._get_src_permutation_idx2(indices)
        src_boxes = outputs['pred_boxes'][idx] # num_boxes, 2
        avg_duration = torch.mean(src_boxes[:, 1])
        center_point = src_boxes[:,0]
        N = len(indices[-1][0])

        losses = {}

        if self.opt.use_pseudo_box and self.training:
            # If generate peseudo ground truth boxes from alignment, use the alignment boxes as the target boxes
            target_boxes = torch.cat([t['boxes_pseudo'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
            losses['loss_bbox'] = loss_bbox.sum() / num_boxes

            loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cl_to_xy(src_boxes),
            box_ops.box_cl_to_xy(target_boxes)))
            losses['loss_giou'] = loss_giou.sum() / num_boxes

        if not self.opt.use_pseudo_box:
            ## Squence Ordering loss
            rank_margin = 0.01
            pairs = torch.combinations(torch.arange(center_point.size(0)), 2)
            rank_dist = center_point[pairs[:, 0]] - center_point[pairs[:, 1]] + rank_margin
            # Make sure that the center points are ordered
            rank_loss = torch.relu(rank_margin + rank_dist).mean()

            losses['loss_ref_rank']  = rank_loss

            ## Self IOU loss
            prior_duration = 0.06
            self_iou = torch.triu(box_ops.box_iou(box_ops.box_cl_to_xy(src_boxes),
                                                box_ops.box_cl_to_xy(src_boxes))[0], diagonal=1)
            sizes = [len(v[0]) for v in indices]
            self_iou_split = 0
            for i, c in enumerate(self_iou.split(sizes, -1)):
                cc = c.split(sizes, -2)[i]
                self_iou_split += cc.sum() / (0.5 * (sizes[i]) * (sizes[i]-1))
            duration_constraint = torch.abs(prior_duration/(avg_duration + 1e-6) - 1)
            self_iou_split += duration_constraint
            
            
            losses['loss_self_iou'] = self_iou_split

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_src_permutation_idx2(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        src_idx2 = torch.cat([src for (_, src) in indices])
        return (batch_idx, src_idx), src_idx2

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'cardinality': self.loss_cardinality,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, others):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        text_embed = others['text_embed'] # num_dec_layers, num_sentence, dim
        event_embed = others['event_embed'] # num_dec_layers, num_query, dim
        dim = event_embed.shape[-1]

        # Retrieve the matching between the outputs of the last layer and the targets
        # if self.opt.matcher_type == 'DTW':
        #     last_indices = self.matcher(text_embed[-1], event_embed[-1].reshape(-1, dim))
        # elif self.opt.matcher_type == 'Sim':
        #     last_indices = self.matcher(outputs, targets, text_embed[-1], event_embed[-1].reshape(-1, dim))
        # else:
        #     raise NotImplementedError('Align Criterion does not support:{}'.format(self.opt.matcher_type))
        #breakpoint()
        last_indices = self.matcher(outputs, targets, text_embed[-1], event_embed[-1].reshape(-1, dim))
        outputs['matched_indices'] = last_indices

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, last_indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            aux_indices = []
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(outputs, targets, text_embed[-1], event_embed[-1].reshape(-1, dim))
                aux_indices.append(indices)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

            return losses, last_indices, aux_indices
        return losses, last_indices

class ContrastiveCriterion(nn.Module):
    '''
    Contrastive loss between event feature and caption feature
    '''

    def __init__(self, temperature=0.1, enable_cross_video_cl=False, enable_e2t_cl=False, enable_bg_for_cl=False):
        super().__init__()
        self.temperature = temperature
        self.enable_cross_video_cl = enable_cross_video_cl
        self.enable_e2t_cl = enable_e2t_cl
        self.enable_bg_for_cl = enable_bg_for_cl

    def forward_logits(self, text_embed, event_embed, bg_embed=None):
        normalized_text_emb = F.normalize(text_embed, p=2, dim=1)
        normalized_event_emb = F.normalize(event_embed, p=2, dim=1)
        logits = torch.mm(normalized_text_emb, normalized_event_emb.t())
        if bg_embed is not None:
            bg_logits = torch.sum(normalized_event_emb * F.normalize(bg_embed, p=2), dim=1)
            logits = torch.cat((logits, bg_logits.unsqueeze(0)), dim=0)
        return logits


    def forward(self, text_embed, event_embed, matching_indices, return_logits=False, bg_embed=None):

        '''
        :param text_embed: [(event_num, contrastive_hidden_size)], len = batch size
                            total_event_number = sum of event number of each item in current batch
        :param event_embed: (bsz, max_event_num, contrastive_hiddent_size), which need to be
                            expand in this function
        :param matching_indices: (bsz, event_num)
        '''
        batch_size, max_event_num, _ = event_embed.shape
        event_embed, text_embed, gt_labels, gt_event_num = self._preprocess(event_embed, [text_embed], matching_indices)
        raw_logits = self.forward_logits(text_embed, event_embed)
        logits = raw_logits / self.temperature

        if self.enable_cross_video_cl:
            t2e_loss = F.cross_entropy(logits, gt_labels)
            if self.enable_e2t_cl:
                gt_label_matrix = torch.zeros(len(text_embed) + 1, len(event_embed), device=text_embed.device)
                gt_label_matrix[torch.arange(len(gt_labels)), gt_labels] = 1
                event_mask = gt_label_matrix.sum(dim=0) == 0
                gt_label_matrix[-1, event_mask] = 1
                e2t_gt_label = gt_label_matrix.max(dim=0)[1]
                bg_logits = torch.sum(F.normalize(event_embed, p=2) * F.normalize(bg_embed, p=2), dim=1)
                e2t_logits = torch.cat((logits, bg_logits.unsqueeze(0) / self.temperature), dim=0)
                if self.enable_bg_for_cl:
                    e2t_loss = F.cross_entropy(e2t_logits.t(), e2t_gt_label)
                else:
                    e2t_loss = F.cross_entropy(e2t_logits.t()[~event_mask], e2t_gt_label[~event_mask])
                loss = 0.5 * (t2e_loss + e2t_loss)
            else:
                loss = t2e_loss
        else:
            loss = 0; base = 0
            for i in range(batch_size):
                current_gt_event_num = gt_event_num[i]
                current_logits = logits[base: base + current_gt_event_num, i * max_event_num: (i + 1) * max_event_num]
                current_gt_labels = gt_labels[base: base + current_gt_event_num]
                t2e_loss = F.cross_entropy(current_logits, current_gt_labels)
                if self.enable_e2t_cl:
                    gt_label_matrix = torch.zeros(gt_event_num[i] + 1, max_event_num, device=text_embed.device)
                    gt_label_matrix[torch.arange(current_gt_labels), current_gt_labels] = 1
                    event_mask = gt_label_matrix.sum(dim=0) == 0
                    e2t_gt_label = gt_label_matrix.max(dim=0)[1]
                    bg_logits = torch.sum(F.normalize(event_embed, p=2) * F.normalize(bg_embed, p=2), dim=1)
                    e2t_logits = torch.cat((current_logits, bg_logits.unsqueeze(0) / self.temperature), dim=0)
                    if self.enable_bg_for_cl:
                        e2t_loss = F.cross_entropy(e2t_logits.t(), e2t_gt_label)
                    else:
                        e2t_loss = F.cross_entropy(e2t_logits.t(), e2t_gt_label, ignore_index=len(text_embed), reduction='sum') / (1e-5 + sum(~event_mask))
                    loss += 0.5 * (t2e_loss + e2t_loss)
                else:
                    loss += t2e_loss
                base += current_gt_event_num
            loss = loss / batch_size
        # pdb.set_trace()
        if return_logits:
            return loss, raw_logits
        return loss


    def _preprocess(self, event_embed, text_embed, matching_indices):
        '''
        Flatten event_embed of a batch, get gt label

        :param matching_indices: [(event_num, )]  len = bsz
        '''
        batch_size, max_event_num, f_dim = event_embed.shape
        gt_labels = []
        text_features = []
        gt_event_num = []
        event_features = event_embed.view(-1, f_dim)
        for i in range(batch_size):
            base = i * max_event_num if self.enable_cross_video_cl else 0
            feat_ids, cap_ids = matching_indices[i]
            gt_event_num.append(len(feat_ids))
            text_features.append(text_embed[i][cap_ids])
            gt_labels.append(feat_ids + base)
        text_features = torch.cat(text_features, dim=0)
        gt_labels = torch.cat(gt_labels, dim=0)
        gt_labels = gt_labels.to(event_embed.device)
            
        return event_features, text_features, gt_labels, gt_event_num

def cross_entropy_with_gaussian_mask(inputs, targets, opt, weight):
    gau_mask = opt.lloss_gau_mask
    beta = opt.lloss_beta

    N_, max_seq_len = targets.shape
    gassian_mu = torch.arange(max_seq_len, device=inputs.device).unsqueeze(0).expand(max_seq_len,
                                                                                     max_seq_len).float()
    x = gassian_mu.transpose(0, 1)
    gassian_sigma = 2
    mask_dict = torch.exp(-(x - gassian_mu) ** 2 / (2 * gassian_sigma ** 2))
    _, ind = targets.max(dim=1)
    mask = mask_dict[ind]

    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none", weight= 1 - weight)
    if gau_mask:
        coef = targets + ((1 - mask) ** beta) * (1 - targets)
    else:
        coef = targets + (1 - targets)
    loss = loss * coef
    loss = loss.mean(1)
    return loss.mean()

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """

    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none") # with_logits func calculates sigmoid and CE jointly
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes

def regression_loss(inputs, targets, opt, weight):
    inputs = F.relu(inputs) + 2
    max_id = torch.argmax(targets, dim=1)
    if opt.regression_loss_type == 'l1':
        loss = nn.L1Loss()(inputs[:, 0], max_id.float())
    elif opt.regression_loss_type == 'l2':
        loss = nn.MSELoss()(inputs[:, 0], max_id.float())
    return loss