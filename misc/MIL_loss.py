import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.losses import accuracy
from mmdet.models.losses.cross_entropy_loss import _expand_onehot_labels
from .utils import weight_reduce_loss


class MILLoss(nn.Module):

    def __init__(self,
                 # use_binary=True,
                 # reduction='mean',
                 binary_ins=False,
                 loss_weight=1.0, eps=1e-6, loss_type='gfocal_loss'):
        """
        Args:
            use_binary (bool, optional): Whether to the prediction is
                used for binary cross entopy
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
        """
        super(MILLoss, self).__init__()
        # self.use_binary = use_binary
        # self.reduction = reduction
        self.loss_weight = loss_weight
        # if self.use_sigmoid:
        # self.loss_cls = CrossEntropyLoss(use_sigmoid=True, loss_weight=loss_weight)
        self.eps = eps
        self.loss_type = loss_type
        self.binary_ins = binary_ins

    def gfocal_loss(self, p, q, w=1.0):
        l1 = (p - q) ** 2
        l2 = q * (p + self.eps).log() + (1 - q) * (1 - p + self.eps).log()
        return -(l1 * l2 * w).sum(dim=-1)

    def forward(self, bag_cls_prob, bag_ins_outs, labels, valid, weight=None):
        """
            bag_cls_outs: (B, N, C),
            bag_ins_outs: (B, N, C*2/C)
            valid: (B, N, 1/C)
            labels: (B, )
        Returns:
        """
        if self.binary_ins:
            assert bag_ins_outs.shape[-1] / bag_cls_prob.shape[-1] == 2
        else:
            assert bag_ins_outs.shape[-1] == bag_cls_prob.shape[-1]

        B, N, C = bag_cls_prob.shape
        prob_cls = bag_cls_prob.unsqueeze(dim=-1)  # (B, N, C, 1)
        prob_ins = bag_ins_outs.reshape(B, N, C, -1)  # (B, N, C, 2/1)
        prob_ins = prob_ins.softmax(dim=1) * valid.unsqueeze(dim=-1)
        prob_ins = F.normalize(prob_ins, dim=1, p=1)
        prob = (prob_cls * prob_ins).sum(dim=1)
        acc = accuracy(prob[..., 0], labels)

        label_weights = (valid.sum(dim=1) > 0).float()
        labels = _expand_onehot_labels(labels, None, C)[0].float()
        num_sample = max(torch.sum(label_weights.sum(dim=-1) > 0).float().item(), 1.)

        if prob.shape[-1] == 1:
            prob = prob.squeeze(dim=-1)
        elif prob.shape[-1] == 2:  # with binary ins
            pos_prob, neg_prob = prob[..., 0], prob[..., 1]
            prob = torch.cat([pos_prob, neg_prob])
            neg_labels = labels.new_zeros(labels.shape)
            labels = torch.cat([labels, neg_labels])
            label_weights = torch.cat([label_weights, label_weights])

        if self.loss_type == 'gfocal_loss':
            loss = self.gfocal_loss(prob, labels, label_weights)
            if weight is not None:
                # modified by fei ##############################################################3
                weight=weight.squeeze(-1)
        elif self.loss_type == 'binary_cross_entropy':
            # if self.use_sigmoid:
            # method 1:
            # loss = self.loss_cls(
            #     prob,
            #     labels,
            #     label_weights,
            #     avg_factor=avg_factor,
            #     reduction_override=reduction_override)
            # method 2
            prob = prob.clamp(0, 1)
            # modified by fei ##############################################################3
            loss = F.binary_cross_entropy(prob, labels.float(), None, reduction="none")
        else:
            raise ValueError()
        loss = weight_reduce_loss(loss, weight, avg_factor=num_sample) * self.loss_weight
        return loss, acc, num_sample