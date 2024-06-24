import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.nn import CrossEntropyLoss
import math

def compute_cls_loss(pred, labels, use_cosface=False):
    if use_cosface:
        # CosFace Loss
        s, m = 30.0, 0.4
        cos_value = torch.diagonal(pred.transpose(0, 1)[labels])
        numerator = s * (cos_value - m)
        excl = torch.cat([torch.cat((pred[i, :y], pred[i, y + 1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(s * excl), dim=1)
        L = numerator - torch.log(denominator)
        loss = -torch.mean(L)
    else:
        # Softmax Loss
        criterion = CrossEntropyLoss().cuda()
        loss = criterion(pred, labels)

    return loss


def frame_blank_align_loss(seq_features1, seq_features2, step_num):
    seq_features1 = seq_features1[:, 1:]
    blank2 = seq_features2[:, :1]
    seq_features2 = seq_features2[:, 1:]
    (B, T, C), device = seq_features1.shape, seq_features1.device
    
    K = 2 * step_num + 1
    sparse_seq_features2 = torch.cat((blank2, seq_features2[:, [5, 7, 8, 9, 11, 12, 13, 14], :]), dim=1)
    pred = (torch.einsum('bic,bjc->bij', seq_features1, sparse_seq_features2) / math.sqrt(C)).log_softmax(-1)

    D_pre = torch.full((B, K), fill_value=float('-99999999'), device=device)
    D_pre[:, 0] = pred[:, 0, 0]
    D_pre[:, 1] = pred[:, 0, 1]
    
    for t in range(1, T):
        D_cur = torch.full((B, K), fill_value=float('-99999999'), device=device)
        D_cur[:, 0] = D_pre[:, 0] + pred[:, t, 0]
        D_cur[:, 1] = torch.logsumexp(torch.stack([D_pre[:, 0], D_pre[:, 1]]), dim=0) + pred[:, t, 1]
        
        # blank term
        blank_pre_ind = torch.arange(1, K, 2)[None, :].repeat(B, 1)
        blank_pre = D_pre[torch.arange(B, device=device).unsqueeze(-1), blank_pre_ind]
        
        blank_cur_ind = torch.arange(2, K, 2)[None, :].repeat(B, 1)
        blank_cur = D_pre[torch.arange(B, device=device).unsqueeze(-1), blank_cur_ind]
        
        blank_log_prob = torch.logsumexp(torch.stack([blank_pre, blank_cur]), dim=0)
        D_cur[:, 2:][:, ::2] = blank_log_prob + pred[:, t, 0][:, None].repeat(1, blank_log_prob.shape[-1])
        
        # step term
        step_prepre_ind = torch.arange(1, K, 2)[None, :-1].repeat(B, 1)
        step_prepre = D_pre[torch.arange(B, device=device).unsqueeze(-1), step_prepre_ind]
        
        step_pre_ind = torch.arange(2, K, 2)[None, :-1].repeat(B, 1)
        step_pre = D_pre[torch.arange(B, device=device).unsqueeze(-1), step_pre_ind]
        
        step_cur_ind = torch.arange(3, K, 2)[None, :].repeat(B, 1)
        step_cur = D_pre[torch.arange(B, device=device).unsqueeze(-1), step_cur_ind]
        
        step_log_prob = torch.logsumexp(torch.stack([step_prepre, step_pre, step_cur]), dim=0)
        D_cur[:, 2:][:, 1::2] = step_log_prob + pred[:, t, 2:]
        D_pre = D_cur

    fsa_distance = -torch.logsumexp(D_cur[:, -2:], dim=-1) / 13
    loss = fsa_distance.mean(0)
    
    return loss


def consist_step_mining(seq_features1, seq_features2, step_num):
    (B, T, C), device = seq_features1.shape, seq_features1.device
    
    pred = (torch.einsum('bic,bjc->bij', seq_features1, seq_features2) / math.sqrt(C)).softmax(-1)
    # pred = torch.cosine_similarity(seq_features1.unsqueeze(2), seq_features2.unsqueeze(1), dim=-1)
    pred = pred.cumsum(-2).cumsum(-1)
    
    D = torch.zeros((B, T, T, T), device=device)
    D_ind = torch.zeros((B, T, T, T), dtype=torch.long, device=device)
    
    D[:, 0] = pred / torch.ones_like(pred).cumsum(-2).cumsum(-1)
    
    area = torch.ones_like(pred).cumsum(-2).cumsum(-1)
    area = (area[:, :, :, None, None] - area[:, :, None, None, :] - area.transpose(1,2)[:, None, :, :, None] + area[:, None, None, :, :])
    block_mat = (pred[:, :, :, None, None] - pred[:, :, None, None, :] - pred.transpose(1,2)[:, None, :, :, None] + pred[:, None, None, :, :])
    
    top, left, bottom, right = torch.meshgrid(*[torch.arange(T, device=device)]*4)
    area = area.clamp_min(1).sqrt()

    block_mat = block_mat.masked_fill(((bottom >= top) | (right >= left)).unsqueeze(0), float('-inf')) / area
    
    for k in range(1, T):
        tmp = ((D[:, k-1, None, None, :, :] * k) + block_mat) / (k+1)
        D[:, k] = torch.max(tmp.flatten(3), -1).values
        D_ind[:, k] = torch.max(tmp.flatten(3), -1).indices
    
    segment1, segment2 = [torch.full((B, 1), T, dtype=torch.long, device=device)]*2
    k = step_num - 1
    i, j, a, b = [torch.full((B, 1), T-1, dtype=torch.long, device=device)]*4
    
    while k >= 0:
        ind = D_ind[range(B), k, i.squeeze(), j.squeeze()][:, None]
        a = ind // T
        b = ind % T
        segment1 = torch.cat([a, segment1], dim=-1)
        segment2 = torch.cat([b, segment2], dim=-1)
        i, j, k = a, b, k-1
    
    repeat_times1 = (segment1[:, 1:] - segment1[:, :-1]).flatten()
    repeat_target1 = torch.arange(step_num, device=device).repeat((B, ))
    step_index1 = repeat_target1.repeat_interleave(repeat_times1).reshape(B, T)
    
    repeat_times2 = (segment2[:, 1:] - segment2[:, :-1]).flatten()
    repeat_target2 = torch.arange(step_num, device=device).repeat((B, ))
    step_index2 = repeat_target2.repeat_interleave(repeat_times2).reshape(B, T)
    
    div_term = torch.exp(torch.arange(0, C, 2, device=device) * -(math.log(10000.0) / C))
    
    pos_emb1 = torch.zeros(B, T, C, device=device)
    pos_emb1[:, :, 0::2] = torch.sin(step_index1.unsqueeze(-1) * div_term)
    pos_emb1[:, :, 1::2] = torch.cos(step_index1.unsqueeze(-1) * div_term)
    
    pos_emb2 = torch.zeros(B, T, C, device=device)
    pos_emb2[:, :, 0::2] = torch.sin(step_index2.unsqueeze(-1) * div_term)
    pos_emb2[:, :, 1::2] = torch.cos(step_index2.unsqueeze(-1) * div_term)
    
    return pos_emb1, pos_emb2, segment1[:, :-1]+1, segment2[:, :-1]+1



def consist_step_mining_train(seq_features1, seq_features2, step_num, pair_labels):
    # seq_features1 = seq_features1[:, 1:]
    # seq_features2 = seq_features2[:, 1:]
    (B, T, C), device = seq_features1.shape, seq_features1.device
    
    pred = (torch.einsum('bic,bjc->bij', seq_features1, seq_features2) / math.sqrt(C)).softmax(-1)
    pred = pred.cumsum(-2).cumsum(-1)
    
    D = torch.zeros((B, T, T, T), device=device)
    D_ind = torch.zeros((B, T, T, T), dtype=torch.long, device=device)
    
    D[:, 0] = pred / torch.ones_like(pred).cumsum(-2).cumsum(-1)
    
    area = torch.ones_like(pred).cumsum(-2).cumsum(-1)
    area = (area[:, :, :, None, None] - area[:, :, None, None, :] \
            - area.transpose(1,2)[:, None, :, :, None] + area[:, None, None, :, :])
    
    block_mat = (pred[:, :, :, None, None] - pred[:, :, None, None, :] \
                - pred.transpose(1,2)[:, None, :, :, None] + pred[:, None, None, :, :])
    
    top, left, bottom, right = torch.meshgrid(*[torch.arange(T, device=device)]*4)
    area = area.clamp_min(1)

    block_mat = block_mat.masked_fill(((bottom >= top) | (right >= left)).unsqueeze(0), float('-inf')) / area

    for k in range(1, T):
        tmp = D[:, k-1, None, None, :, :] + block_mat
        D[:, k] = tmp.flatten(3).max(-1).values
        D_ind[:, k] = tmp.flatten(3).max(-1).indices
    
    segment1, segment2 = [torch.full((B, 1), T, dtype=torch.long, device=device)]*2
    k = step_num
    i, j, a, b = [torch.full((B, 1), T-1, dtype=torch.long, device=device)]*4
    
    while k > 0:
        ind = D_ind[range(B), k, i.squeeze(), j.squeeze()][:, None]
        a = ind // T
        b = ind % T
        segment1 = torch.cat([a, segment1], dim=-1)
        segment2 = torch.cat([b, segment2], dim=-1)
        i, j, k = a, b, k-1
        
    final_result = D[:, :, T-1, T-1]
    
    video_seg1 = segment1[:, :-1] + 1
    video_seg2 = segment2[:, :-1] + 1
    
    # loss_step = (-(pair_labels * final_result.max(dim=-1).values)).sum()
    loss_step = -(pair_labels * final_result.max(dim=-1).values).mean()
    
    return loss_step, video_seg1, video_seg2



def consist_step_mining_inference(seq_features1, seq_features2, step_num):
    seq_features1 = seq_features1[:, 1:]
    seq_features2 = seq_features2[:, 1:]
    (B, T, C), device = seq_features1.shape, seq_features1.device
    
    # pred = (torch.einsum('bic,bjc->bij', seq_features1, seq_features2) / math.sqrt(C)).softmax(-1)
    pred = torch.cosine_similarity(seq_features1.unsqueeze(2), seq_features2.unsqueeze(1), dim=-1)
    pred = pred.cumsum(-2).cumsum(-1)
    
    D = torch.zeros((B, T, T, T), device=device)
    D_ind = torch.zeros((B, T, T, T), dtype=torch.long, device=device)
    
    D[:, 0] = pred / torch.ones_like(pred).cumsum(-2).cumsum(-1)
    
    area = torch.ones_like(pred).cumsum(-2).cumsum(-1)
    area = (area[:, :, :, None, None] - area[:, :, None, None, :] \
            - area.transpose(1,2)[:, None, :, :, None] + area[:, None, None, :, :])
    
    block_mat = (pred[:, :, :, None, None] - pred[:, :, None, None, :] \
                - pred.transpose(1,2)[:, None, :, :, None] + pred[:, None, None, :, :])
    
    top, left, bottom, right = torch.meshgrid(*[torch.arange(T, device=device)]*4)
    area = area.clamp_min(1).sqrt()

    block_mat = block_mat.masked_fill(((bottom >= top) | (right >= left)).unsqueeze(0), float('-inf')) / area
    
    for k in range(1, T):
        tmp = ((D[:, k-1, None, None, :, :] * k) + block_mat) / (k+1)
        D[:, k] = torch.max(tmp.flatten(3), -1).values
        D_ind[:, k] = torch.max(tmp.flatten(3), -1).indices
    
    segment1, segment2 = [torch.full((B, 1), T, dtype=torch.long, device=device)]*2
    k = step_num
    i, j, a, b = [torch.full((B, 1), T-1, dtype=torch.long, device=device)]*4
    
    while k > 0:
        ind = D_ind[range(B), k, i.squeeze(), j.squeeze()][:, None]
        a = ind // T
        b = ind % T
        segment1 = torch.cat([a, segment1], dim=-1)
        segment2 = torch.cat([b, segment2], dim=-1)
        i, j, k = a, b, k-1
        
    return segment1[:, :-1] + 1, segment2[:, :-1] + 1


def step_align_loss(seq_features1, seq_features2):
    B, T, C = seq_features1.shape
    # the similarity matrix: 16 * 16
    pred = (torch.einsum('bic,bjc->bij', seq_features1, seq_features2) / math.sqrt(C)).softmax(-1)
    # pred = torch.cosine_similarity(seq_features1.unsqueeze(2), seq_features2.unsqueeze(1), dim=-1)
    pred = pred.cumsum(-2).cumsum(-1)
    
    D = torch.zeros((B, T, T, T), device=seq_features1.device)
    D_ind = torch.zeros((B, T, T, T), dtype=torch.long, device=pred.device)
    
    D[:, 0] = pred / torch.ones_like(pred).cumsum(-2).cumsum(-1)
    
    area = torch.ones_like(pred).cumsum(-2).cumsum(-1)
    area = (area[:, :, :, None, None] - area[:, :, None, None, :] - area.transpose(1,2)[:, None, :, :, None] + area[:, None, None, :, :])
    block_mat = (pred[:, :, :, None, None] - pred[:, :, None, None, :] - pred.transpose(1,2)[:, None, :, :, None] + pred[:, None, None, :, :])
    
    i, j, a, b = torch.meshgrid(*[torch.arange(T, device=seq_features1.device)]*4)
    area = area.clamp_min(1).sqrt()

    block_mat = block_mat.masked_fill(((a >= i) | (b >= j)).unsqueeze(0), float('-inf')) / area
    
    for k in range(1, T):
        # tmp = ((D[:, k-1, None, None, :, :] * k) + block_mat) / (k+1)
        tmp = D[:, k-1, None, None, :, :] + block_mat
        D[:, k] = torch.max(tmp.flatten(3), -1).values
        D_ind[:, k] = torch.max(tmp.flatten(3), -1).indices
    
    final_result = D[:, :, T-1, T-1]
    return -(final_result.max(dim=-1).values).mean(), final_result.max(dim=-1).indices, D_ind


def single_align_loss(seq_features1, seq_features2):
    device = seq_features1.device
    T, C = seq_features1.shape
    pred = (torch.einsum('ic,jc->ij', seq_features1, seq_features2) / math.sqrt(C)).log_softmax(-1)
    
    ZERO_PAD = torch.zeros((1), device=device)
    ONE_PAD = torch.ones((1), device=device)
    S = seq_features2.shape[0]

    target = (torch.arange(S, device=device))
    
    D_TABLE = ONE_PAD.log()
    for t in range(T):
        D_VEC_1 = torch.logsumexp(torch.stack([D_TABLE[1:t+1], D_TABLE[:-1][:t]]), 0) + pred[t, target[:t]]
        D_VEC_2 = D_TABLE[t:t+1] + pred[t, target[t:t+1]]
        D_TABLE = torch.cat([ZERO_PAD.log(), D_VEC_1, D_VEC_2], dim=-1)
    # changed by hotel: remove " / s"
    ctc_distance = -D_TABLE[S]
    return ctc_distance


def frame2varstep_loss(seq_features1, seq_features2, video_seg):
    B, T, C = seq_features1.shape
    losses = []
    for batch in range(B):
        seq_feature1 = seq_features1[batch]
        
        cur_seg = video_seg[batch]
        cur_seg = cur_seg[:-1] + 1
        sparse_feature2 = seq_features2[batch, cur_seg, :]
        frame_loss = single_align_loss(seq_feature1, sparse_feature2)
        losses.append(frame_loss)
        
    return torch.stack(losses, dim=-1).mean(-1)


def frame2varstep_dist(seq_features1, seq_features2, video_seg):
    B, T, C = seq_features1.shape
    losses = []
    for batch in range(B):
        seq_feature1 = seq_features1[batch]
        
        cur_seg = video_seg[batch]
        cur_seg = cur_seg[:-1] + 1
        sparse_feature2 = seq_features2[batch, cur_seg, :]
        frame_loss = single_align_loss(seq_feature1, sparse_feature2)
        losses.append(frame_loss)
        
    return torch.stack(losses, dim=-1)


def frame2learnedstep_dist(frame_feats1, step_feats2):
    B, T, C = frame_feats1.shape
    losses = []
    for batch in range(B):
        frame_feat1 = frame_feats1[batch]
        step_feat2 = step_feats2[batch]
        # step_feat2 = step_feat2[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
        frame_loss = single_align_loss(frame_feat1, step_feat2)
        losses.append(frame_loss)
        
    return torch.stack(losses, dim=-1)
