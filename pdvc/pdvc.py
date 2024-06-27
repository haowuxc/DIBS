# ------------------------------------------------------------------------
# PDVC
# ------------------------------------------------------------------------
# Modified from Deformable DETR(https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import json
import torch
import torch.nn.functional as F
from torch import nn
import math
import time

from misc.detr_utils import box_ops
from misc.detr_utils.misc import (inverse_sigmoid)

from .matcher import build_matcher

from .deformable_transformer import build_deforamble_transformer
from pdvc.CaptioningHead import build_captioner
import copy
from .criterion import AlignCriterion, SetCriterion, ContrastiveCriterion
# from .rl_tool import init_scorer
from misc.utils import decide_two_stage
from .base_encoder import build_base_encoder
# from .video_segmentation import segment_video_into_steps, alignment_to_boundary, to_center_duration, align_frame_into_steps
from .video_segmentation import *
# from transformers import AutoModel, BertConfig
# from transformers.models.bert.modeling_bert import BertEncoder
import numpy as np
from itertools import chain
# from .UniVL import load_pretrained_UniVL


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class PDVC(nn.Module):
    """ This is the PDVC module that performs dense video captioning """

    def __init__(self, base_encoder, transformer, captioner, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, opt=None, translator=None):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture. See transformer.py
            captioner: captioning head for generate a sentence for each event queries
            num_classes: number of foreground classes
            num_queries: number of event queries. This is the maximal number of events
                         PDVC can detect in a single video. For ActivityNet Captions, we recommend 10-30 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            opt: all configs
        """
        super().__init__()
        self.opt = opt
        self.base_encoder = base_encoder
        self.transformer = transformer
        self.caption_head = captioner
        num_pred_text = 0

        # if opt.matcher_type == 'DTW' or opt.use_pseudo_box:
        #     self.text_encoder = text_encoder
        #     text_encoder_hidden_dim = self.text_encoder.config.hidden_size
        #     num_pred_text += 1

        hidden_dim = transformer.d_model
        text_hidden_dim = opt.text_hidden_dim
        
        if self.opt.use_anchor:
            # self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
            self.anchor_embed = nn.Embedding(num_queries, 2) # num_queries, 2 (center, duration) 
            self.query_embed = self.transformer.prepare_init_anchor_and_query(self.anchor_embed, hidden_dim, \
                                                                        random_anchor_init=True, prior_anchor_duration_init=True, \
                                                                        prior_duration=0.048)
            self.query_embed = nn.Parameter(self.query_embed, requires_grad=True)
        else:
            self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)

        self.class_head = nn.Linear(hidden_dim, num_classes)
        self.class_refine_head = nn.Linear(hidden_dim, num_classes) # For refine pseudo box if use additional score layer
        self.count_head = nn.Linear(hidden_dim, opt.max_eseq_length + 1)
        self.bbox_head = MLP(hidden_dim, hidden_dim, 2, 3)

        self.num_feature_levels = num_feature_levels
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.share_caption_head = opt.share_caption_head

        # initialization
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_head.bias.data = torch.ones(num_classes) * bias_value
        self.class_refine_head.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_head.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_head.layers[-1].bias.data, 0)

        if self.opt.matcher_type == 'DTW' or self.opt.matcher_type == 'Sim' \
            or self.opt.use_pseudo_box:
            self.load_text_embed = True
        else:
            self.load_text_embed = False


        num_pred = transformer.decoder.num_layers
        if self.share_caption_head:
            print('all decoder layers share the same caption head')
            self.caption_head = nn.ModuleList([self.caption_head for _ in range(num_pred)])
        else:
            print('do NOT share the caption head')
            self.caption_head = _get_clones(self.caption_head, num_pred)

        if self.opt.use_additional_cap_layer:
            self.caption_head_refine = _get_clones(captioner, self.opt.refine_pseudo_stage_num)

        if with_box_refine:
            self.class_head = _get_clones(self.class_head, num_pred)
            self.count_head = _get_clones(self.count_head, num_pred)
            self.bbox_head = _get_clones(self.bbox_head, num_pred)
            nn.init.constant_(self.bbox_head[0].layers[-1].bias.data[1:], -2)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_head = self.bbox_head
        else:
            nn.init.constant_(self.bbox_head.layers[-1].bias.data[1:], -2)
            self.class_head = nn.ModuleList([self.class_head for _ in range(num_pred)])
            self.count_head = nn.ModuleList([self.count_head for _ in range(num_pred)])
            self.bbox_head = nn.ModuleList([self.bbox_head for _ in range(num_pred)])
            self.transformer.decoder.bbox_head = None

        self.class_refine_head = _get_clones(self.class_refine_head, self.opt.refine_pseudo_stage_num)
        # if opt.matcher_type == 'DTW' or opt.use_pseudo_box:
        if opt.disable_contrastive_projection:
            projection_event = nn.Identity()
            projection_text = nn.Identity()
        else:
            projection_event = nn.Linear(hidden_dim, opt.contrastive_hidden_size)
            projection_text = nn.Linear(text_hidden_dim, opt.contrastive_hidden_size)
        self.contrastive_projection_event = nn.ModuleList(
            [projection_event for _ in range(num_pred)])
        self.contrastive_projection_text = nn.ModuleList(
            [projection_text for _ in range(num_pred)])
        if opt.enable_bg_for_cl:
            self.background_embed = nn.Parameter(torch.randn(1, opt.contrastive_hidden_size), requires_grad=True)
        else:
            self.background_embed = None
            

        self.translator = translator

        self.disable_mid_caption_heads = opt.disable_mid_caption_heads
        if self.disable_mid_caption_heads:
            print('only calculate caption loss in the last decoding layer')
        
        self.pseudo_boxes = {}
        

    def get_filter_rule_for_encoder(self):
        filter_rule = lambda x: 'input_proj' in x \
                                or 'transformer.encoder' in x \
                                or 'transformer.level_embed' in x \
                                or 'base_encoder' in x
        return filter_rule

    def encoder_decoder_parameters(self):
        filter_rule = self.get_filter_rule_for_encoder()
        enc_paras = []
        dec_paras = []
        for name, para in self.named_parameters():
            if filter_rule(name):
                print('enc: {}'.format(name))
                enc_paras.append(para)
            else:
                print('dec: {}'.format(name))
                dec_paras.append(para)
        return enc_paras, dec_paras

    # def text_encoding(self, text_encoder_input):
    #     '''
    #     Produce the text embedding for each caption
    #     :param text_encoder_input: a dict of input for text encoder
    #     '''
    #     if self.opt.pretrained_language_model == 'UniVL' or self.opt.use_pseudo_box:
    #         # breakpoint()
    #         dtype = next(self.parameters()).dtype
    #         enable_grad = False
    #         use_amp = False
    #         with torch.cuda.amp.autocast(enabled=use_amp):
    #             with torch.set_grad_enabled(enable_grad):
    #                 text_embed = self.text_encoder(**text_encoder_input, output_all_encoded_layers=True)[0][-1]
    #         text_embed = text_embed.to(dtype=dtype) # num_sentence, num_word, dim
    #         attention_mask = text_encoder_input['attention_mask'].unsqueeze(-1).to(dtype=dtype) # num_sentence, num_word, 1
    #         attention_mask[:,0,:] = 0. # This operation follows from the UniVL 
    #         text_embed = text_embed * attention_mask # num_sentence, num_word, dim
    #         text_embed = text_embed.sum(dim=1) / attention_mask.sum(dim=1) # num_sentence, dim
    #         raw_text_embed = text_embed
    #         # if video_name:
    #         #     text_feature_path = '/cpfs01/shared/Gvlab-A100/Gvlab-A100_hdd/wuhao/youcook2/UniVL_features/UniVL_text'
    #         #     np.save('{}/{}.npy'.format(text_feature_path, video_name), text_embed.detach().cpu().numpy())
    #         text_embed = self.contrastive_projection_text[-1](text_embed)
            
    #     else:
    #         dtype = next(self.parameters()).dtype
    #         enable_grad = False
    #         use_amp = False
    #         with torch.cuda.amp.autocast(enabled=use_amp):
    #             with torch.set_grad_enabled(enable_grad):
    #                 text_embed = self.text_encoder(**text_encoder_input)
    #         text_embed = text_embed['pooler_output'].to(dtype=dtype) # num_sentence, dim
    #         text_embed = self.contrastive_projection_text[-1](text_embed) # num_sentence, dim_contrastive_learning
    #         # TODO: add more paradigm to generate the text_embedding

    #     return text_embed, raw_text_embed

    def forward(self, dt, criterion, contrastive_criterion, eval_mode=False):
        transformer_input_type = self.opt.transformer_input_type
        vf = dt['video_tensor']  # (N, L, C)
        mask = ~ dt['video_mask']  # (N, L)
        duration = dt['video_length'][:, 1]
        video_name = dt['video_key'][0][2:]
        # text_encoder_input = dt['text_encoder_input'] if (self.opt.matcher_type=='DTW' or self.opt.use_pseudo_box) else None
        N, L, C = vf.shape
        # assert N == 1, "batch size must be 1."s

        srcs, masks, pos = self.base_encoder(vf, mask, duration)

        src_flatten, temporal_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten = self.transformer.prepare_encoder_inputs(
            srcs, masks, pos)
        memory = self.transformer.forward_encoder(src_flatten, temporal_shapes, level_start_index, valid_ratios,
                                                  lvl_pos_embed_flatten, mask_flatten)

        two_stage, disable_iterative_refine, proposals, proposals_mask = decide_two_stage(transformer_input_type,
                                                                                                dt, criterion)
        if two_stage:
            if transformer_input_type == 'prior_proposals':
                if self.opt.prior_manner == 'add':
                    #print('Insert the prior knowledge by adding the prior proposals to the query embed')
                    init_query_embed = self.query_embed.weight
                    _, tgt = torch.chunk(init_query_embed, 2, dim=1)
                    tgt = tgt.unsqueeze(0).expand(N, -1, -1)
                    init_reference, _, reference_points, query_embed = self.transformer.prepare_decoder_input_prior(proposals, num_queries = self.query_embed.weight.shape[0])
                    proposals_mask = torch.ones(N, self.query_embed.weight.shape[0], device=query_embed.device).bool()
                else:
                    init_reference, tgt, reference_points, query_embed = self.transformer.prepare_decoder_input_prior(proposals, num_queries = self.query_embed.weight.shape[0])
                    proposals_mask = torch.ones(N, self.query_embed.weight.shape[0], device=query_embed.device).bool()
            else:
                init_reference, tgt, reference_points, query_embed = self.transformer.prepare_decoder_input_proposal(
                    proposals)
        else:
            if self.opt.use_anchor:
                # tgt = self.tgt_embed.weight
                anchor = self.anchor_embed.weight # num_queries, 2
                query_anchor = (self.query_embed, anchor)
                proposals_mask = torch.ones(N, self.query_embed.shape[0], device=self.query_embed.device).bool()
                init_reference, tgt, reference_points, query_embed = self.transformer.prepare_decoder_input_anchor(memory, query_anchor)
            else:            
                query_embed = self.query_embed.weight
                proposals_mask = torch.ones(N, query_embed.shape[0], device=query_embed.device).bool()
                init_reference, tgt, reference_points, query_embed = self.transformer.prepare_decoder_input_query(memory,
                                                                                                              query_embed)
        hs, inter_references = self.transformer.forward_decoder(tgt, reference_points, memory, temporal_shapes,
                                                                level_start_index, valid_ratios, query_embed,
                                                                mask_flatten, proposals_mask, disable_iterative_refine)
        # hs: [num_decoder_layer, bs, num_query, feat_dim]

        # breakpoint()
        # project to co-embedding space
        if self.load_text_embed and eval_mode==False:
            # text_embed, raw_text_embed = self.text_encoding(text_encoder_input)
            # text_embed = [text_embed] * hs.shape[0]
            # text_embed = torch.stack(text_embed, dim=0)
            raw_text_embed = dt['cap_embed'] * hs.shape[0]# dt['caption_embedding'] returns a tuple(list)
            # text_embed: [num_decoder_layer, num_sentence, contrastive_dim]
            event_embed = torch.stack([self.contrastive_projection_event[i](hs_i) for i, hs_i in enumerate(hs)])
            text_embed = torch.stack([self.contrastive_projection_text[j](hs_j.cuda()) for j, hs_j in enumerate(raw_text_embed)])
            # breakpoint()
            # event_embed: [num_decoder_layer, num_query, contrastive_dim]
        else:
            raw_text_embed = None
            text_embed = None
            event_embed = hs
        # breakpoint()
        if self.opt.use_pseudo_box and self.training:
            # breakpoint()
            # print('use pseudo box')
            video_frame_num = dt['video_length'][:,0].cpu().numpy() # [feature_len, raw_video_len, video_len]
            video_name = dt['video_key'][0]
            if self.pseudo_boxes.get(video_name) is not None and 'box' in self.pseudo_boxes[video_name].keys() and 'loss' in self.pseudo_boxes[video_name].keys():
                # if self.opt.pseudo_box_type == 'similarity_op_order_v2' or self.opt.pseudo_box_type == 'similarity_op_v2':
                video_step_alignment = [self.pseudo_boxes[video_name]['box']]

            else:
                if self.opt.pseudo_box_type == 'align':
                    video_step_segment = [segment_video_into_steps(dt['video_tensor'][i], raw_text_embed[i].to(memory.device)) for i in range(N)]
                    bbox_alignment = [torch.tensor(alignment_to_boundary(video_step_segment[i], video_frame_num)).to(memory.device) for i in range(N)]
                elif self.opt.pseudo_box_type == 'similarity_op_order_v2':
                    video_step_alignment = [align_frame_into_steps_op_order_v2(dt['video_tensor'][i], raw_text_embed[i].to(memory.device), topk=self.opt.top_frames, threshold=self.opt.width_th, ratio=self.opt.width_ratio, iteration=self.opt.iteration) for i in range(N)]
                elif self.opt.pseudo_box_type == 'similarity_op_v2':
                    video_step_alignment = [align_frame_into_steps_op_v2(dt['video_tensor'][i], raw_text_embed[i].to(memory.device), topk=self.opt.top_frames, threshold=self.opt.width_th, ratio=self.opt.width_ratio, iteration=self.opt.iteration) for i in range(N)]
                elif self.opt.pseudo_box_type == 'uniform':
                    video_step_alignment = [uniform_box(dt['video_tensor'][i], raw_text_embed[i].to(memory.device)) for i in range(N)]
                    # breakpoint()
                else:
                    raise NotImplementedError('pseudo_box_type {} is not implemented'.format(self.opt.pseudo_box_type))
                

                if self.opt.pseudo_box_type != 'align':
                    if self.opt.pseudo_box_type == 'similarity_op_order_v2' or self.opt.pseudo_box_type == 'similarity_op_v2':
                        # breakpoint()
                        video_step_alignment, loss_op = [out[0] for out in video_step_alignment], [out[1] for out in video_step_alignment]
                        self.pseudo_boxes[video_name] = {'box': video_step_alignment[0], 'loss': loss_op[0].item()}
                    else:
                        self.pseudo_boxes[video_name] = {'box': video_step_alignment[0]}
            
            if self.opt.pseudo_box_type != 'align':
                bbox_alignment = [(torch.tensor(video_step_alignment[i]) / video_frame_num).to(memory.device).to(torch.float32) for i in range(N)]
            else:
                bbox_alignment = [torch.tensor(alignment_to_boundary(video_step_segment[i], video_frame_num)).to(memory.device) for i in range(N)]

                
                # self.pseudo_boxes[video_name] = video_step_alignment[0]
                # self.pseudo_boxes[video_name] = video_step_alignment[0]
                # bbox_alignment = [torch.tensor(alignment_to_boundary(video_step_segment[i], video_frame_num)).to(memory.device) for i in range(N)]

            bbox_alignment = to_center_duration(bbox_alignment)


            for sample in range(len(dt['video_target'])):
                dt['video_target'][sample]['boxes_pseudo'] = bbox_alignment[sample]
                # dt['video_target'][sample]['boxes'] = bbox_alignment[sample]
        # else:
        #     print('use gt box')

        #breakpoint()
        others = {'memory': memory,
                  'mask_flatten': mask_flatten,
                  'spatial_shapes': temporal_shapes,
                  'level_start_index': level_start_index,
                  'valid_ratios': valid_ratios,
                  'proposals_mask': proposals_mask,
                  'text_embed': text_embed,
                  'event_embed': event_embed}
        # breakpoint()
        if eval_mode or self.opt.caption_loss_coef == 0:
            out, loss = self.parallel_prediction_full(dt, criterion, hs, init_reference, inter_references, others,
                                                      disable_iterative_refine, transformer_input_type)
        else:
            if self.opt.refine_pseudo_box and self.opt.use_pseudo_box:
                # print('refine')
                out, loss = self.parallel_prediction_refine_matched(dt, criterion, contrastive_criterion, hs, init_reference, inter_references, others,
                                                         disable_iterative_refine, transformer_input_type)
            else:
                # print('no refine')
                out, loss = self.parallel_prediction_matched(dt, criterion, contrastive_criterion, hs, init_reference, inter_references, others,
                                                         disable_iterative_refine, transformer_input_type)
        return out, loss

    def predict_event_num(self, counter, hs_lid):
        hs_lid_pool = torch.max(hs_lid, dim=1, keepdim=False)[0]  # [bs, feat_dim]
        outputs_class0 = counter(hs_lid_pool)
        return outputs_class0

    def parallel_prediction_full(self, dt, criterion, hs, init_reference, inter_references, others,
                                 disable_iterative_refine, transformer_input_type='queries'):
        '''
        hs: [decoder_layer, bs, num_query, feat_dim]
        init_reference: [bs, num_query, 1]
        inter_references: [decoder_layer, bs, num_query, 2]
        '''
        outputs_classes = []
        outputs_classes0 = []
        outputs_coords = []
        outputs_cap_losses = []
        outputs_cap_probs = []
        outputs_cap_seqs = []
        num_pred = hs.shape[0]
        #breakpoint()
        for l_id in range(hs.shape[0]):
            if l_id == 0:
                reference = init_reference
            else:
                reference = inter_references[l_id - 1]  # [decoder_layer, batch, query_num, ...]
            hs_lid = hs[l_id]
            outputs_class = self.class_head[l_id](hs_lid)  # [bs, num_query, N_class]
            output_count = self.predict_event_num(self.count_head[l_id], hs_lid)
            n_pred_sentence = output_count.argmax(dim=-1).clamp(min=1).item()
            tmp = self.bbox_head[l_id](hs_lid)  # [bs, num_query, 4]

            # if self.opt.disable_mid_caption_heads and (l_id != hs.shape[0] - 1):
            if l_id != hs.shape[0] - 1:
                cap_probs, seq = self.caption_prediction_eval(
                    self.caption_head[l_id], dt, hs_lid, reference, others, 'none')
            else:
                cap_probs, seq = self.caption_prediction_eval(
                    self.caption_head[l_id], dt, hs_lid, reference, others, self.opt.caption_decoder_type)  # Only output caption in the last decoding layer

            # if self.opt.use_anchor:
            #     outputs_coord = reference
            # else:
            if disable_iterative_refine:
                outputs_coord = reference
            else:
                reference = inverse_sigmoid(reference)
                if self.opt.matcher_type == 'DTW':
                    assert reference.shape[-1] == 2 and tmp.shape[-1] == 2
                if reference.shape[-1] == 2:
                    tmp += reference
                else:
                    assert reference.shape[-1] == 1
                    tmp[..., :2] += reference
                outputs_coord = tmp.sigmoid()  # [bs, num_query, 2]

            outputs_classes.append(outputs_class)
            outputs_classes0.append(output_count)
            outputs_coords.append(outputs_coord)
            outputs_cap_probs.append(cap_probs)
            outputs_cap_seqs.append(seq)
        outputs_class = torch.stack(outputs_classes)  # [decoder_layer, bs, num_query, N_class]
        output_count = torch.stack(outputs_classes0)
        outputs_coord = torch.stack(outputs_coords)  # [decoder_layer, bs, num_query, 4]

        all_out = {'pred_logits': outputs_class,
                   'pred_count': output_count,
                   'pred_boxes': outputs_coord,
                   'caption_probs': outputs_cap_probs,
                   'seq': outputs_cap_seqs}
        out = {k: v[-1] for k, v in all_out.items()}

        if self.aux_loss:
            ks, vs = list(zip(*(all_out.items())))
            out['aux_outputs'] = [{ks[i]: vs[i][j] for i in range(len(ks))} for j in range(num_pred - 1)]

        # loss, _, _ = criterion(out, dt['video_target'], others)
        return out, []

    def parallel_prediction_refine_matched(self, dt, criterion, contrastive_criterion, hs, init_reference, inter_references, others,
                                    disable_iterative_refine, transformer_input_type='queries'):
        
        outputs_classes = []
        outputs_counts = []
        outputs_coords = []
        outputs_cap_costs = []
        outputs_cap_losses = []
        outputs_cap_probs = []
        outputs_cap_seqs = []
        cl_match_mats = []

        num_pred = hs.shape[0]
        if self.opt.pseudo_box_aug:
            assert self.opt.use_pseudo_box
            num_sentence = dt['gt_boxes'].size(-2)
            assert num_sentence == len(dt['cap_raw'][0])
            if self.opt.pseudo_box_aug_num * num_sentence > self.opt.num_queries:
                aug_num = self.opt.num_queries // num_sentence
            else:
                aug_num = self.opt.pseudo_box_aug_num
            if self.opt.refine_pseudo_box:
                ori_dt_cap_tensor = copy.deepcopy(dt['cap_tensor'])
                ori_dt_cap_mask = copy.deepcopy(dt['cap_mask'])
            cap_dim = dt['cap_tensor'].shape[-1] #(num_sen, num_max_word)
            dt['cap_tensor'] = dt['cap_tensor'].repeat(1, aug_num).reshape(-1, cap_dim)
            dt['cap_mask'] = dt['cap_mask'].repeat(1, aug_num).reshape(-1, cap_dim)

        for l_id in range(num_pred):
            hs_lid = hs[l_id]
            reference = init_reference if l_id == 0 else inter_references[
                l_id - 1]  # [decoder_layer, batch, query_num, ...]
            outputs_class = self.class_head[l_id](hs_lid)  # [bs, num_query, N_class]
            outputs_count = self.predict_event_num(self.count_head[l_id], hs_lid)
            tmp = self.bbox_head[l_id](hs_lid)  # [bs, num_query, 2]
            
            cost_caption, loss_caption, cap_probs, seq = self.caption_prediction(self.caption_head[l_id], dt, hs_lid,
                                                                                 reference, others, 'none')

            if disable_iterative_refine:
                outputs_coord = reference
            else:
                reference = inverse_sigmoid(reference)
                if reference.shape[-1] == 2:
                    tmp += reference
                else:
                    assert reference.shape[-1] == 1
                    tmp[..., :1] += reference
                outputs_coord = tmp.sigmoid()  # [bs, num_query, 4]

            # Processing the text embed and event embed for alignment
            if self.load_text_embed or self.opt.disable_contrastive_projection:
                assert others['text_embed'].shape[0] == num_pred, \
                    'visual features have {} levels, but text have {}'.format(num_pred, others['text_embed'].shape[0])
                text_embed = others['text_embed'][l_id]   # [num_sentence, contrastive_dim]
                event_embed = others['event_embed'][l_id] 
                event_embed = event_embed.reshape(-1, event_embed.shape[-1]) # [num_query, contrastive_dim]
                # event_embed = event_embed.reshape(-1, event_embed.shape[-1])
                # TODO: complete the contrastive learning to return the similarity matrices as 'cl_match_mat'


            if self.opt.enable_contrastive and self.opt.set_cost_cl > 0:
                assert len(others['text_embed']) == num_pred, \
                    'visual features have {} levels, but text have {}'.format(num_pred, len(others['text_embed']))
                text_embed = torch.cat(others['text_embed'][l_id], dim=0)   # [num_sentence, contrastive_dim]
                event_embed = others['event_embed'][l_id]
                event_embed = event_embed.reshape(-1, event_embed.shape[-1]) # [num_query, contrastive_dim]
                cl_match_mat = contrastive_criterion.forward_logits(text_embed, event_embed, self.background_embed).t()
                # cl_match_mat: [num_query, num_sentence]
                cl_match_mats.append(cl_match_mat)
            else:
                cl_match_mats.append(0)

            outputs_classes.append(outputs_class)
            outputs_counts.append(outputs_count)
            outputs_coords.append(outputs_coord)
            # outputs_cap_losses.append(cap_loss)
            outputs_cap_probs.append(cap_probs)
            outputs_cap_seqs.append(seq)

        outputs_class = torch.stack(outputs_classes)  # [decoder_layer, bs, num_query, N_class]
        outputs_count = torch.stack(outputs_counts)
        outputs_coord = torch.stack(outputs_coords)  # [decoder_layer, bs, num_query, 4]
        # outputs_cap_loss = torch.stack(outputs_cap_losses)

        all_out = {
            'pred_logits': outputs_class,
            'pred_count': outputs_count,
            'pred_boxes': outputs_coord,
            'caption_probs': outputs_cap_probs,
            'seq': outputs_cap_seqs,
            'cl_match_mats': cl_match_mats}
        out = {k: v[-1] for k, v in all_out.items()}


        # ============================= Refine pseudo box here ================================
        ks, vs = list(zip(*(all_out.items())))
        out['aux_outputs'] = [{ks[i]: vs[i][j] for i in range(len(ks))} for j in range(num_pred - 1)]
        mil_dict = {}
        bag_score_cache = []
        for stage in range(self.opt.refine_pseudo_stage_num):
            # Decay augment ratio as the stage increases
            aug_ratio = self.opt.pseudo_box_aug_ratio * (0.5 ** stage)
            _, last_indices, aux_indices = criterion(out, dt['video_target'], others, aug_num, aug_ratio)
            # Only use the last decoder layer output to conduct the pseudo box refinement
            hs_lid = hs[-1]
            reference = inter_references[-1] #[1, num_query, 2]
            indices = last_indices[0] # [tensor(): num_matched_query ,tensor(): num_matched_cap]
            query_indices = indices[0][0] # the indices of matched query is ordered
            cap_indices = indices[0][1] # the indices of matched sentence is unordered
            # breakpoint()
            # num_sentence = cap_indices.size(0) // self.opt.pseudo_box_aug_num
            cap_sort = torch.sort(cap_indices)[1]
            reorder_query_indices = query_indices[cap_sort]
            if self.opt.use_neg_pseudo_box:
                neg_query_indices = []
                neg_cap_indices = torch.arange(0,cap_indices.size(0),aug_num).view(num_sentence,-1).repeat(1,self.opt.num_neg_box).view(-1)
                for i in range(num_sentence):
                    # select some negetive indices from reordered query indices
                    candidates_r = (reorder_query_indices[(i+1)*aug_num:])
                    candidates_l = (reorder_query_indices[:(i)*aug_num])
                    if (candidates_r.size(0) > 0) and (candidates_l.size(0) > 0):
                        candidates = torch.cat((candidates_r, candidates_l))
                    else:
                        candidates = candidates_r if candidates_r.size(0) > 0 else candidates_l
                    if candidates.size(0) == 0:
                        candidates = reorder_query_indices
                    if candidates.size(0) < self.opt.num_neg_box:
                        random_selected_indices = torch.randperm(candidates.size(0))
                        padding_num = self.opt.num_neg_box - candidates.size(0)
                        random_selected_indices = torch.cat((random_selected_indices, random_selected_indices[:padding_num]))
                    else:
                        random_selected_indices = torch.randperm(reorder_query_indices.size(0)-aug_num)[:self.opt.num_neg_box]
                    neg_query_indices.append(candidates[random_selected_indices])
                neg_query_indices = torch.cat(neg_query_indices)
                neg_indices = [(neg_query_indices, neg_cap_indices)]
            # query_indices: ordered, cap_indices: unordered
            # ++++++ <1>. Produce the instance score and classification score
            if self.opt.use_additional_cap_layer:
                cap_loss, cap_probs, seq, sentence_cap_prob = self.caption_prediction(self.caption_head_refine[stage], dt, hs_lid, reference,
                                                                    others, self.opt.caption_decoder_type, indices)
                if (stage > 0) and self.opt.use_neg_pseudo_box:
                    _, _, _, neg_cap_prob = self.caption_prediction(self.caption_head_refine[stage], dt, hs_lid, reference,
                                                                    others, self.opt.caption_decoder_type, neg_indices)
            else:
                cap_loss, cap_probs, seq, sentence_cap_prob = self.caption_prediction(self.caption_head[-1], dt, hs_lid, reference,
                                                                    others, self.opt.caption_decoder_type, indices)
                if (stage > 0) and self.opt.use_neg_pseudo_box:
                    _, _, _, neg_cap_prob = self.caption_prediction(self.caption_head[-1], dt, hs_lid, reference,
                                                                    others, self.opt.caption_decoder_type, neg_indices) 
            # breakpoint()   
            # sentence_cap_prob: the caption probility for each matched query torch.Size([num_matched_query])
            if self.opt.use_additional_score_layer:
                query_ins_score = self.class_refine_head[stage](hs_lid)[:, query_indices, :]
            else:
                query_ins_score = outputs_classes[-1][:, query_indices, :] # [1, num_matched_query, 1]
            query_pred_boxes = outputs_coord[-1][:, query_indices, :] # [1, num_matched_query, 2]
            query_pred_boxes = query_pred_boxes[0,:,:][cap_sort].view(-1, 2) # [num_matched_query, 2]
            # breakpoint()
            try:
                query_ins_score = query_ins_score[0,cap_sort,0].view(-1, aug_num) # [num_cap, num_aug]
            except:
                breakpoint()
            if self.opt.norm_ins_score == 'softmax':
                query_ins_score = torch.softmax(query_ins_score, dim=-1)
            elif self.opt.norm_ins_score == 'sigmoid':
                query_ins_score = query_ins_score.sigmoid()
            else:
                raise NotImplementedError

            # breakpoint()
            # sentence_cap_score = cap_probs['cap_prob_train']
            temperature = 2
            sentence_cap_prob = sentence_cap_prob[cap_sort].view(-1, aug_num) # [num_cap, num_aug]
            cap_len = torch.tensor([len(cap.split()) for cap in dt['cap_raw'][0]], device=sentence_cap_prob.device).unsqueeze(1)
            sentence_cap_score = (sentence_cap_prob / cap_len) ** temperature + 1e-5

            sentence_cap_score[torch.isinf(sentence_cap_score)] = 1e8

            sentence_cap_score = sentence_cap_score.detach()
            query_ins_score = query_ins_score.detach()

            # breakpoint()
            query_score = sentence_cap_score + query_ins_score
            # sentence_score = 
            # if (stage == 0) or (self.opt.focal_mil == False):
            #     sentence_cap_prob = torch.softmax(sentence_cap_prob, dim=-1) # Softmax over queries in the same bag
            # else:
            #     sentence_cap_prob = sentence_cap_prob.sigmoid()

            # if self.opt.cap_prob_clip:
            #     query_score = sentence_cap_prob.detach() * query_ins_score # [num_cap, num_aug]
            # else:
            #     query_score = sentence_cap_prob * query_ins_score # [num_cap, num_aug]

            # # ++++++ <2>. Calculate the MIL loss and Neg loss
            bag_score = query_score.sum(dim=-1) # [num_cap]
            bag_score = bag_score.clamp(0,1)
            bag_score_cache.append(bag_score)
            mil_weight = bag_score_cache[stage-1] if self.opt.weighted_mil_loss else torch.ones_like(bag_score).to(bag_score.device)
            if stage > 0:
                if self.opt.focal_mil:
                    focal_weight = (torch.ones_like(bag_score).to(bag_score.device) - bag_score).pow(2)
                    mil_loss =  - focal_weight * (bag_score + 1e-6).log()
                    mil_loss = (mil_weight * mil_loss).mean()
                else:
                    # breakpoint()
                    mil_loss = - (mil_weight * bag_score.log()).mean()
                if self.opt.use_neg_pseudo_box:
                    neg_cap_prob = neg_cap_prob.sigmoid()
                    neg_loss = - ((neg_cap_prob).pow(2) * (1- neg_cap_prob).log()).view(num_sentence,-1).mean(dim=-1)
                    neg_loss = (mil_weight * neg_loss).mean()
                    mil_loss += neg_loss
            else:
                mil_loss = F.binary_cross_entropy(bag_score, torch.ones_like(bag_score).to(bag_score.device))
            if 'loss_mil' in mil_dict.keys():
                mil_dict['loss_mil'] += mil_loss
            else:
                mil_dict['loss_mil'] =  mil_loss
            # ++++++ <3>. Merge the pseudo box to generate new pseudo box
            if self.opt.merge_criterion == 'cap_topk':
                topk_pseudo_scores, topk_pseudo_indices = torch.topk(sentence_cap_score, k=self.opt.merge_k_boxes, dim=-1) # [num_caption, k]
            elif self.opt.merge_criterion == 'ins_topk':
                topk_pseudo_scores, topk_pseudo_indices = torch.topk(query_ins_score, k=self.opt.merge_k_boxes, dim=-1)
            elif self.opt.merge_criterion == 'ins_cap_topk':
                topk_pseudo_scores, topk_pseudo_indices = torch.topk(query_score, k=self.opt.merge_k_boxes, dim=-1) # [num_caption, k]
            else:
                raise NotImplementedError('merge_criterion {} is not implemented'.format(self.opt.merge_criterion))
            # breakpoint()
            topk_pseudo_scores = topk_pseudo_scores / (topk_pseudo_scores.sum(dim=-1, keepdim=True) + 1e-6) # [num_caption, k]
            weight = topk_pseudo_scores.unsqueeze(-1).repeat(1,1,2) # [num_caption, k, 2]
            for i in range(len(dt['video_target'])):
                previous_pseudo_box = dt['video_target'][i]['box_pseudo_aug'] #[num_caption*num_aug, 2]
                if self.opt.use_query_box_for_refine:
                    # Use the coordinates of query as part of guidance for refinement
                    previous_pseudo_box = (previous_pseudo_box + query_pred_boxes) / 2
                if self.opt.merge_mode == 'weighted_sum':
                    # Merge top-k boxes with weighted sum
                    selected_pseudo_box = torch.gather(previous_pseudo_box.view(-1,aug_num,2), 1, \
                                                    topk_pseudo_indices.unsqueeze(-1).expand(-1,-1,previous_pseudo_box.size(-1))) # [num_caption, k, 2]
                    refined_pseudo_box = (weight * selected_pseudo_box).sum(dim=1).clamp(0,1) # [num_caption, 2]
                    dt['video_target'][i]['boxes_pseudo'] = refined_pseudo_box.detach().clone()
                # I met the following problem with ''targets_cp = copy.deepcopy(targets)'' in criterion.py:
                # RuntimeError: Only Tensors created explicitly by the user (graph leaves) support the deepcopy protocol at the moment
                # When I tried to conduct the deepcopy operation with the targets which have been updated with 'boxes_pseudo' keys
                # So I detach the refined_pseudo_box here to avoid the deepcopy operation here
                # Commented by Huabin, 2023/9/14
                elif self.opt.merge_mode == 'interpolate':
                    # Generate new box with linear interpolation between previous pbox and pbox with max score
                    max_pseudo_scores = topk_pseudo_scores[:,:1]
                    max_coef = 0.5 * torch.ones_like(max_pseudo_scores).to(max_pseudo_scores.device) # Set a max coef for box interpolatation
                    max_pseudo_box = torch.gather(previous_pseudo_box.view(-1,aug_num,2), 1, \
                                                    topk_pseudo_indices[:,:1].unsqueeze(-1).expand(-1,-1,previous_pseudo_box.size(-1)))
                    interpolate_coef = torch.min(max_pseudo_scores, max_coef)
                    refined_pseudo_box = (1-interpolate_coef) * previous_pseudo_box[(aug_num-1)::aug_num, :] \
                                        + interpolate_coef * max_pseudo_box.squeeze(1)
                    refined_pseudo_box = refined_pseudo_box.clamp(0,1)
                    dt['video_target'][i]['boxes_pseudo'] = refined_pseudo_box.detach().clone()

        # ++++++ <4>. End of the refinement, inverse-repeat the dt['cap_tensor'] and dt['cap_mask']
        dt['cap_tensor'] = ori_dt_cap_tensor
        dt['cap_mask'] = ori_dt_cap_mask
        mil_dict['loss_mil'] = mil_dict['loss_mil'] / self.opt.refine_pseudo_stage_num
        criterion.pseudo_box_aug = False
        # ================== End of refinement ========================================
        # breakpoint()
        if self.aux_loss:
            ks, vs = list(zip(*(all_out.items())))
            out['aux_outputs'] = [{ks[i]: vs[i][j] for i in range(len(ks))} for j in range(num_pred - 1)]
            loss, last_indices, aux_indices = criterion(out, dt['video_target'], others)
            if self.opt.disable_rematch:
                # Disable re-matching and directly use the indices with max score in the last stage of refinment
                selected_indices = query_score.argmax(dim=-1).unsqueeze(-1)
                query_indices_in_refine = reorder_query_indices.to(selected_indices.device).view(-1, aug_num)
                query_indices_in_refine = query_indices_in_refine.gather(1, selected_indices)
                query_indices_in_refine, index_sort = torch.sort(query_indices_in_refine, 0)
                cap_indices_in_refine = last_indices[0][0][1].sort()[0]
                last_indices = [[(query_indices_in_refine.view(-1), cap_indices_in_refine[index_sort.view(-1)])], last_indices[1]]
            loss.update(mil_dict)
            criterion.pseudo_box_aug = True 
            for l_id in range(hs.shape[0]):
                hs_lid = hs[l_id]
                reference = init_reference if l_id == 0 else inter_references[l_id - 1]
                indices = last_indices[0] if l_id == hs.shape[0] - 1 else aux_indices[l_id][0]
                cap_loss, cap_probs, seq, sentence_cap_prob = self.caption_prediction(self.caption_head[l_id], dt, hs_lid, reference,
                                                                   others, self.opt.caption_decoder_type, indices)
                l_dict = {'loss_caption': cap_loss}
                if l_id != hs.shape[0] - 1:
                    l_dict = {k + f'_{l_id}': v for k, v in l_dict.items()}
                loss.update(l_dict)
            out.update({'caption_probs': cap_probs, 'seq': seq})
        else:
            loss, last_indices = criterion(out, dt['video_target'], others)
            criterion.pseudo_box_aug = True
            l_id = hs.shape[0] - 1
            reference = inter_references[l_id - 1]  # [decoder_layer, batch, query_num, ...]
            hs_lid = hs[l_id]
            indices = last_indices[0]
            cap_loss, cap_probs, seq, sentence_cap_prob = self.caption_prediction(self.caption_head[l_id], dt, hs_lid, reference,
                                                               others, self.opt.caption_decoder_type, indices)
            l_dict = {'loss_caption': cap_loss}
            loss.update(l_dict)

            out.pop('caption_losses')
            out.pop('caption_costs')
            out.update({'caption_probs': cap_probs, 'seq': seq})


        return out, loss

    def parallel_prediction_matched(self, dt, criterion, contrastive_criterion, hs, init_reference, inter_references, others,
                                    disable_iterative_refine, transformer_input_type='queries'):
        
        outputs_classes = []
        outputs_counts = []
        outputs_coords = []
        outputs_cap_costs = []
        outputs_cap_losses = []
        outputs_cap_probs = []
        outputs_cap_seqs = []
        cl_match_mats = []

        num_pred = hs.shape[0]

        if self.opt.pseudo_box_aug:
            assert self.opt.use_pseudo_box
            cap_dim = dt['cap_tensor'].shape[-1] # (num_sen, num_max_word)
            dt['cap_tensor'] = dt['cap_tensor'].repeat(1, self.opt.pseudo_box_aug_num).reshape(-1, cap_dim)
            dt['cap_mask'] = dt['cap_mask'].repeat(1, self.opt.pseudo_box_aug_num).reshape(-1, cap_dim)

        for l_id in range(num_pred):
            hs_lid = hs[l_id]
            reference = init_reference if l_id == 0 else inter_references[
                l_id - 1]  # [decoder_layer, batch, query_num, ...]
            outputs_class = self.class_head[l_id](hs_lid)  # [bs, num_query, N_class]
            outputs_count = self.predict_event_num(self.count_head[l_id], hs_lid)
            tmp = self.bbox_head[l_id](hs_lid)  # [bs, num_query, 2]


            cost_caption, loss_caption, cap_probs, seq = self.caption_prediction(self.caption_head[l_id], dt, hs_lid,
                                                                                 reference, others, 'none')
            # if self.opt.use_anchor:
            #     outputs_coord = reference
            # else:
            if disable_iterative_refine:
                outputs_coord = reference
            else:
                reference = inverse_sigmoid(reference)
                if reference.shape[-1] == 2:
                    tmp += reference
                else:
                    assert reference.shape[-1] == 1
                    tmp[..., :1] += reference
                outputs_coord = tmp.sigmoid()  # [bs, num_query, 4]

            # Processing the text embed and event embed for alignment
            if self.load_text_embed or not self.opt.disable_contrastive_projection:
                assert others['text_embed'].shape[0] == num_pred, \
                    'visual features have {} levels, but text have {}'.format(num_pred, others['text_embed'].shape[0])
                text_embed = others['text_embed'][l_id]   # [num_sentence, contrastive_dim]
                event_embed = others['event_embed'][l_id] 
                event_embed = event_embed.reshape(-1, event_embed.shape[-1]) # [num_query, contrastive_dim]
                # event_embed = event_embed.reshape(-1, event_embed.shape[-1])
                # TODO: complete the contrastive learning to return the similarity matrices as 'cl_match_mat'


            if self.opt.enable_contrastive and self.opt.set_cost_cl > 0:
                assert len(others['text_embed']) == num_pred, \
                    'visual features have {} levels, but text have {}'.format(num_pred, len(others['text_embed']))
                text_embed = torch.cat(others['text_embed'][l_id], dim=0)   # [num_sentence, contrastive_dim]
                event_embed = others['event_embed'][l_id]
                event_embed = event_embed.reshape(-1, event_embed.shape[-1]) # [num_query, contrastive_dim]
                cl_match_mat = contrastive_criterion.forward_logits(text_embed, event_embed, self.background_embed).t()
                # cl_match_mat: [num_query, num_sentence]
                cl_match_mats.append(cl_match_mat)
            else:
                cl_match_mats.append(0)

            outputs_classes.append(outputs_class)
            outputs_counts.append(outputs_count)
            outputs_coords.append(outputs_coord)
            # outputs_cap_losses.append(cap_loss)
            outputs_cap_probs.append(cap_probs)
            outputs_cap_seqs.append(seq)

        outputs_class = torch.stack(outputs_classes)  # [decoder_layer, bs, num_query, N_class]
        outputs_count = torch.stack(outputs_counts)
        outputs_coord = torch.stack(outputs_coords)  # [decoder_layer, bs, num_query, 4]
        # outputs_cap_loss = torch.stack(outputs_cap_losses)

        all_out = {
            'pred_logits': outputs_class,
            'pred_count': outputs_count,
            'pred_boxes': outputs_coord,
            'caption_probs': outputs_cap_probs,
            'seq': outputs_cap_seqs,
            'cl_match_mats': cl_match_mats}
        out = {k: v[-1] for k, v in all_out.items()}

        if self.aux_loss:
            ks, vs = list(zip(*(all_out.items())))
            out['aux_outputs'] = [{ks[i]: vs[i][j] for i in range(len(ks))} for j in range(num_pred - 1)]
            if transformer_input_type == 'prior_proposals':
                loss, _, _ = criterion(out, dt['video_target'])
                # Random select an query from each segment
                num_sentence = dt['cap_tensor'].shape[0]
                num_query = hs.shape[-2]
                num_query_interval = num_query // num_sentence
                query_indices = []
                for i in range(num_sentence):
                    interval_min = i * num_query_interval
                    interval_max = interval_min + num_query_interval
                    sample = torch.randint(interval_min, interval_max, (hs.shape[0],))
                    query_indices.append(sample)
                query_indices = torch.cat(query_indices, dim=0)
                gt_indices = torch.arange(num_sentence)

                last_indices = ([(query_indices[::hs.shape[0]], gt_indices)], [None, None])
                aux_indices = []
                for l_id in range(hs.shape[0]-1):
                    aux_indices.append(([(query_indices[(l_id+1)::hs.shape[0]], gt_indices)], [None, None]))
            else:
                loss, last_indices, aux_indices = criterion(out, dt['video_target'], others)
            for l_id in range(hs.shape[0]):
                hs_lid = hs[l_id]
                reference = init_reference if l_id == 0 else inter_references[l_id - 1]
                indices = last_indices[0] if l_id == hs.shape[0] - 1 else aux_indices[l_id][0]
                cap_loss, cap_probs, seq, sentence_cap_prob = self.caption_prediction(self.caption_head[l_id], dt, hs_lid, reference,
                                                                   others, self.opt.caption_decoder_type, indices)

                l_dict = {'loss_caption': cap_loss}
                if (self.opt.matcher_type == 'DTW' or self.opt.matcher_type == 'Sim'):
                    contrastive_loss = contrastive_criterion(
                        text_embed = others['text_embed'][l_id],
                        event_embed = others['event_embed'][l_id],
                        matching_indices = indices,
                        bg_embed = self.background_embed,
                    )

                    l_dict.update({'contrastive_loss': contrastive_loss})
                if l_id != hs.shape[0] - 1:
                    l_dict = {k + f'_{l_id}': v for k, v in l_dict.items()}
                loss.update(l_dict)
            out.update({'caption_probs': cap_probs, 'seq': seq})
        else:
            loss, last_indices = criterion(out, dt['video_target'], others)

            l_id = hs.shape[0] - 1
            reference = inter_references[l_id - 1]  # [decoder_layer, batch, query_num, ...]
            hs_lid = hs[l_id]
            indices = last_indices[0]
            cap_loss, cap_probs, seq, sentence_cap_prob = self.caption_prediction(self.caption_head[l_id], dt, hs_lid, reference,
                                                               others, self.opt.caption_decoder_type, indices)
            l_dict = {'loss_caption': cap_loss}
            loss.update(l_dict)

            out.pop('caption_losses')
            out.pop('caption_costs')
            out.update({'caption_probs': cap_probs, 'seq': seq})

        return out, loss

    def caption_prediction(self, cap_head, dt, hs, reference, others, captioner_type, indices=None):
        N_, N_q, C = hs.shape
        # all_cap_num = len(dt['cap_tensor'])
        # if self.opt.pseudo_box_aug:
        #     assert self.opt.use_pseudo_box
        #     cap_dim = dt['cap_tensor'].shape[-1] # (num_sen, num_max_word)
        #     # breakpoint()
        #     if indices != None:
        #         breakpoint()
        #     dt['cap_tensor'] = dt['cap_tensor'].repeat(1, self.opt.pseudo_box_aug_num).reshape(-1, cap_dim)
        #     dt['cap_mask'] = dt['cap_mask'].repeat(1, self.opt.pseudo_box_aug_num).reshape(-1, cap_dim)
        all_cap_num = len(dt['cap_tensor'])
        query_mask = others['proposals_mask']
        gt_mask = dt['gt_boxes_mask']
        mix_mask = torch.zeros(query_mask.sum().item(), gt_mask.sum().item())
        query_nums, gt_nums = query_mask.sum(1).cpu(), gt_mask.sum(1).cpu()
        hs_r = torch.masked_select(hs, query_mask.unsqueeze(-1)).reshape(-1, C)

        if indices == None:
            row_idx, col_idx = 0, 0
            for i in range(N_):
                mix_mask[row_idx: (row_idx + query_nums[i]), col_idx: (col_idx + gt_nums[i])] = 1
                row_idx=row_idx + query_nums[i]
                col_idx= col_idx + gt_nums[i]

            bigids = mix_mask.nonzero(as_tuple=False)
            feat_bigids, cap_bigids = bigids[:, 0], bigids[:, 1]
        else:
            # breakpoint()
            feat_bigids = torch.zeros(sum([len(_[0]) for _ in indices])).long()
            cap_bigids = torch.zeros_like(feat_bigids)
            total_query_ids = 0
            total_cap_ids = 0
            total_ids = 0
            max_pair_num = max([len(_[0]) for _ in indices])
            new_hr_for_dsa = torch.zeros(N_, max_pair_num, C)  # only for lstm-dsa
            cap_seq = dt['cap_tensor']
            new_seq_for_dsa = torch.zeros(N_, max_pair_num, cap_seq.shape[-1], dtype=cap_seq.dtype)  # only for lstm-dsa
            for i, index in enumerate(indices):
                feat_ids, cap_ids = index
                feat_bigids[total_ids: total_ids + len(feat_ids)] = total_query_ids + feat_ids
                cap_bigids[total_ids: total_ids + len(feat_ids)] = total_cap_ids + cap_ids
                new_hr_for_dsa[i, :len(feat_ids)] = hs[i, feat_ids]
                new_seq_for_dsa[i, :len(feat_ids)] = cap_seq[total_cap_ids + cap_ids]
                total_query_ids += query_nums[i]
                total_cap_ids += gt_nums[i]
                total_ids += len(feat_ids)
            # if self.opt.pseudo_box_aug:
            #     # Revise the matched targer ids for pseudo box augmentation to caption id
            #     cap_bigids = cap_bigids // self.opt.pseudo_box_aug_num
        cap_probs = {}
        flag = True

        if captioner_type == 'none':
            cost_caption = torch.zeros(N_, N_q, all_cap_num,
                                       device=hs.device)  # batch_size * num_queries * all_caption_num
            loss_caption = torch.zeros(N_, N_q, all_cap_num, device=hs.device)
            cap_probs['cap_prob_train'] = torch.zeros(1, device=hs.device)
            cap_probs['cap_prob_eval'] = torch.zeros(N_, N_q, 3, device=hs.device)
            seq = torch.zeros(N_, N_q, 3, device=hs.device)
            return cost_caption, loss_caption, cap_probs, seq

        elif captioner_type in ['light']:
            clip = hs_r.unsqueeze(1)
            clip_mask = clip.new_ones(clip.shape[:2])
            event = None
        elif self.opt.caption_decoder_type == 'standard':
            # breakpoint()
            # assert N_ == 1, 'only support batchsize = 1'
            if self.training:
                # breakpoint()
                seq = dt['cap_tensor'][cap_bigids]
                if self.opt.caption_cost_type != 'rl':
                    if self.opt.refine_pseudo_box: # Only training and refine_pseudo_box = True returns the raw_cap_prob
                        cap_prob, raw_cap_prob = cap_head(hs[:, feat_bigids], reference[:, feat_bigids], others, seq)
                        # shape: [num_sentence, max_num_word, num_vocab]
                        # cap_prob is log_softmax(prob), raw_cap_prob is (prob)
                        cap_probs['cap_prob_train'] = cap_prob
                        cap_probs['raw_cap_prob'] = raw_cap_prob
                    else:
                        cap_prob = cap_head(hs[:, feat_bigids], reference[:, feat_bigids], others, seq) 
                        # [num_matched_query, max_length_sentence, num_word_in_vocab], e.g., [5, 13, 1608], here 13 is the max length among 5 sentences
                        cap_probs['cap_prob_train'] = cap_prob
            else:
                with torch.no_grad():
                    cap_prob = cap_head(hs[:, feat_bigids], reference[:, feat_bigids], others,
                                        dt['cap_tensor'][cap_bigids])
                    seq, cap_prob_eval = cap_head.sample(hs, reference, others)
                    if len(seq):
                        seq = seq.reshape(-1, N_q, seq.shape[-1])
                        cap_prob_eval = cap_prob_eval.reshape(-1, N_q, cap_prob_eval.shape[-1])
                    cap_probs['cap_prob_eval'] = cap_prob_eval

            flag = False
            pass

        if flag:
            clip_ext = clip[feat_bigids]
            clip_mask_ext = clip_mask[feat_bigids]

            if self.training:
                seq = dt['cap_tensor'][cap_bigids]
                if self.opt.caption_cost_type != 'rl':
                    cap_prob = cap_head(event, clip_ext, clip_mask_ext, seq)
                    cap_probs['cap_prob_train'] = cap_prob
            else:
                with torch.no_grad():
                    seq_gt = dt['cap_tensor'][cap_bigids]
                    cap_prob = cap_head(event, clip_ext, clip_mask_ext, seq_gt)
                    seq, cap_prob_eval = cap_head.sample(event, clip, clip_mask)

                    if len(seq):
                        # re_seq = torch.zeros(N_, N_q, seq.shape[-1])
                        # re_cap_prob_eval = torch.zeros(N_, N_q, cap_prob_eval.shape[-1])
                        seq = seq.reshape(-1, N_q, seq.shape[-1])
                        cap_prob_eval = cap_prob_eval.reshape(-1, N_q, cap_prob_eval.shape[-1])
                    cap_probs['cap_prob_eval'] = cap_prob_eval

        if self.opt.caption_cost_type == 'loss':
            cap_prob = cap_prob.reshape(-1, cap_prob.shape[-2], cap_prob.shape[-1]) # [num_matched_query, max_length_sentence, num_word_in_vocab], e.g., [5, 13, 1608]
            caption_tensor = dt['cap_tensor'][:, 1:][cap_bigids] # [num_sentence, max_num_sentence], e.g, [5, 13]
            caption_mask = dt['cap_mask'][:, 1:][cap_bigids]  # [num_sentence, max_num_sentence], e.g, [5, 13]
            cap_loss = cap_head.build_loss(cap_prob, caption_tensor, caption_mask) # [num_query]
            cap_cost = cap_loss
        else:
            raise AssertionError('caption cost type error')

        # Calculate caption probs for each query
        # breakpoint()
        # if self.opt.refine_pseudo_box:
        #     sentence_cap_prob = cap_head.build_prob(raw_cap_prob, caption_tensor, caption_mask)
        # else:
        sentence_cap_prob = - cap_loss

        if indices:
            return cap_loss.mean(), cap_probs, seq, sentence_cap_prob
        # cap_loss.mean(): [num_matched_query] --> [1], 
        # cap_probs: dict, contains 'cap_prob_train' or 'cap_prob_eval' [num_matched_query, max_length_sentence, num_word_in_vocab]
        # seq： [num_sentence, max_length_sentence+1], here the '+1' means the 1st col is all '0'
 
        cap_id, query_id = cap_bigids, feat_bigids
        cost_caption = hs_r.new_zeros((max(query_id) + 1, max(cap_id) + 1))
        cost_caption[query_id, cap_id] = cap_cost
        loss_caption = hs_r.new_zeros((max(query_id) + 1, max(cap_id) + 1))
        loss_caption[query_id, cap_id] = cap_loss
        cost_caption = cost_caption.reshape(-1, N_q,
                                            max(cap_id) + 1)  # batch_size * num_queries * all_caption_num
        loss_caption = loss_caption.reshape(-1, N_q, max(cap_id) + 1)
        return cost_caption, loss_caption, cap_probs, seq

    def caption_prediction_eval(self, cap_head, dt, hs, reference, others, decoder_type, pred_num=None, indices=None):
        assert indices == None
        N_, N_q, C = hs.shape
        query_mask = others['proposals_mask']
        gt_mask = dt['gt_boxes_mask']
        mix_mask = torch.zeros(query_mask.sum().item(), gt_mask.sum().item())
        query_nums, gt_nums = query_mask.sum(1).cpu(), gt_mask.sum(1).cpu()
        hs_r = torch.masked_select(hs, query_mask.unsqueeze(-1)).reshape(-1, C)

        row_idx, col_idx = 0, 0
        for i in range(N_):
            mix_mask[row_idx: (row_idx + query_nums[i]), col_idx: (col_idx + gt_nums[i])] = 1
            row_idx = row_idx + query_nums[i]
            col_idx = col_idx + gt_nums[i]

        cap_probs = {}

        if decoder_type in ['none']:
            cap_probs['cap_prob_train'] = torch.zeros(1, device=hs.device)
            cap_probs['cap_prob_eval'] = torch.zeros(N_, N_q, 3, device=hs.device)
            seq = torch.zeros(N_, N_q, 3, device=hs.device)
            return cap_probs, seq

        elif decoder_type in ['light']:
            clip = hs_r.unsqueeze(1)
            clip_mask = clip.new_ones(clip.shape[:2])
            event = None
            seq, cap_prob_eval = cap_head.sample(event, clip, clip_mask)
            if len(seq):
                seq = seq.reshape(-1, N_q, seq.shape[-1])
                cap_prob_eval = cap_prob_eval.reshape(-1, N_q, cap_prob_eval.shape[-1])
            cap_probs['cap_prob_eval'] = cap_prob_eval

        elif decoder_type in ['standard']:
            assert N_ == 1, 'only support batchsize = 1'
            with torch.no_grad():
                if self.opt.transformer_input_type == 'prior_proposals':
                    # hs: [bs, num_query, feat_dim]  
                    # reference: [bs, num_query, 2]
                    if pred_num:
                        num_cap =  pred_num
                    else:
                        num_cap =  dt['cap_tensor'].shape[0]
                    interval = N_q // num_cap
                    pool_layer = torch.nn.AvgPool1d(interval,stride=interval)
                    hs = pool_layer(hs.permute(0,2,1)).permute(0,2,1)[:,:num_cap,:] # [batch, num_sentence, dim]
                    reference = pool_layer(reference.permute(0,2,1)).permute(0,2,1)[:,:num_cap,:] # # [batch, num_sentence, 2]
                    seq, cap_prob_eval = cap_head.sample(hs, reference, others)
                    if len(seq):
                        seq = seq.reshape(-1, num_cap, seq.shape[-1]) #
                        cap_prob_eval = cap_prob_eval.reshape(-1, num_cap, cap_prob_eval.shape[-1])
                    cap_probs['cap_prob_eval'] = cap_prob_eval
                else:
                    seq, cap_prob_eval = cap_head.sample(hs, reference, others)
                    if len(seq):
                        seq = seq.reshape(-1, N_q, seq.shape[-1]) #
                        cap_prob_eval = cap_prob_eval.reshape(-1, N_q, cap_prob_eval.shape[-1])
                    cap_probs['cap_prob_eval'] = cap_prob_eval
        return cap_probs, seq


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

    @torch.no_grad()
    def forward(self, outputs, target_sizes, loader):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size] containing the size of each video of the batch
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        N, N_q, N_class = out_logits.shape
        assert len(out_logits) == len(target_sizes)
        prob = out_logits.sigmoid() # batch, num_queries, 1

        if self.opt.transformer_input_type == 'prior_proposals':
            #topk_values = prob.view(N, N_q)
            #topk_indexes = torch.arange(N_q, device=prob.device).unsqueeze(0).repeat(N, 1)
            topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), outputs['seq'].shape[1], dim=1)
        else:
            topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), N_q, dim=1)
        scores = topk_values
        # topk_boxes = topk_indexes // out_logits.shape[2]
        topk_boxes = torch.div(topk_indexes, out_logits.shape[2], rounding_mode='floor')
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cl_to_xy(out_bbox)
        raw_boxes = copy.deepcopy(boxes)
        boxes[boxes < 0] = 0
        boxes[boxes > 1] = 1
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 2))

        scale_fct = torch.stack([target_sizes, target_sizes], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        seq = outputs['seq']  # [batch_size, num_queries, max_Cap_len=30]
        cap_prob = outputs['caption_probs']['cap_prob_eval']  # [batch_size, num_queries]
        eseq_lens = outputs['pred_count'].argmax(dim=-1).clamp(min=1)

        if len(seq):
            mask = (seq > 0).float()
            # cap_scores = (mask * cap_prob).sum(2).cpu().numpy().astype('float') / (
            #         1e-5 + mask.sum(2).cpu().numpy().astype('float'))
            cap_scores = (mask * cap_prob).sum(2).cpu().numpy().astype('float')
            seq = seq.detach().cpu().numpy().astype('int')  # (eseq_batch_size, eseq_len, cap_len)
            caps = [[loader.dataset.translator.rtranslate(s) for s in s_vid] for s_vid in seq]
            if self.opt.transformer_input_type != 'prior_proposals':
                caps = [[caps[batch][idx] for q_id, idx in enumerate(b)] for batch, b in enumerate(topk_boxes)]  # Re-arrange the caption order accroding to the logits
                cap_scores = [[cap_scores[batch, idx] for q_id, idx in enumerate(b)] for batch, b in enumerate(topk_boxes)]
        else:
            bs, num_queries = boxes.shape[:2]
            cap_scores = [[-1e5] * num_queries] * bs
            caps = [[''] * num_queries] * bs

        results = [
            {'scores': s, 'labels': l, 'boxes': b, 'raw_boxes': b, 'captions': c, 'caption_scores': cs, 'query_id': qid,
             'vid_duration': ts, 'pred_seq_len': sl} for s, l, b, rb, c, cs, qid, ts, sl in
            zip(scores, labels, boxes, raw_boxes, caps, cap_scores, topk_boxes, target_sizes, eseq_lens)]
        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    device = torch.device(args.device)
    base_encoder = build_base_encoder(args)
    # For text encoder when using DTW matcher
    # if args.matcher_type == 'DTW' or args.use_pseudo_box:
    #     if args.pretrained_language_model == 'UniVL':
    #         print('Load pretrained UniVL model weights')
    #         text_encoder = load_pretrained_UniVL()
    #     else:
    #         for i in range(10):
    #             try:
    #                 text_encoder = AutoModel.from_pretrained(args.pretrained_language_model, cache_dir=args.huggingface_cache_dir)
    #                 break
    #             except:
    #                 print('download error in AutoModel, retry...')
    #                 time.sleep(1)
    # else:
    #     text_encoder = None

    transformer = build_deforamble_transformer(args)
    captioner = build_captioner(args)

    model = PDVC(
        base_encoder,
        transformer,
        captioner,
        num_classes=args.num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        opt=args
    )

    matcher = build_matcher(args)
    if args.matcher_type == 'DTW' and args.use_anchor:
        weight_dict = {'loss_ce': args.cls_loss_coef,
                    'loss_bbox': args.bbox_loss_coef,
                    'loss_giou': args.giou_loss_coef,
                    'loss_self_iou': args.self_iou_loss_coef,
                    'loss_ref_rank': args.ref_rank_loss_coef,
                    'loss_counter': args.count_loss_coef,
                    'loss_caption': args.caption_loss_coef,
                    'contrastive_loss': args.contrastive_loss_start_coef,
                    }
    else:
        weight_dict = {'loss_ce': args.cls_loss_coef,
                    'loss_bbox': args.bbox_loss_coef,
                    'loss_giou': args.giou_loss_coef,
                    'loss_counter': args.count_loss_coef,
                    'loss_caption': args.caption_loss_coef,
                    'contrastive_loss': args.contrastive_loss_start_coef,
                    }
    if args.refine_pseudo_box:
        weight_dict.update({'loss_mil': args.mil_loss_coef})
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']

    if args.matcher_type == 'DTW' or args.matcher_type == 'Sim':
        criterion = AlignCriterion(args.num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha,
                                focal_gamma=args.focal_gamma, opt=args)
        contrastive_criterion = ContrastiveCriterion(temperature=args.contrastive_loss_temperature,
                                                 enable_cross_video_cl=args.enable_cross_video_cl,
                                                 enable_e2t_cl = args.enable_e2t_cl,
                                                 enable_bg_for_cl = args.enable_bg_for_cl)
        contrastive_criterion.to(device)
    else:
        criterion = SetCriterion(args.num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha,
                                focal_gamma=args.focal_gamma, opt=args)
        contrastive_criterion = None
    
    criterion.to(device)
    postprocessors = {'bbox': PostProcess(args)}

    return model, criterion, contrastive_criterion, postprocessors


