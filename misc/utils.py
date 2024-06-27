# coding:utf-8
# from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
import numpy as np
import glob
import shutil
import os
import colorlog
import random
import six
from six.moves import cPickle
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt


def match_name_keywords(n, name_keywords):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out


def decide_two_stage(transformer_input_type, dt, criterion):
    if transformer_input_type == 'gt_proposals':
        two_stage = True
        proposals = dt['gt_boxes']
        proposals_mask = dt['gt_boxes_mask']
        criterion.matcher.cost_caption = 0
        for q_k in ['loss_length', 'loss_ce', 'loss_bbox', 'loss_giou']:
            for key in criterion.weight_dict.keys():
                if q_k in key:
                    criterion.weight_dict[key] = 0
        disable_iterative_refine = True
    elif transformer_input_type == 'prior_proposals':
        two_stage = True
        proposals = dt['gt_boxes']
        proposals_mask = None
        criterion.matcher.cost_caption = 0
        for q_k in ['loss_length', 'loss_ce', 'loss_bbox', 'loss_giou']:
            for key in criterion.weight_dict.keys():
                if q_k in key:
                    criterion.weight_dict[key] = 0
        disable_iterative_refine = False
    elif transformer_input_type == 'queries':  #
        two_stage = False
        proposals = None
        proposals_mask = None
        disable_iterative_refine = False
    else:
        raise ValueError('Wrong value of transformer_input_type, got {}'.format(transformer_input_type))
    return two_stage, disable_iterative_refine, proposals, proposals_mask


def pickle_load(f):
    """ Load a pickle.
    Parameters
    ----------
    f: file-like object
    """
    if six.PY3:
        return cPickle.load(f, encoding='latin-1')
    else:
        return cPickle.load(f)


def pickle_dump(obj, f):
    """ Dump a pickle.
    Parameters
    ----------
    obj: pickled object
    f: file-like object
    """
    if six.PY3:
        return cPickle.dump(obj, f, protocol=2)
    else:
        return cPickle.dump(obj, f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # grid_sampler_2d_backward_cuda does not have a deterministic implementation. try set torch.use_deterministic_algorithms(True, warn_only=True) to see the non-deterministic operation
    # torch.use_deterministic_algorithms(True, warn_only=True)


def update_values(dict_from, dict_to):
    for key, value in dict_from.items():
        if key not in dict_to.keys():
            raise AssertionError('key mismatching: {}'.format(key))
        if isinstance(value, dict):
            update_values(dict_from[key], dict_to[key])
        elif value is not None:
            dict_to[key] = dict_from[key]


def print_opt(opt, model, logger):
    print_alert_message('All args:', logger)
    for key, item in opt._get_kwargs():
        logger.info('{} = {}'.format(key, item))
    print_alert_message('Model structure:', logger)
    logger.info(model)


def build_folder_name(opt):
    # The dataset
    # breakpoint()
    if len(opt.visual_feature_folder) == 2:
        if ('youcook2' in opt.visual_feature_folder[1]) or ('yc2' in opt.visual_feature_folder[1]):
            dataset_name = 'howto-yc2_yc2'
        elif ('Tasty' in opt.visual_feature_folder[1]) or ('tasty' in opt.visual_feature_folder[1]):
            dataset_name = 'howto-tasty_tasty'
        elif ('anet' in opt.visual_feature_folder[1]) or ('Anet' in opt.visual_feature_folder[1]):
            dataset_name = 'howto-anet_anet'
        # elif ('vlep' in opt.visual_feature_folder[1]) or ('Vlep' in opt.visual_feature_folder[1]):
        #     dataset_name = 'howto-vlep_vlep'
        else:
            raise ValueError('Wrong dataset name')
        
        if 'vlep' in opt.visual_feature_folder[0] or 'Vlep' in opt.visual_feature_folder[0]:
            dataset_name = dataset_name.replace('howto', 'vlep')
    else:
        if ('youcook2' in opt.visual_feature_folder[0]) or ('yc2' in opt.visual_feature_folder[0]):
            dataset_name = 'yc2'
        elif ('Anet' in opt.visual_feature_folder[0]) or ('anet' in opt.visual_feature_folder[0]):
            dataset_name = 'anet'
        elif ('Tasty' in opt.visual_feature_folder[0]) or ('tasty' in opt.visual_feature_folder[0]):
            dataset_name = 'tasty'
        elif ('Howto' in opt.visual_feature_folder[0]) or ('howto' in opt.visual_feature_folder[0]):
            if ('yc2' in opt.visual_feature_folder_val[0]) or ('youcook2' in opt.visual_feature_folder_val[0]):
                dataset_name = 'howto_yc2'
            elif 'tasty' in opt.visual_feature_folder_val[0] or 'Tasty' in opt.visual_feature_folder_val[0]:
                dataset_name = 'howto_tasty'
            elif 'anet' in opt.visual_feature_folder_val[0] or 'Anet' in opt.visual_feature_folder_val[0]:
                dataset_name = 'howto_anet'
        elif ('vlep' in opt.visual_feature_folder[0]) or ('Vlep' in opt.visual_feature_folder[0]):
            if ('yc2' in opt.visual_feature_folder_val[0]) or ('youcook2' in opt.visual_feature_folder_val[0]):
                dataset_name = 'vlep_yc2'
            elif 'tasty' in opt.visual_feature_folder_val[0] or 'Tasty' in opt.visual_feature_folder_val[0]:
                dataset_name = 'vlep_tasty'
            elif 'anet' in opt.visual_feature_folder_val[0] or 'Anet' in opt.visual_feature_folder_val[0]:
                dataset_name = 'vlep_anet'
        else:
            raise ValueError('Wrong dataset name')
    if 'tasty_14' in opt.dict_file:
        dataset_name += '_voc14'
    
    # The code base
    if opt.use_anchor:
        use_anchor = 'anc' # Means learnable anchor is used
    else:
        use_anchor = 'ori' # Means original anchor in pdvc is used

    # The state of using pseudo boxes
    if opt.use_pseudo_box:
        use_pseudo = 'pbox'
        if opt.pseudo_box_type == 'similarity':
            use_pseudo += '(sim)'
        else:
            use_pseudo += '({})'.format(opt.pseudo_box_type)
    else:
        use_pseudo = 'GT'

    # The viusal-text model used
    if opt.pretrained_language_model == 'CLIP-ViP':
        text_model = 'ViP'
    elif opt.pretrained_language_model == 'UniVL':
        text_model = 'Uni'
    else:
        text_model = opt.pretrained_language_model
    
    format_folder_name = '_'.join([dataset_name, use_anchor, use_pseudo, text_model])
    


    return format_folder_name

def build_folder(opt):
    # breakpoint()
    if opt.start_from:
        print('Start training from id:{}'.format(opt.start_from))
        save_folder = os.path.join(opt.save_dir, opt.start_from)
        assert os.path.exists(save_folder) and os.path.isdir(save_folder), 'Wrong start_from path: {}'.format(save_folder)
    else:
        if not os.path.exists(opt.save_dir):
            os.mkdir(opt.save_dir)
        format_folder_name = build_folder_name(opt)
        # breakpoint()
        save_foldername = ''
        if opt.use_pseudo_box:
            if opt.pseudo_box_type != 'align':
                if opt.pseudo_box_type == 'similarity_op' or opt.pseudo_box_type == 'similarity_op_order':
                    save_foldername = '{}_topf{}_beta{}_iter{}_r{}'.format(opt.pseudo_box_type, opt.top_frames, opt.beta, opt.iteration, opt.width_ratio)
                elif opt.pseudo_box_type == 'similarity_op_order_v2':
                    save_foldername = '{}_topf{}_iter{}_r{}_th{}'.format(opt.pseudo_box_type, opt.top_frames, opt.iteration, opt.width_ratio, opt.width_th)
                else:
                    save_foldername = '{}_topf{}_w{}_{}_r{}'.format(opt.pseudo_box_type, opt.top_frames, opt.window_size, opt.statistic_mode, opt.width_ratio)
            else:
                save_folder = 'align'
        else:
            save_foldername = 'gtbox'

        if opt.refine_pseudo_box:
            save_foldername += '_refine_aug({},{})_top{}_{}stage'.format(opt.pseudo_box_aug_num, \
                                                                        opt.pseudo_box_aug_ratio, \
                                                                        opt.merge_k_boxes, \
                                                                        opt.refine_pseudo_stage_num)
            if opt.pseudo_box_aug_mode == 'uniform':
                save_foldername += '_uniform'
            elif opt.pseudo_box_aug_mode == 'random_new':
                save_foldername += '_random_new'
            save_foldername += ('_' + opt.merge_criterion)
            if opt.merge_mode == 'interpolate':
                save_foldername += '_interpolate'
            if opt.use_neg_pseudo_box:
                save_foldername += '_{}neg'.format(opt.num_neg_box)
            if opt.mil_loss_coef != 1.0:
                save_foldername += '_mil_coef{}'.format(str(opt.mil_loss_coef))
            if opt.weighted_mil_loss:
                save_foldername += '_wMIL'
            if not opt.focal_mil:
                save_foldername += '_noFocal'
            if opt.disable_rematch:
                save_foldername += '_nomatch'
            if opt.use_additional_score_layer:
                save_foldername += '_S-layer'
            if opt.use_additional_cap_layer:
                save_foldername += '_C-layer'


        if opt.id != '':
            save_foldername += '_{}'.format(opt.id)
        # breakpoint()
        # basefilename = os.path.basename(opt.cfg_path)
        # basefilename = os.path.splitext(basefilename)[0]
        save_folder = os.path.join(opt.save_dir, format_folder_name)
        save_folder = os.path.join(save_folder, save_foldername)
        if os.path.exists(save_folder):
            print('Results folder "{}" already exists, renaming it...'.format(save_folder))
            i = 1
            while 1:
                new_save_folder = save_folder + '_{}'.format(i)
                if not os.path.exists(new_save_folder):
                    save_folder = new_save_folder
                    break
                i += 1
            # wait_flag = input('Warning! Path {} already exists, rename it? (Y/N) : '.format(save_folder))
            # if wait_flag in ['Y', 'y']:
            #     # opt.id = opt.id + '_{}'.format(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
            #     # save_folder = os.path.join(opt.save_dir, opt.id)
            #     # print('Rename opt.id as "{}".'.format(opt.id))
            #     new_name = input('the new name to be appended :')
            #     save_folder = save_folder + '_' + new_name
            # # elif wait_flag in ['N', 'n']:
            # #     wait_flag_new = input('Are you sure re-write this folder:{}? (Y/N): '.format(save_folder))
            # #     if wait_flag_new in ['Y', 'y']:
            # #         return save_folder
            # #     else:
            # #         raise AssertionError('Folder {} already exists'.format(save_folder))
            # else:
            #     raise AssertionError('Folder {} already exists'.format(save_folder))
        print('Results folder "{}" does not exist, creating folder...'.format(save_folder))
        os.makedirs(save_folder)
        os.makedirs(os.path.join(save_folder, 'prediction'))
    return save_folder


def backup_envir(save_folder, opt):
    cfg_path = opt.cfg_path
    dir_path = os.path.dirname(cfg_path)
    backup_folders = ['cfgs_base', 'cfgs', 'misc', 'pdvc']
    if dir_path not in backup_folders:
        backup_folders.append(dir_path)

    backup_files = glob.glob('./*.py')
    for folder in backup_folders:
        shutil.copytree(folder, os.path.join(save_folder, 'backup', folder))
    for file in backup_files:
        shutil.copyfile(file, os.path.join(save_folder, 'backup', file))


def create_logger(folder, filename):
    log_colors = {
        'DEBUG': 'blue',
        'INFO': 'white',
        'WARNING': 'green',
        'ERROR': 'red',
        'CRITICAL': 'yellow',
    }

    import logging
    logger = logging.getLogger('DVC')
    # %(filename)s$RESET:%(lineno)d
    # LOGFORMAT = "%(log_color)s%(asctime)s [%(log_color)s%(filename)s:%(lineno)d] | %(log_color)s%(message)s%(reset)s |"
    LOGFORMAT = ""
    LOG_LEVEL = logging.DEBUG
    logging.root.setLevel(LOG_LEVEL)
    stream = logging.StreamHandler()
    stream.setLevel(LOG_LEVEL)
    stream.setFormatter(colorlog.ColoredFormatter(LOGFORMAT, datefmt='%d %H:%M', log_colors=log_colors))

    # print to log file
    hdlr = logging.FileHandler(os.path.join(folder, filename))
    hdlr.setLevel(LOG_LEVEL)
    # hdlr.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
    hdlr.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(hdlr)
    logger.addHandler(stream)
    return logger


def print_alert_message(str, logger=None):
    msg = '*' * 20 + ' ' + str + ' ' + '*' * (58 - len(str))
    if logger:
        logger.info('\n\n' + msg)
    else:
        print(msg)


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for i, param in enumerate(group['params']):
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


if __name__ == '__main__':
    # import opts
    #
    # info = {'opt': vars(opts.parse_opts()),
    #         'loss': {'tap_loss': 0, 'tap_reg_loss': 0, 'tap_conf_loss': 0, 'lm_loss': 0}}
    # record_this_run_to_csv(info, 'save/results_all_runs.csv')

    logger = create_logger('./', 'mylogger.log')
    logger.info('debug')
    logger.info('test2')
