# coding:utf-8

''' 
similar to train_ft_gt.py. it fine-tunes the model on the target dataset with ground-truth annotations. but the pretrain data includes both pretrain and target data (only use captions)

set pretrain_data_mode to 'single', it is same as train_ft_gt.py.

使用全部的howto subset数据进行pretrain， 然后用部分的gt数据进行fine-tune
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import time
import torch
import os
import sys
import collections
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from os.path import dirname, abspath
import re 

pdvc_dir = dirname(abspath(__file__))
sys.path.insert(0, pdvc_dir)
sys.path.insert(0, os.path.join(pdvc_dir, 'densevid_eval3'))
sys.path.insert(0, os.path.join(pdvc_dir, 'densevid_eval3/SODA'))
# print(sys.path)


os.environ["TOKENIZERS_PARALLELISM"] = "false" # To avoid warning of tokenizer
from eval_utils import evaluate
import opts
from tensorboardX import SummaryWriter
from misc.utils import print_alert_message, build_folder, create_logger, backup_envir, print_opt, set_seed
from data.video_dataset import PropSeqDataset, collate_fn, PercentageSubsetDataset
from pdvc.pdvc import build
from collections import OrderedDict
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import copy

a100_folder = ['/cpfs01/shared/Gvlab-A100/Gvlab-A100_hdd/wuhao/youcook2', '/cpfs01/shared/Gvlab-A100/Gvlab-A100_hdd/wuhao/Tasty/features', '/cpfs01/shared/Gvlab-A100/Gvlab-A100_hdd/huabin/dataset/Tasty/UniVL_feature', '/cpfs01/shared/Gvlab-A100/Gvlab-A100_hdd/huabin/dataset/Anet', '/cpfs01/shared/Gvlab-A100/Gvlab-A100_hdd/wuhao/howto100m/features']
r3090_folder = ['/mnt/data/Gvlab/wuhao/features/yc2', '/mnt/data/Gvlab/wuhao/features/tasty', '/mnt/data/Gvlab/wuhao/features/tasty/univl', '/mnt/data/Gvlab/wuhao/features/anet', '/mnt/data/Gvlab/wuhao/features/howto100m']

pretrain_data_mode = 'mix' # 'mix' or 'seq' or 'single'

# /cpfs01/shared/Gvlab-A100/Gvlab-A100_hdd/wuhao/howto100m/features -> /mnt/data/Gvlab/wuhao/features/howto100m
# /cpfs01/shared/Gvlab-A100/Gvlab-A100_hdd/wuhao/howto100m/features/clip -> /mnt/data/Gvlab/wuhao/features/howto100m/clip_features
# /cpfs01/shared/Gvlab-A100/Gvlab-A100_hdd/wuhao/howto100m/features/UniVL -> /mnt/data/Gvlab/wuhao/features/howto100m/univl_features

def _init_fn(worker_id):
    np.random.seed(12 + worker_id)


def map_path(path):
    path_backup = copy.deepcopy(path)
    # breakpoint()
    for i, folder in enumerate(a100_folder):
        if folder in path:
            path = path.replace(folder, r3090_folder[i])
            
            
    if path == path_backup:
        if path.startswith('/mnt/data'):
            pass
        else:
            # path = '/mnt' + path[6:]
            print('map failed')
            exit(1)
    return path


def train(opt):
    set_seed(opt.seed)
    save_folder = build_folder(opt)
    opt.epoch = 20
    opt.use_pseudo_box = False
    opt.refine_pseudo_box = False
    opt.pseudo_box_aug = False
    # breakpoint()

    # breakpoint()
    if 'howto-tasty_tasty' in save_folder:
        if pretrain_data_mode == 'mix':
            checkpoint_folder = re.sub(r"_seq2-ft.*", "", save_folder)
        elif pretrain_data_mode == 'seq':
            checkpoint_folder = re.sub(r"_seq2-ft.*", "_seq-train", save_folder) # .replace('_seq2-ft', '')
        elif pretrain_data_mode == 'single':
            checkpoint_folder = re.sub(r"_seq2-ft.*", "", save_folder.replace('howto-tasty_tasty', 'howto_tasty')) # .replace('_seq2-ft', '')
    elif 'howto-yc2_yc2' in save_folder:
        if pretrain_data_mode == 'mix':
            checkpoint_folder = re.sub(r"_seq2-ft.*", "", save_folder)
        elif pretrain_data_mode == 'seq':
            checkpoint_folder = re.sub(r"_seq2-ft.*", "_seq-train", save_folder)
        elif pretrain_data_mode == 'single':
            checkpoint_folder = re.sub(r"_seq2-ft.*", "", save_folder.replace('howto-yc2_yc2', 'howto_yc2')) # .replace('_seq2-ft', '')
    elif 'vlep-yc2_yc2' in save_folder:
        if pretrain_data_mode == 'mix':
            checkpoint_folder = re.sub(r"_seq2-ft.*", "", save_folder)
        elif pretrain_data_mode == 'seq':
            checkpoint_folder = re.sub(r"_seq2-ft.*", "_seq-train", save_folder)
        elif pretrain_data_mode == 'single':
            checkpoint_folder = re.sub(r"_seq2-ft.*", "", save_folder.replace('vlep-yc2_yc2', 'vlep_yc2')) # .replace('_seq2-ft', '')
    elif 'howto-anet_anet' in save_folder:
        if pretrain_data_mode == 'mix':
            checkpoint_folder = re.sub(r"_seq2-ft.*", "", save_folder)
        elif pretrain_data_mode == 'seq':
            checkpoint_folder = re.sub(r"_seq2-ft.*", "_seq-train", save_folder)
        elif pretrain_data_mode == 'single':
            checkpoint_folder = re.sub(r"_seq2-ft.*", "", save_folder.replace('howto-anet_anet', 'howto_anet'))
    
    else:
        print('the script only support settings howto-XXX_XXX')
        exit(1)
    # breakpoint()

    if opt.id_ori != '':
        checkpoint_folder = checkpoint_folder + '_' + opt.id_ori
    # breakpoint()
    # if opt.id == "":
    #     pass
    # else:
    #     checkpoint_folder = checkpoint_folder + '_' + opt.id

    if not os.path.exists(checkpoint_folder) and not os.path.exists(checkpoint_folder + '_es20'):
        print('the checkpoint folder {} does not exist'.format(checkpoint_folder))
        exit(1)
    else:
        if not os.path.exists(os.path.join(checkpoint_folder, 'val.log')):
            # print('the checkpoint folder has no val.log, denoting the setting is not fully trained')
            for i in range(1, 100):
                if os.path.exists(f'{checkpoint_folder}_{i}'):
                    if os.path.exists(os.path.join(f'{checkpoint_folder}_{i}', 'val.log')):
                        checkpoint_folder = f'{checkpoint_folder}_{i}'
                        break
                    else:
                        continue
                else:
                    print(f'{checkpoint_folder}_{i} does not exist')
                    print('the checkpoint folder does not exist')
                    exit(1)

    logger = create_logger(save_folder, 'train.log')
    tf_writer = SummaryWriter(os.path.join(save_folder, 'tf_summary'))

    if not opt.start_from:
        backup_envir(save_folder, opt)
        logger.info('backup evironment completed !')

    saved_info = {'best': {}, 'last': {}, 'history': {}, 'eval_history': {}}

    # # continue training
    # if opt.start_from:
    #     opt.pretrain = False
    #     infos_path = os.path.join(save_folder, 'info.json')
    #     with open(infos_path) as f:
    #         logger.info('Load info from {}'.format(infos_path))
    #         saved_info = json.load(f)
    #         prev_opt = saved_info[opt.start_from_mode[:4]]['opt']

    #         exclude_opt = ['start_from', 'start_from_mode', 'pretrain']
    #         for opt_name in prev_opt.keys():
    #             if opt_name not in exclude_opt:
    #                 vars(opt).update({opt_name: prev_opt.get(opt_name)})
    #             if prev_opt.get(opt_name) != vars(opt).get(opt_name):
    #                 logger.info('Change opt {} : {} --> {}'.format(opt_name, prev_opt.get(opt_name),
    #                                                                vars(opt).get(opt_name)))
    if len(opt.visual_feature_folder) == 2:
        # train_dataset_pretrain = PropSeqDataset(opt.train_caption_file[0],
        #                                     [opt.visual_feature_folder[0]],
        #                                     [opt.text_feature_folder[0]],
        #                                     opt.dict_file, True, 'gt',
        #                                     opt)
        train_dataset_target = PropSeqDataset(opt.train_caption_file[1],
                                            [opt.visual_feature_folder[1]],
                                            [opt.text_feature_folder[1]],
                                            opt.dict_file, True, 'gt',
                                            opt)
        subset_data = PercentageSubsetDataset(train_dataset_target, opt.ft_gt_percent)
        # train_loader_pretrain = DataLoader(train_dataset_pretrain, batch_size=opt.batch_size,
        #                       shuffle=True, num_workers=opt.nthreads, collate_fn=collate_fn, worker_init_fn=_init_fn)
        train_loader_target = DataLoader(subset_data, batch_size=opt.batch_size,
                                    shuffle=True, num_workers=opt.nthreads, collate_fn=collate_fn, worker_init_fn=_init_fn)
        
        # train_dataloaders = [train_loader_pretrain, train_loader_target]
        # train_dataset = torch.utils.data.ConcatDataset([train_dataset_1, train_dataset_2])
        # train_dataset.translator = train_dataset_1.translator

    else:
        # print('the script only support two dataset for pretrain and target task respectively')
        # exit(1)
        train_dataset_target = PropSeqDataset(opt.train_caption_file,
                                    opt.visual_feature_folder,
                                    opt.text_feature_folder,
                                    opt.dict_file, True, 'gt',
                                    opt)
        subset_data = PercentageSubsetDataset(train_dataset_target, opt.ft_gt_percent)
        train_loader_target = DataLoader(subset_data, batch_size=opt.batch_size,
                                shuffle=True, num_workers=opt.nthreads, collate_fn=collate_fn, worker_init_fn=_init_fn)
        # train_dataloaders = [train_loader_target]

    # val_dataset = PropSeqDataset(opt.val_caption_file,
    #                              opt.visual_feature_folder,
    #                              opt.text_feature_folder,
    #                              opt.dict_file, False, 'gt',
    #                              opt)
    if not hasattr(opt, 'dict_file_val'):
        opt.dict_file_val = opt.dict_file
        opt.vocab_size_val = opt.vocab_size

    val_dataset = PropSeqDataset(opt.val_caption_file,
                                opt.visual_feature_folder_val,
                                opt.text_feature_folder_val,
                                opt.dict_file, False, 'gt',
                                opt)


    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size_for_eval,
                            shuffle=False, num_workers=opt.nthreads, collate_fn=collate_fn, worker_init_fn=_init_fn)

    epoch = saved_info[opt.start_from_mode[:4]].get('epoch', 0)
    iteration = saved_info[opt.start_from_mode[:4]].get('iter', 0)
    best_val_score = saved_info[opt.start_from_mode[:4]].get('best_val_score', -1e5)
    val_result_history = saved_info['history'].get('val_result_history', {})
    loss_history = saved_info['history'].get('loss_history', {})
    lr_history = saved_info['history'].get('lr_history', {})
    opt.current_lr = vars(opt).get('current_lr', opt.lr)

    # Build model

    model, criterion, contrastive_criterion, postprocessors = build(opt)
    model.translator = train_dataset_target.translator
    model.train()


    # load pretrained model

    # breakpoint()
    # load pretrained model
    model_pth = torch.load(os.path.join(checkpoint_folder, 'model-best.pth'))
    logger.info('Loading pth from {}'.format(checkpoint_folder))
    model.load_state_dict(model_pth['model'], strict=False)
    

    # # Recover the parameters
    # if opt.start_from and (not opt.pretrain):
    #     if opt.start_from_mode == 'best':
    #         model_pth = torch.load(os.path.join(save_folder, 'model-best.pth'))
    #     elif opt.start_from_mode == 'last':
    #         model_pth = torch.load(os.path.join(save_folder, 'model-last.pth'))
    #     logger.info('Loading pth from {}, iteration:{}'.format(save_folder, iteration))
    #     model.load_state_dict(model_pth['model'])

    # # Load the pre-trained model
    # if opt.pretrain and (not opt.start_from):
    #     logger.info('Load pre-trained parameters from {}'.format(opt.pretrain_path))
    #     model_pth = torch.load(opt.pretrain_path, map_location=torch.device(opt.device))
    #     # query_weight = model_pth['model'].pop('query_embed.weight')
    #     if opt.pretrain == 'encoder':
    #         encoder_filter = model.get_filter_rule_for_encoder()
    #         encoder_pth = {k:v for k,v in model_pth['model'].items() if encoder_filter(k)}
    #         model.load_state_dict(encoder_pth, strict=True)
    #     elif opt.pretrain == 'decoder':
    #         encoder_filter = model.get_filter_rule_for_encoder()
    #         decoder_pth = {k:v for k,v in model_pth['model'].items() if not encoder_filter(k)}
    #         model.load_state_dict(decoder_pth, strict=True)
    #         pass
    #     elif opt.pretrain == 'full':
    #         # model_pth = transfer(model, model_pth)
    #         model.load_state_dict(model_pth['model'], strict=True)
    #     else:
    #         raise ValueError("wrong value of opt.pretrain")
        

    model.to(opt.device)

    # Decide which parameters need to be trained
    # if (opt.matcher_type =='DTW' or opt.use_pseudo_box) and opt.text_encoder_learning_strategy == 'frozen':
    #     for _, p in model.text_encoder.named_parameters():
    #         p.requires_grad = False
    #         text_encoder_params = list(map(id, model.text_encoder.parameters()))
    #         other_params = filter(lambda p: id(p) not in text_encoder_params, model.parameters())
    # else:
    #     other_params = model.parameters()
    other_params = model.parameters()

    training_params = [{'params': other_params, 'lr': opt.lr * 0.5}]

    if opt.optimizer_type == 'adam':
        optimizer = optim.Adam(training_params, weight_decay=opt.weight_decay)

    elif opt.optimizer_type == 'adamw':
        optimizer = optim.AdamW(training_params, weight_decay=opt.weight_decay)

    milestone = [opt.learning_rate_decay_start + opt.learning_rate_decay_every * _ for _ in range(int((opt.epoch - opt.learning_rate_decay_start) / opt.learning_rate_decay_every))]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestone, gamma=opt.learning_rate_decay_rate)

    # Load tokenizer for text encoder
    # for i in range(10):
    #     try:
    #         if opt.pretrained_language_model == 'UniVL':
    #             tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    #         else:
    #             tokenizer = AutoTokenizer.from_pretrained(opt.pretrained_language_model)
    #         break
    #     except:
    #         print('download error in AutoTokenizer, retry...')
    #         time.sleep(1)

    # if opt.start_from:
    # breakpoint()
    # optimizer.load_state_dict(model_pth['optimizer'], strict=False)
        # lr_scheduler.step(epoch-1)

    # print the args for debugging  
    print_opt(opt, model, logger)
    print_alert_message('Strat training !', logger)

    loss_sum = OrderedDict()
    bad_video_num = 0

    start = time.time()

    weight_dict = criterion.weight_dict
    logger.info('loss type: {}'.format(weight_dict.keys()))
    logger.info('loss weights: {}'.format(weight_dict.values()))

    # breakpoint()

    # Epoch-level iteration
    # opt.use_pseudo_box = False

    while True:
        if True:
            # scheduled sampling rate update
            if epoch > opt.scheduled_sampling_start >= 0:
                frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                opt.ss_prob = min(opt.basic_ss_prob + opt.scheduled_sampling_increase_prob * frac,
                                  opt.scheduled_sampling_max_prob)
                model.caption_head.ss_prob = opt.ss_prob

            print('lr:{}'.format(float(opt.current_lr)))
            pass

        # breakpoint()
        # Batch-level iteration
        # for train_loader in train_dataloaders:
        trained_samples = 0
        for dt in tqdm(train_loader_target, disable=opt.disable_tqdm):
            # # # for fast debugging
            # if trained_samples > 5:
            #     break
            # else:
            #     trained_samples += 1
            if opt.device=='cuda':
                torch.cuda.synchronize(opt.device)
            if opt.debug:
                # each epoch contains less mini-batches for debugging
                if (iteration + 1) % 5 == 0:
                    iteration += 1
                    break
            iteration += 1

            optimizer.zero_grad()
            dt = {key: _.to(opt.device) if isinstance(_, torch.Tensor) else _ for key, _ in dt.items()}
            dt['video_target'] = [
                {key: _.to(opt.device) if isinstance(_, torch.Tensor) else _ for key, _ in vid_info.items()} for vid_info in
                dt['video_target']]

            # Add text encoder
            # if opt.matcher_type == 'DTW' or opt.use_pseudo_box:
            #     captions = list()
            #     for video_sents in dt['cap_raw']:  # dt['cap_raw']: [[sent_1, sent_2, ..., sent_n]]
            #         captions.extend(video_sents) 
            #     text_encoder_input = tokenizer(captions, return_tensors='pt', truncation=True, padding=True, max_length=opt.max_text_input_len)
            #     text_encoder_input = {key: _.to(opt.device) if isinstance(_, torch.Tensor) else _ for key, _ in text_encoder_input.items()} 
            #     # text_encoder_input: {'input_ids': tensor([[  101,  1996,  2307,  ...,     0,     0,     0],...]),  'attention_mask': tensor([[1, 1, 1,  ..., 0, 0, 0],...])}
            #     # len(text_encoder_input['input_ids']) = n * max_text_input_len
            #     dt['text_encoder_input'] = text_encoder_input

            # dt = collections.defaultdict(lambda: None, dt) # Commented to 

            output, loss = model(dt, criterion, contrastive_criterion)
            final_loss = sum(loss[k] * weight_dict[k] for k in loss.keys() if k in weight_dict)
            # breakpoint()
            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)

            optimizer.step()

            for loss_k,loss_v in loss.items():
                loss_sum[loss_k] = loss_sum.get(loss_k, 0)+ loss_v.item()
            loss_sum['total_loss'] = loss_sum.get('total_loss', 0) + final_loss.item()

            if opt.device=='cuda':
                torch.cuda.synchronize()

            losses_log_every = int(len(train_loader_target) / 10)

            if opt.debug:
                losses_log_every = 6

            if iteration % losses_log_every == 0:
                end = time.time()
                for k in loss_sum.keys():
                    loss_sum[k] = np.round(loss_sum[k] /losses_log_every, 3).item()

                logger.info(
                    "ID {} iter {} (epoch {}), \nloss = {}, \ntime/iter = {:.3f}, bad_vid = {:.3f}"
                        .format(opt.id, iteration, epoch, loss_sum,
                                (end - start) / losses_log_every, bad_video_num))

                tf_writer.add_scalar('lr', opt.current_lr, iteration)
                for loss_type in loss_sum.keys():
                    tf_writer.add_scalar(loss_type, loss_sum[loss_type], iteration)
                loss_history[iteration] = loss_sum
                lr_history[iteration] = opt.current_lr
                loss_sum = OrderedDict()
                start = time.time()
                bad_video_num = 0
                torch.cuda.empty_cache()

        # evaluation
        if (epoch % opt.save_checkpoint_every == 0) and (epoch >= opt.min_epoch_when_save):
            
            # Save model
            saved_pth = {'epoch': epoch,
                         'model': model.state_dict(),
                         'optimizer': optimizer.state_dict()}

            if opt.save_all_checkpoint:
                checkpoint_path = os.path.join(save_folder, 'model_iter_{}.pth'.format(iteration))
            else:
                checkpoint_path = os.path.join(save_folder, 'model-last.pth')

            torch.save(saved_pth, checkpoint_path)

            model.eval()
            result_json_path = os.path.join(save_folder, 'prediction',
                                         'num{}_epoch{}.json'.format(
                                             len(val_dataset), epoch))
            #eval_score, eval_loss = evaluate(model, criterion,  postprocessors, val_loader, result_json_path, logger=logger, args=opt, alpha=opt.ec_alpha, device=opt.device, debug=opt.debug)
            eval_score, _ = evaluate(model, criterion,  postprocessors, val_loader, result_json_path, logger=logger, args=opt, alpha=opt.ec_alpha, device=opt.device, debug=opt.debug)
            if opt.caption_decoder_type == 'none':
                current_score = 2./(1./eval_score['Precision'] + 1./eval_score['Recall'])
            else:
                if opt.criteria_for_best_ckpt == 'dvc':
                    current_score = np.array(eval_score['METEOR']).mean() + np.array(eval_score['soda_c']).mean()
                else:
                    current_score = np.array(eval_score['para_METEOR']).mean() + np.array(eval_score['para_CIDEr']).mean() + np.array(eval_score['para_Bleu_4']).mean()

            # add to tf summary
            for key in eval_score.keys():
                tf_writer.add_scalar(key, np.array(eval_score[key]).mean(), iteration)

            # Huabin comment this part for avoiding reporting losses during evaluation
            # for loss_type in eval_loss.keys():
            #     tf_writer.add_scalar('eval_' + loss_type, eval_loss[loss_type], iteration)

            _ = [item.append(np.array(item).mean()) for item in eval_score.values() if isinstance(item, list)]
            print_info = '\n'.join([key + ":" + str(eval_score[key]) for key in eval_score.keys()])
            logger.info('\nValidation results of iter {}:\n'.format(iteration) + print_info)
            logger.info('\noverall score of iter {}: {}\n'.format(iteration, current_score))
            val_result_history[epoch] = {'eval_score': eval_score}
            logger.info('Save model at iter {} to {}.'.format(iteration, checkpoint_path))

            # save the model parameter and  of best epoch
            if current_score >= best_val_score:
                best_val_score = current_score
                best_epoch = epoch
                saved_info['best'] = {'opt': vars(opt),
                                      'iter': iteration,
                                      'epoch': best_epoch,
                                      'best_val_score': best_val_score,
                                      'result_json_path': result_json_path,
                                      'avg_proposal_num': eval_score['avg_proposal_number'],
                                      'Precision': eval_score['Precision'],
                                      'Recall': eval_score['Recall']
                                      }

                # suffix = "RL" if sc_flag else "CE"
                torch.save(saved_pth, os.path.join(save_folder, 'model-best.pth'))
                logger.info('Save Best-model at iter {} to checkpoint file.'.format(iteration))

            saved_info['last'] = {'opt': vars(opt),
                                  'iter': iteration,
                                  'epoch': epoch,
                                  'best_val_score': best_val_score,
                                  }
            saved_info['history'] = {'val_result_history': val_result_history,
                                     'loss_history': loss_history,
                                     'lr_history': lr_history,
                                     # 'query_matched_fre_hist': query_matched_fre_hist,
                                     }
            with open(os.path.join(save_folder, 'info.json'), 'w') as f:
                json.dump(saved_info, f)
            logger.info('Save info to info.json')

            model.train()

        epoch += 1
        lr_scheduler.step()
        opt.current_lr = optimizer.param_groups[0]['lr']
        torch.cuda.empty_cache()
        # Stop criterion
        if epoch >= opt.epoch:
            # # load Best model and conduct evaluation
            # print('====== Conduct the Final Evaluation to test Best Checkpoint ======')
            # val_logger = create_logger(save_folder, 'val.log')
            # loaded_pth = torch.load(os.path.join(save_folder, 'model-best.pth'), map_location='cuda')
            # model.load_state_dict(loaded_pth['model'], strict=True)
            # model.eval()
            # result_json_path = saved_info['best']['result_json_path']
            # eval_score, _ = evaluate(model, criterion,  postprocessors, val_loader, result_json_path, logger=logger, args=opt, alpha=opt.ec_alpha, device=opt.device, debug=opt.debug)
            # if opt.caption_decoder_type == 'none':
            #     current_score = 2./(1./eval_score['Precision'] + 1./eval_score['Recall'])
            # else:
            #     if opt.criteria_for_best_ckpt == 'dvc':
            #         current_score = np.array(eval_score['METEOR']).mean() + np.array(eval_score['soda_c']).mean()
            #     else:
            #         current_score = np.array(eval_score['para_METEOR']).mean() + np.array(eval_score['para_CIDEr']).mean() + np.array(eval_score['para_Bleu_4']).mean()

            # _ = [item.append(np.array(item).mean()) for item in eval_score.values() if isinstance(item, list)]
            # print_info = '\n'.join([key + ":" + str(eval_score[key]) for key in eval_score.keys()])
            # val_logger.info('Best-model is saved at iter {}.\n'.format(saved_info['best']['iter']))
            # val_logger.info('\nBest Model Performance:\n' + print_info)
            # val_logger.info('\nBest Overall Score {}: {}\n'.format(iteration, current_score))

            # tf_writer.close()
            # break

            val_logger = create_logger(save_folder, 'val.log')
            infos_path = os.path.join(save_folder, 'info.json')

            with open(infos_path, 'r') as f:
                data = json.load(f)
            val_history = data['history']['val_result_history']

            metric_sum = {}
            metrics = ['METEOR', 'CIDEr', 'soda_c', 'Precision', 'Recall']
            for k, v in val_history.items():
                metric_sum[k] = sum([v['eval_score'][metric] for metric in metrics])
                # print(f"{k}: {metric_sum[k]}")

            best_epoch = max(metric_sum, key=metric_sum.get)
            best_val_score = val_history[best_epoch]['eval_score']
            val_logger.info(f"Best epoch: {best_epoch}")
            print_info = '\n'.join([key + ":" + str(best_val_score[key]) for key in best_val_score.keys()])
            val_logger.info('\nBest Model Performance:\n' + print_info)
            val_logger.info('\nBest Overall Score epoch{}: {}\n'.format(best_epoch, metric_sum[best_epoch]))

            break 

    return saved_info


if __name__ == '__main__':
    opt = opts.parse_opts()
    opt.id_ori = opt.id
    
    
    opt.id = 'seq2-ft({})-gt_percent-{}'.format(pretrain_data_mode, opt.ft_gt_percent)
    if opt.id_ori != '':
        opt.id = opt.id + '_' + opt.id_ori
    assert opt.ft_gt_percent <= 1.0 and opt.ft_gt_percent >= 0.0


    if not hasattr(opt, 'visual_feature_folder_val'):
        opt.visual_feature_folder_val = opt.visual_feature_folder
        opt.text_feature_folder_val = opt.text_feature_folder

    if opt.map:
        opt.visual_feature_folder = [map_path(path) for path in opt.visual_feature_folder]
        opt.text_feature_folder = [map_path(path) for path in opt.text_feature_folder]
        opt.visual_feature_folder_val = [map_path(path) for path in opt.visual_feature_folder_val]
        opt.text_feature_folder_val = [map_path(path) for path in opt.text_feature_folder_val]

    if opt.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in opt.gpu_id])
    if opt.disable_cudnn:
        torch.backends.cudnn.enabled = False

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # to avoid OMP problem on macos
    # breakpoint()
    train(opt)

