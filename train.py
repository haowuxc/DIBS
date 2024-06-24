# coding:utf-8
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

pdvc_dir = dirname(abspath(__file__))
sys.path.insert(0, pdvc_dir)
sys.path.insert(0, os.path.join(pdvc_dir, 'densevid_eval3'))
sys.path.insert(0, os.path.join(pdvc_dir, 'densevid_eval3/SODA'))
# print(sys.path)
CUDA_LAUNCH_BLOCKING=1

os.environ["TOKENIZERS_PARALLELISM"] = "false" # To avoid warning of tokenizer
from eval_utils import evaluate
import opts
from tensorboardX import SummaryWriter
from misc.utils import print_alert_message, build_folder, create_logger, backup_envir, print_opt, set_seed
from data.video_dataset import PropSeqDataset, collate_fn
from pdvc.pdvc import build
from collections import OrderedDict
# from transformers import AutoTokenizer, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import copy
import random
import numpy as np 

a100_folder = ['/cpfs01/shared/Gvlab-A100/Gvlab-A100_hdd/wuhao/youcook2', '/cpfs01/shared/Gvlab-A100/Gvlab-A100_hdd/wuhao/Tasty/features', '/cpfs01/shared/Gvlab-A100/Gvlab-A100_hdd/huabin/dataset/Tasty/UniVL_feature', '/cpfs01/shared/Gvlab-A100/Gvlab-A100_hdd/huabin/dataset/Anet', '/cpfs01/shared/Gvlab-A100/Gvlab-A100_hdd/wuhao/howto100m/features']
# r3090_folder = ['/mnt/data/Gvlab/wuhao/features/yc2', '/mnt/data/Gvlab/wuhao/features/tasty', '/mnt/data/Gvlab/wuhao/features/tasty/univl', '/mnt/data/Gvlab/wuhao/features/anet', '/mnt/data/Gvlab/wuhao/features/howto100m']
r3090_folder = ['/ailab/group/pjlab-sport/wuhao/cpfs_3090/features/yc2', '/ailab/group/pjlab-sport/wuhao/cpfs_3090/features/tasty', '/ailab/group/pjlab-sport/wuhao/cpfs_3090/features/tasty/univl', '/ailab/group/pjlab-sport/wuhao/cpfs_3090/features/anet', '/ailab/group/pjlab-sport/wuhao/cpfs_3090/features/howto100m']

# /cpfs01/shared/Gvlab-A100/Gvlab-A100_hdd/wuhao/howto100m/features -> /mnt/data/Gvlab/wuhao/features/howto100m
# /cpfs01/shared/Gvlab-A100/Gvlab-A100_hdd/wuhao/howto100m/features/clip -> /mnt/data/Gvlab/wuhao/features/howto100m/clip_features
# /cpfs01/shared/Gvlab-A100/Gvlab-A100_hdd/wuhao/howto100m/features/UniVL -> /mnt/data/Gvlab/wuhao/features/howto100m/univl_features

def construct_save_path(opt, save_folder="/mnt/data/pjlab-3090-sport/wuhao/logs/dibs"):
    elements = []
    # breakpoint()    
    if len(opt.train_caption_file) == 2:
        elements.append('howto_llama2')
        elements.append('howto')
        if 'yc2' in opt.train_caption_file[1]:            
            elements.append('yc2')
        elif 'anet' in opt.train_caption_file[1]:
            elements.append('anet')
    else:
        if 'yc2' in opt.train_caption_file:
            elements.append('yc2')
        elif 'anet' in opt.train_caption_file:
            elements.append('anet')
        elif 'howto' in opt.train_caption_file:
            elements.append('howto_llama2')
            # elements.append('howto')

    if 'clip' in opt.visual_feature_folder[0] or 'CLIP' in opt.visual_feature_folder[0]:
        elements.append('clip')
    elif 'UniVL' in opt.visual_feature_folder[0] or 'univl' in opt.visual_feature_folder[0]:
        elements.append('univl')
    # add pbox parameters
    pbox_type = "simop_v2" if opt.pseudo_box_type == "similarity_op_order_v2" else "simop"
    elements.append(pbox_type)
    elements.append(f"top{opt.top_frames}")
    elements.append(f"r{opt.width_ratio}")
    elements.append(f"iter{opt.iteration}")
    elements.append(f"th{opt.width_th}")
    return os.path.join(save_folder, '_'.join(elements) + '.json')
    


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def map_path(path):
    path_backup = copy.deepcopy(path)
    # breakpoint()
    for i, folder in enumerate(a100_folder):
        if folder in path:
            path = path.replace(folder, r3090_folder[i])
            
            
    if path == path_backup:
        if path.startswith('/ailab'):
            pass
        else:
            # path = '/mnt' + path[6:]
            print('map failed')
            exit(1)
    return path

def train(opt):
    set_seed(opt.seed)
    save_folder = build_folder(opt)
    logger = create_logger(save_folder, 'train.log')
    tf_writer = SummaryWriter(os.path.join(save_folder, 'tf_summary'))


    saved_path = construct_save_path(opt, save_folder=save_folder)


    if not opt.start_from:
        backup_envir(save_folder, opt)
        logger.info('backup evironment completed !')

    saved_info = {'best': {}, 'last': {}, 'history': {}, 'eval_history': {}}

    # continue training
    if opt.start_from:
        opt.pretrain = False
        infos_path = os.path.join(save_folder, 'info.json')
        with open(infos_path) as f:
            logger.info('Load info from {}'.format(infos_path))
            saved_info = json.load(f)
            prev_opt = saved_info[opt.start_from_mode[:4]]['opt']

            exclude_opt = ['start_from', 'start_from_mode', 'pretrain']
            for opt_name in prev_opt.keys():
                if opt_name not in exclude_opt:
                    vars(opt).update({opt_name: prev_opt.get(opt_name)})
                if prev_opt.get(opt_name) != vars(opt).get(opt_name):
                    logger.info('Change opt {} : {} --> {}'.format(opt_name, prev_opt.get(opt_name),
                                                                   vars(opt).get(opt_name)))
    # print(opt.text_feature_folder)
    # print(opt.train_caption_file)
    if len(opt.visual_feature_folder) == 2:
        train_dataset_1 = PropSeqDataset(opt.train_caption_file[0],
                                            [opt.visual_feature_folder[0]],
                                            [opt.text_feature_folder[0]],
                                            opt.dict_file, True, 'gt',
                                            opt)
        train_dataset_2 = PropSeqDataset(opt.train_caption_file[1],
                                            [opt.visual_feature_folder[1]],
                                            [opt.text_feature_folder[1]],
                                            opt.dict_file, True, 'gt',
                                            opt)
        train_dataset = torch.utils.data.ConcatDataset([train_dataset_1, train_dataset_2])
        train_dataset.translator = train_dataset_1.translator

    else:
        train_dataset = PropSeqDataset(opt.train_caption_file,
                                    opt.visual_feature_folder,
                                    opt.text_feature_folder,
                                    opt.dict_file, True, 'gt',
                                    opt)

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
    g = torch.Generator()
    g.manual_seed(0)

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size,
                              shuffle=True, num_workers=opt.nthreads, collate_fn=collate_fn, worker_init_fn=seed_worker, generator=g)

    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size_for_eval,
                            shuffle=False, num_workers=opt.nthreads, collate_fn=collate_fn, worker_init_fn=seed_worker, generator=g)

    epoch = saved_info[opt.start_from_mode[:4]].get('epoch', 0)
    iteration = saved_info[opt.start_from_mode[:4]].get('iter', 0)
    best_val_score = saved_info[opt.start_from_mode[:4]].get('best_val_score', -1e5)
    val_result_history = saved_info['history'].get('val_result_history', {})
    loss_history = saved_info['history'].get('loss_history', {})
    lr_history = saved_info['history'].get('lr_history', {})
    opt.current_lr = vars(opt).get('current_lr', opt.lr)

    # Build model

    model, criterion, contrastive_criterion, postprocessors = build(opt)
    model.translator = train_dataset.translator
    model.train()

    # try to load saved pbox
    if os.path.exists(saved_path):
        try:
            with open(saved_path, 'r') as f:
                model.pseudo_boxes = json.load(f)
        except:
            # delete the bad file
            os.remove(saved_path)

    # Recover the parameters
    if opt.start_from and (not opt.pretrain):
        if opt.start_from_mode == 'best':
            model_pth = torch.load(os.path.join(save_folder, 'model-best.pth'))
        elif opt.start_from_mode == 'last':
            model_pth = torch.load(os.path.join(save_folder, 'model-last.pth'))
        logger.info('Loading pth from {}, iteration:{}'.format(save_folder, iteration))
        model.load_state_dict(model_pth['model'])

    # Load the pre-trained model
    if opt.pretrain and (not opt.start_from):
        logger.info('Load pre-trained parameters from {}'.format(opt.pretrain_path))
        model_pth = torch.load(opt.pretrain_path, map_location=torch.device(opt.device))
        # query_weight = model_pth['model'].pop('query_embed.weight')
        if opt.pretrain == 'encoder':
            encoder_filter = model.get_filter_rule_for_encoder()
            encoder_pth = {k:v for k,v in model_pth['model'].items() if encoder_filter(k)}
            model.load_state_dict(encoder_pth, strict=True)
        elif opt.pretrain == 'decoder':
            encoder_filter = model.get_filter_rule_for_encoder()
            decoder_pth = {k:v for k,v in model_pth['model'].items() if not encoder_filter(k)}
            model.load_state_dict(decoder_pth, strict=True)
            pass
        elif opt.pretrain == 'full':
            # model_pth = transfer(model, model_pth)
            model.load_state_dict(model_pth['model'], strict=True)
        else:
            raise ValueError("wrong value of opt.pretrain")
        

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

    training_params = [{'params': other_params, 'lr': opt.lr}]

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

    if opt.start_from:
        optimizer.load_state_dict(model_pth['optimizer'])
        lr_scheduler.step(epoch-1)

    # print the args for debugging  
    print_opt(opt, model, logger)
    print_alert_message('Strat training !', logger)

    loss_sum = OrderedDict()
    bad_video_num = 0

    start = time.time()
    # breakpoint()
    weight_dict = criterion.weight_dict
    logger.info('loss type: {}'.format(weight_dict.keys()))
    logger.info('loss weights: {}'.format(weight_dict.values()))

    # Epoch-level iteration
    refine_pseudo_box_copy = copy.deepcopy(opt.refine_pseudo_box)
    pseudo_box_aug_copy = copy.deepcopy(opt.pseudo_box_aug)

    while True:
        # if epoch > opt.start_refine_epoch:
        #     opt.refine_pseudo_box = refine_pseudo_box_copy
        #     opt.pseudo_box_aug = pseudo_box_aug_copy
        #     criterion.refine_pseudo_box = refine_pseudo_box_copy
        #     criterion.pseudo_box_aug = pseudo_box_aug_copy
        #     model.opt = opt 
        # else:
        #     opt.refine_pseudo_box = False
        #     opt.pseudo_box_aug = False
        #     criterion.refine_pseudo_box = False
        #     criterion.pseudo_box_aug = False
        #     model.opt = opt
        
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
        trained_samples = 0
        for dt in tqdm(train_loader, disable=opt.disable_tqdm):
            # if dt['video_key'][0] != 'LGArj9Do0xc':
            #     continue
            # # for fast debugging
            if opt.test:
                if trained_samples > 5:
                    break
                else:
                    trained_samples += 1
            # if trained_samples < 1714:
            #     trained_samples += 1
            #     continue
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

            try:
                output, loss = model(dt, criterion, contrastive_criterion)
            except Exception as e:
                print(e)
                print(dt['video_key'])
                continue
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

            losses_log_every = int(len(train_loader) / 10)

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
            eval_score, _ = evaluate(model, criterion,  postprocessors, val_loader, result_json_path, logger=logger, args=opt, alpha=opt.ec_alpha, device=opt.device, debug=opt.debug)
            if opt.caption_decoder_type == 'none':
                current_score = 2./(1./eval_score['Precision'] + 1./eval_score['Recall'])
            else:
                if opt.criteria_for_best_ckpt == 'dvc':
                    current_score = np.array(eval_score['METEOR']).mean() + np.array(eval_score['soda_c']).mean()
                elif opt.criteria_for_best_ckpt == 'overall':
                    current_score = np.array(eval_score['Bleu_4']).mean() + \
                    np.array(eval_score['CIDEr']).mean() + \
                    np.array(eval_score['METEOR']).mean() + \
                    2./(1./eval_score['Precision'] + 1./eval_score['Recall'])
                else:
                    current_score = np.array(eval_score['para_METEOR']).mean() + np.array(eval_score['para_CIDEr']).mean() + np.array(eval_score['para_Bleu_4']).mean()

            # add to tf summary
            for key in eval_score.keys():
                tf_writer.add_scalar(key, np.array(eval_score[key]).mean(), iteration)

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

        if epoch == 1 and model.pseudo_boxes is not None:
            # save the pseudo boxes
            pbox_save_path = construct_save_path(opt)
            if not os.path.exists(pbox_save_path):
                with open(pbox_save_path, 'w') as f:
                    json.dump(model.pseudo_boxes, f)

        lr_scheduler.step()
        opt.current_lr = optimizer.param_groups[0]['lr']
        torch.cuda.empty_cache()
        # Stop criterion
        if epoch >= opt.epoch:
            # save the pesudo box



            # # ===============================old code==============================================
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
            # =================================new code=========================================================
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

    if not hasattr(opt, 'visual_feature_folder_val'):
        opt.visual_feature_folder_val = opt.visual_feature_folder
        opt.text_feature_folder_val = opt.text_feature_folder
    # breakpoint()
    if opt.map:
        opt.visual_feature_folder = [map_path(path) for path in opt.visual_feature_folder]
        opt.text_feature_folder = [map_path(path) for path in opt.text_feature_folder]
        opt.visual_feature_folder_val = [map_path(path) for path in opt.visual_feature_folder_val]
        opt.text_feature_folder_val = [map_path(path) for path in opt.text_feature_folder_val]

    # breakpoint()

    if opt.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in opt.gpu_id])
    if opt.disable_cudnn:
        torch.backends.cudnn.enabled = False

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # to avoid OMP problem on macos
    # breakpoint()
    train(opt)

