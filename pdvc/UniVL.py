
import os
import random
import numpy as np
from pathlib import Path
from pdvc.modules.modeling import UniVL
from pdvc.modules.tokenization import BertTokenizer
from transformers import AutoTokenizer, BertForPreTraining
import torch
import argparse

PYTORCH_PRETRAINED_BERT_CACHE = Path(os.getenv('PYTORCH_PRETRAINED_BERT_CACHE',
                                               Path.home() / '.pytorch_pretrained_bert'))

class UniVL_args(object):
    def __init__(self) -> None:
        self.do_pretrain = False
        self.do_train = False
        self.do_eval = True
        self.train_csv = 'data/youcookii_singlef_train.csv'
        self.val_csv = 'data/youcookii_singlef_val.csv'
        self.data_path = 'data/youcookii_caption.pickle'
        self.features_path = 'data/youcookii_videos_feature.pickle'
        self.num_thread_reader = 1
        self.lr = 0.0001
        self.epochs = 20
        self.batch_size = 256
        self.batch_size_val = 3500
        self.lr_decay = 0.9
        self.n_display = 100
        self.video_dim = 1024
        self.seed = 42
        self.max_words = 48
        self.max_frames = 100
        self.feature_framerate = 1
        self.margin = 0.1
        self.hard_negative_rate = 0.5
        self.negative_weighting = 1
        self.n_pair = 1
        self.output_dir = None
        self.bert_model = "bert-base-uncased"
        self.visual_model = "visual-base"
        self.cross_model = "cross-base"
        self.decoder_model = "decoder-base"
        self.init_model = None
        self.do_lower_case = True
        self.warmup_proportion = 0.1
        self.gradient_accumulation_steps = 1
        self.n_gpu = 1
        self.cache_dir = ""
        self.fp16 = False
        self.fp16_opt_level = 'O1'
        self.task_type = "retrieval"
        self.datatype = "youcook"
        self.world_size = 0
        self.local_rank = 0
        self.coef_lr = 0.1
        self.use_mil = False
        self.sampled_use_mil = False
        self.text_num_hidden_layers = 12
        self.visual_num_hidden_layers = 6
        self.cross_num_hidden_layers = 2
        self.decoder_num_hidden_layers = 3
        self.train_sim_after_cross = False
        self.expand_msrvtt_sentences = False
        self.batch_size = int(self.batch_size / self.gradient_accumulation_steps)

    def __repr__(self) -> str:
        return str(self.__dict__)




# def get_args(description='UniVL on Retrieval Task'):
#     parser = argparse.ArgumentParser(description=description)
#     parser.add_argument("--do_pretrain", action='store_true', help="Whether to run training.")
#     parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
#     parser.add_argument("--do_eval", action='store_true', default=True, help="Whether to run eval on the dev set.")

#     parser.add_argument('--train_csv', type=str, default='data/youcookii_singlef_train.csv', help='')
#     parser.add_argument('--val_csv', type=str, default='data/youcookii_singlef_val.csv', help='')
#     parser.add_argument('--data_path', type=str, default='data/youcookii_caption.pickle', help='data pickle file path')
#     parser.add_argument('--features_path', type=str, default='data/youcookii_videos_feature.pickle', help='feature path')

#     parser.add_argument('--num_thread_reader', type=int, default=1, help='')
#     parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
#     parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')
#     parser.add_argument('--batch_size', type=int, default=256, help='batch size')
#     parser.add_argument('--batch_size_val', type=int, default=3500, help='batch size eval')
#     parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate exp epoch decay')
#     parser.add_argument('--n_display', type=int, default=100, help='Information display frequence')
#     parser.add_argument('--video_dim', type=int, default=1024, help='video feature dimension')
#     parser.add_argument('--seed', type=int, default=42, help='random seed')
#     parser.add_argument('--max_words', type=int, default=20, help='')
#     parser.add_argument('--max_frames', type=int, default=100, help='')
#     parser.add_argument('--feature_framerate', type=int, default=1, help='')
#     parser.add_argument('--margin', type=float, default=0.1, help='margin for loss')
#     parser.add_argument('--hard_negative_rate', type=float, default=0.5, help='rate of intra negative sample')
#     parser.add_argument('--negative_weighting', type=int, default=1, help='Weight the loss for intra negative')
#     parser.add_argument('--n_pair', type=int, default=1, help='Num of pair to output from data loader')

#     parser.add_argument("--output_dir", default=None, type=str,
#                         help="The output directory where the model predictions and checkpoints will be written.")
#     parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
#                         help="Bert pre-trained model")
#     parser.add_argument("--visual_model", default="visual-base", type=str, required=False, help="Visual module")
#     parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
#     parser.add_argument("--decoder_model", default="decoder-base", type=str, required=False, help="Decoder module")
#     parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
#     parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
#     parser.add_argument("--warmup_proportion", default=0.1, type=float,
#                         help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
#     parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
#                         help="Number of updates steps to accumulate before performing a backward/update pass.")
#     parser.add_argument('--n_gpu', type=int, default=1, help="Changed in the execute process.")

#     parser.add_argument("--cache_dir", default="", type=str,
#                         help="Where do you want to store the pre-trained models downloaded from s3")

#     parser.add_argument('--fp16', action='store_true',
#                         help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
#     parser.add_argument('--fp16_opt_level', type=str, default='O1',
#                         help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
#                              "See details at https://nvidia.github.io/apex/amp.html")

#     parser.add_argument("--task_type", default="retrieval", type=str, help="Point the task `retrieval` to finetune.")
#     parser.add_argument("--datatype", default="youcook", type=str, help="Point the dataset `youcook` to finetune.")

#     parser.add_argument("--world_size", default=0, type=int, help="distribted training")
#     parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
#     parser.add_argument('--coef_lr', type=float, default=0.1, help='coefficient for bert branch.')
#     parser.add_argument('--use_mil', action='store_true', help="Whether use MIL as Miech et. al. (2020).")
#     parser.add_argument('--sampled_use_mil', action='store_true', help="Whether MIL, has a high priority than use_mil.")

#     parser.add_argument('--text_num_hidden_layers', type=int, default=12, help="Layer NO. of text.")
#     parser.add_argument('--visual_num_hidden_layers', type=int, default=6, help="Layer NO. of visual.")
#     parser.add_argument('--cross_num_hidden_layers', type=int, default=2, help="Layer NO. of cross.")
#     parser.add_argument('--decoder_num_hidden_layers', type=int, default=3, help="Layer NO. of decoder.")

#     parser.add_argument('--train_sim_after_cross', action='store_true', help="Test retrieval after cross encoder.")
#     parser.add_argument('--expand_msrvtt_sentences', action='store_true', help="")

#     args = parser.parse_args()

#     # Check paramenters
#     if args.gradient_accumulation_steps < 1:
#         raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
#             args.gradient_accumulation_steps))
#     if not args.do_train and not args.do_eval:
#         raise ValueError("At least one of `do_train` or `do_eval` must be True.")

#     args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

#     return args

def set_seed_logger(args):
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # world_size = torch.distributed.get_world_size()
    # torch.cuda.set_device(args.local_rank)
    # args.world_size = world_size

    # if not os.path.exists(args.output_dir):
    #     os.makedirs(args.output_dir, exist_ok=True)

    return args

def load_pretrained_UniVL(return_visual_encoder=False):

    args = UniVL_args()
    args = set_seed_logger(args)
    device, n_gpu = 'cuda', 1

    init_model = '/cpfs01/user/liuhuabin/PDVC/pdvc/modules/univl.pretrained.bin'
    model_state_dict = torch.load(init_model, map_location='cpu')

    # Prepare model
    cache_dir = os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    model = UniVL.from_pretrained('bert-base-uncased', 'visual-base', 'cross-base', 'decoder-base',
                                   cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)

    model.to(device)
    if return_visual_encoder:
        return model.bert, model.visual, model.normalize_video
    else:
        return model.bert

def build_UniVL_tokenizer():
    return BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# if __name__ == '__main__':
#     device, n_gpu = 'cuda', 1
#     captions = ['I love you', 'you believe me']

#     tokenizer_hg = AutoTokenizer.from_pretrained("bert-base-uncased")
#     text_encoder_hg = tokenizer_hg(captions, return_tensors='pt', truncation=True, padding=True, max_length=20)
#     text_encoder_hg = {key: _.to(device) if isinstance(_, torch.Tensor) else _ for key, _ in text_encoder_hg.items()}
#     attention_mask = text_encoder_hg['attention_mask']

#     args = UniVL_args()
#     args = set_seed_logger(args)
#     args.init_model = 'modules/univl.pretrained.bin'
#     # tokenizer = build_UniVL_tokenizer()
#     # input_ids = []
#     # for sent in captions:
#     #     sent = tokenizer.tokenize(sent)
#     #     sent = ['[CLS]'] + sent + ['[SEP]']
#     #     input_ids += tokenizer.convert_tokens_to_ids(sent)
#     model = load_pretrained_UniVL(args, device, n_gpu, args.local_rank, args.init_model)
#     text_embed = model(**text_encoder_hg, output_all_encoded_layers=True)[0][-1]
#     breakpoint()

if __name__ == '__main__':
    device, n_gpu = 'cuda', 1
    args = UniVL_args()
    args = set_seed_logger(args)
    args.init_model = 'modules/univl.pretrained.bin'
    # tokenizer = build_UniVL_tokenizer()
    # input_ids = []
    # for sent in captions:
    #     sent = tokenizer.tokenize(sent)
    #     sent = ['[CLS]'] + sent + ['[SEP]']
    #     input_ids += tokenizer.convert_tokens_to_ids(sent)
    model_bert, model_visual, video_normalizer = load_pretrained_UniVL(args, device, n_gpu, args.local_rank, args.init_model)
    inputs = torch.rand(2,215,1024)
    video_mask = torch.ones(2,215)
    inputs = video_normalizer(inputs)
    visual_embed = model_visual(inputs, video_mask, output_all_encoded_layers=True)[0][-1]
    
    breakpoint()