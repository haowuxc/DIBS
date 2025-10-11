import torch as th
import math
import numpy as np
from video_loader import VideoLoader
from torch.utils.data import DataLoader
import argparse
from model import get_model
from preprocessing import Preprocessing
from random_sequence_shuffler import RandomSequenceSampler
import torch.nn.functional as F
import torch 
import os
import csv 



def generate_csv(video_path, output_path, output_csv):
    video_paths = []
    feature_paths = []


    # normal videos
    for root, _, files in os.walk(video_path):
        for file in files:
            if file.endswith(('.mp4', '.webm', '.mkv')):
                video_path = os.path.join(root, file)
                video_paths.append(video_path)
                
                filename = os.path.splitext(file)[0] 
                feature_path = os.path.join(output_path, filename + '_{}.npy'.format(args.model))
                feature_paths.append(feature_path)
    # Assuming video_path and feature_path are lists of paths
    data = zip(video_paths, feature_paths)
    
    with open(output_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['video_path', 'feature_path'])  # Header
        
        # Writing data to CSV
        for row in data:
            writer.writerow(row)

parser = argparse.ArgumentParser(description='Easy video feature extractor')

parser.add_argument('--video_path', type=str,)
parser.add_argument('--output_path', type=str,)

parser.add_argument(
    '--csv',
    type=str,
    default=None,
    help='input csv with video input path')
parser.add_argument('--batch_size', type=int, default=64,
                            help='batch size')
parser.add_argument('--type', type=str, default='2d',
                            help='CNN type')
parser.add_argument('--model', type=str, default='clip')
parser.add_argument('--half_precision', type=int, default=1,
                            help='output half precision float')
parser.add_argument('--num_decoding_thread', type=int, default=4,
                            help='Num parallel thread for video decoding')
parser.add_argument('--l2_normalize', type=int, default=1,
                            help='l2 normalize feature')
parser.add_argument('--resnext101_model_path', type=str, default='model/resnext101.pth',
                            help='Resnext model path')
args = parser.parse_args()

generate_csv(args.video_path, args.output_path, 'video_feature_tmp.csv')
# os.makedirs(os.path.join(args.output_path, 'features'), exist_ok=True)

# if args.csv is None:
args.csv = 'video_feature_tmp.csv'



dataset = VideoLoader(
    args.csv,
    framerate=1 if args.type == '2d' else 24,
    # size=224 if args.type == '2d' else 112,
    centercrop=(args.type == '3d'),
)
n_dataset = len(dataset)
sampler = RandomSequenceSampler(n_dataset, 10)
loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=args.num_decoding_thread,
    sampler=sampler if n_dataset > 10 else None,
)
preprocess = Preprocessing(args.type)
if args.type == '2d':
    model, processor = get_model(args)
else:
    model = get_model(args)

with th.no_grad():
    for k, data in enumerate(loader):
        # breakpoint()
        input_file = data['input'][0]
        output_file = data['output'][0]
        if os.path.exists(output_file):
            print('Video {} already processed.'.format(input_file))
            continue
        if len(data['video'].shape) > 3:
            print('Computing features of video {}/{}: {}'.format(
                k + 1, n_dataset, input_file))
            video = data['video'].squeeze().cuda()
            # breakpoint()
            if len(video.shape) == 3:
                video = torch.unsqueeze(video, 0)
            if len(video.shape) == 4:
                # video = preprocess(video)
                # video = processor(images=video, return_tensors="pt")
                if args.model == 'blip':
                    inputs = processor(images=video, return_tensors="pt", text=['a photo of a cat', 'a photo of a dog'], padding=True)
                    inputs['pixel_values'] = inputs['pixel_values'].cuda()
                    outputs = model(**inputs)
                    video_features = outputs.image_embeds
                else:
                    try:
                        video = processor(images=list(video), return_tensors="pt")
                        video["pixel_values"] = video["pixel_values"].cuda()
                    # breakpoint()
                        video_features = model.get_image_features(**video) # without projection head
                    except:
                        print("list index out of range, video shape: {}".format(video.shape))
                        continue
                # out = model(**video) # with projection head p video_features.pooler_output.shape
                # breakpoint()
                # video_features = model(**video) # with projection head
                # video_features = video_features.image_embeds
                # print(video_features.shape)
                # exit()
                # breakpoint()
                # video = th.stack([processor(images=video[i], return_tensors="pt") for i in range(video.shape[0])], dim=0)
                # breakpoint()
                # n_chunk = len(video)
                # features = th.cuda.FloatTensor(n_chunk, 2048).fill_(0)
                # n_iter = int(math.ceil(n_chunk / float(args.batch_size)))
                # breakpoint()
                # for i in range(n_iter):
                #     min_ind = i * int(args.batch_size)
                #     max_ind = (i + 1) * int(args.batch_size)
                #     video_batch = video[min_ind:max_ind].cuda()
                #     batch_features = model(video_batch)
                #     if args.l2_normalize:
                #         batch_features = F.normalize(batch_features, dim=1)
                #     features[min_ind:max_ind] = batch_features
                # features = features.cpu().numpy()
                # if args.half_precision:
                #     video_features = video_features.astype('float16')
                np.save(output_file, video_features.cpu().numpy())
        else:
            print('Video {} already processed.'.format(input_file))
