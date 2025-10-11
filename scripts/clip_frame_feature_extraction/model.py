import sys
import torch as th
import torchvision.models as models
# from videocnn.models import resnext
from torch import nn
import torch
from transformers import AutoProcessor, CLIPModel, CLIPVisionModelWithProjection, BlipModel, AutoModel
import os 
os.environ["TRANSFORMERS_CACHE"] = "/mnt/data/Gvlab/wuhao/models/checkpoints"

class GlobalAvgPool(nn.Module):
    def __init__(self):
        super(GlobalAvgPool, self).__init__()

    def forward(self, x):
        return th.mean(x, dim=[-2, -1])


def get_model(args):
    assert args.type in ['2d', '3d']
    if args.type == '2d':
        # # original resnet model
        # print('Loading 2D-ResNet-152 ...')
        # model = models.resnet152(pretrained=True)
        # model = nn.Sequential(*list(model.children())[:-2], GlobalAvgPool())
        # model = model.cuda()
        
        # openai clip vit model 
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if args.model == 'clip':
            print('Loading 2D-ViT-L-14 ...')
            # without projection head
            processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14", cache_dir="/mnt/data/Gvlab/wuhao/models/checkpoints")
            model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", cache_dir="/mnt/data/Gvlab/wuhao/models/checkpoints")
        elif args.model == 'blip':
            print('loading 2D-BLiP ...')
            model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-large", cache_dir="/mnt/data/Gvlab/wuhao/models/checkpoints")#, torch_dtype=th.float16)
            processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large", cache_dir="/mnt/data/Gvlab/wuhao/models/checkpoints")
            # breakpoint() 
        elif args.model == 'xclip':
            processor = AutoProcessor.from_pretrained("microsoft/xclip-large-patch14-kinetics-600", cache_dir="/mnt/data/Gvlab/wuhao/models/checkpoints")
            model = AutoModel.from_pretrained("microsoft/xclip-large-patch14-kinetics-600", cache_dir="/mnt/data/Gvlab/wuhao/models/checkpoints")



        # # with projection head
        # processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14", cache_dir="/mnt/data/Gvlab/wuhao/models/checkpoints")
        # model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14", cache_dir="/mnt/data/Gvlab/wuhao/models/checkpoints")
        model = model.to(device)

    # else:
    #     print('Loading 3D-ResneXt-101 ...')
    #     model = resnext.resnet101(
    #         num_classes=400,
    #         shortcut_type='B',
    #         cardinality=32,
    #         sample_size=112,
    #         sample_duration=16,
    #         last_fc=False)
    #     model = model.cuda()
    #     model_data = th.load(args.resnext101_model_path)
    #     model.load_state_dict(model_data)

    model.eval()
    print('loaded')
    if args.type == '2d':
        return model, processor
    return model
