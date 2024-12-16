# DIBS
Official implementation for "DIBS: Enhancing Dense Video Captioning with Unlabeled Videos via Pseudo Boundary Enrichment and Online Refinement" (CVPR 2024)

[\[Paper\]](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_DIBS_Enhancing_Dense_Video_Captioning_with_Unlabeled_Videos_via_Pseudo_CVPR_2024_paper.html)

## Updates
- [x] 2024.6.28: Release code
- [x] 2024.12.16: Release pre-extracted CLIP and UniVL features of the YouCook2, ActivityNet and Howto100M subset in this [link](https://huggingface.co/datasets/Exclibur/dibs-feature).
## Code
### Environment Configuration
```
conda create -y -n dibs python=3.8
conda activate dibs
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
conda install ffmpeg
pip install -r requirement.txt
git clone --recursive https://github.com/haowuxc/DIBS.git
cd DIBS/pdvc/ops
sh make.sh
```


### Training and Evaluation
1. Training with pseudo boundaries on YouCook2 and ActivityNet.
```
python train.py --cfg_path cfgs/anet_clip_refine.yml
python train.py --cfg_path cfgs/yc2_univl_refine.yml
```
2. Pretraining using HowTo100M video subset.
```
python train.py --cfg_path cfgs/howto-anet_anet_clip_refine.yml
python train.py --cfg_path cfgs/howto-yc2_yc2_univl_refine.yml
```
3. Fine-tuning on YouCook2 and ActivityNet.
```
python train_ft2_gt.py --cfg_path cfgs/howto-anet_anet_clip_refine.yml
python train_ft2_gt.py --cfg_path cfgs/howto-yc2_yc2_univl_refine.yml
```
4. Evaluation on YouCook2 and ActivityNet.
```
YouCook2
python eval.py  --eval_save_dir SAVE_PATH --eval_folder CONFIG_NAME --eval_caption_file data/yc2/captiondata/yc2_val.json --eval_proposal_type queries --gpu_id 0

# ActivityNet
python eval.py  --eval_save_dir SAVE_PATH --eval_folder CONFIG_NAME --eval_caption_file data/anet/captiondata/val_1.json --eval_proposal_type queries --gpu_id 0
```
**Note:**
- `SAVE_PATH` is the folder where all files will be stored.
- `CONFIG_NAME` is the subfolder path of the specified configuration, i.e. `model_path = os.path.join(SAVE_PATH, CONFIG_NAME)`.

## Citation
If you find our work useful in your research, please consider citing:
```
@InProceedings{Wu_2024_CVPR,
    author    = {Wu, Hao and Liu, Huabin and Qiao, Yu and Sun, Xiao},
    title     = {DIBS: Enhancing Dense Video Captioning with Unlabeled Videos via Pseudo Boundary Enrichment and Online Refinement},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {18699-18708}
}
```

## Acknowledgement
We would like to thank the authors of the [PDVC](https://github.com/ttengwang/PDVC) paper and the [Drop-DTW](https://github.com/SamsungLabs/Drop-DTW) paper for making their code available as open-source. Their work has greatly contributed to our project.
