# DIBS
Official implementation for "DIBS: Enhancing Dense Video Captioning with Unlabeled Videos via Pseudo Boundary Enrichment and Online Refinement" (CVPR 2024)

[\[Paper\]](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_DIBS_Enhancing_Dense_Video_Captioning_with_Unlabeled_Videos_via_Pseudo_CVPR_2024_paper.html)

## Updates
- 2024.6.X: Release code
## Code
### Environment Configuration
```
conda create -y -n dibs python=3.9
conda activate dibs
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install ffmpeg
pip install -r requirement.txt
git clone https://github.com/haowuxc/DIBS.git
cd DIBS/pdvc/ops
sh make.sh
```


### Training and Evaluation
1. Training with pseudo boundaries on YouCook2 and ActivityNet.
```
python train.py --cfg_path cfgs/anet_clip-simop_order_v2_top30_r2_iter3_th2_refine_aug\(8,0.02\)_top3_2stage_inscap.yml
python train.py --cfg_path cfgs/yc2_univl-simop_order_v2_top15_r1_iter3_th1_refine_aug\(8,0.02\)_top3_2stage_inscap.yml
```
2. Pretraining using HowTo100M video subset.
```
python train.py --cfg_path cfgs/howto-anet_anet_clip_topk30_r1_iter3_th1_refine_aug\(8,0.02\)_top3_2stage_inscap.yml
python train.py --cfg_path cfgs/howto-yc2_yc2_univl_topk25_r1_iter3_th1_refine_aug\(8,0.02\)_top3_2stage_inscap.yml
```
3. Fine-tuning on YouCook2 and ActivityNet.
```
python train_ft2_gt.py --cfg_path cfgs/howto-anet_anet_clip_topk30_r1_iter3_th1_refine_aug\(8,0.02\)_top3_2stage_inscap.yml
python train_ft2_gt.py --cfg_path cfgs/howto-yc2_yc2_univl_topk25_r1_iter3_th1_refine_aug\(8,0.02\)_top3_2stage_inscap.yml
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