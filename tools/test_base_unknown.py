import argparse
import os
import os.path as osp
import shutil
import tempfile
from scipy import ndimage
import mmcv
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import init_dist, get_dist_info, load_checkpoint

from mmdet.core import coco_eval, results2json, wrap_fp16_model, tensor2imgs, get_classes
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmdet.apis.inference import *
import cv2

import numpy as np
import matplotlib.cm as cm

def vis_seg(data, result, img_norm_cfg, data_id, colors, score_thr,score_thr_min, save_dir):
    img_tensor = data['img'][0]
    img_metas = data['img_meta'][0]
    imgs = tensor2imgs(img_tensor, **img_norm_cfg)
    assert len(imgs) == len(img_metas)
    class_names = get_classes('coco')

    for img, img_meta, cur_result in zip(imgs, img_metas, result):
        if cur_result is None:
            continue
        h, w, _ = img_meta['img_shape']
        img_show = img[:h, :w, :]

        seg_label = cur_result[0]
        seg_label = seg_label.cpu().numpy().astype(np.uint8)
        cate_label = cur_result[1]
        cate_label = cate_label.cpu().numpy()
        score = cur_result[2].cpu().numpy() # classification score

        vis_inds = score > score_thr_min
        seg_label = seg_label[vis_inds]
        num_mask = seg_label.shape[0]
        cate_label = cate_label[vis_inds]
        cate_score = score[vis_inds]

        mask_density = []
        for idx in range(num_mask):
            cur_mask = seg_label[idx, :, :]
            cur_mask = mmcv.imresize(cur_mask, (w, h))
            cur_mask = (cur_mask > 0.5).astype(np.int32)
            mask_density.append(cur_mask.sum())
        orders = np.argsort(mask_density)
        seg_label = seg_label[orders]
        cate_label = cate_label[orders]
        cate_score = cate_score[orders]

        seg_show = img_show.copy()
        for idx in range(num_mask):
            idx = -(idx+1)
            cur_mask = seg_label[idx, :,:]
            cur_mask = mmcv.imresize(cur_mask, (w, h))
            cur_mask = (cur_mask > 0.5).astype(np.uint8)
            if cur_mask.sum() == 0:
               continue
            color_mask = np.random.randint(
                0, 256, (1, 3), dtype=np.uint8)
            cur_mask_bool = cur_mask.astype(np.bool)
            cur_cate = cate_label[idx]
            cur_score = cate_score[idx]

            if cur_score>=score_thr:
                seg_show[cur_mask_bool] = img_show[cur_mask_bool] * 0.6 + color_mask * 0.4
                label_text = class_names[cur_cate]
            else:
                seg_show[cur_mask_bool] = img_show[cur_mask_bool] * 0.4 + color_mask * 0.6
                label_text = 'unknown'



            label_text += '|{:.02f}'.format(cur_score)
            # center
            center_y, center_x = ndimage.measurements.center_of_mass(cur_mask)
            vis_pos = (max(int(center_x) - 10, 0), int(center_y))
            cv2.putText(seg_show, label_text, vis_pos,
                        cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255))  # green
        mmcv.imwrite(seg_show, '{}_{}.jpg'.format(save_dir, data_id))

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('--config', default='configs/solo/solo_r101_fpn_8gpu_3x.py', help='test config file path')
    parser.add_argument('--checkpoint', default='work_dirs/solo_release_r101_fpn_8gpu_3x/SOLO_R101_3x.pth', help='checkpoint file')
    parser.add_argument('--score_thr', type=float, default=0.3, help='score threshold for visualization')
    parser.add_argument('--score_thr_min', type=float, default=0.2,
                        help='score threshold for known objects (low confidence) visualization')
    parser.add_argument('--file', help='Image file')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    device = 'cuda:0'
    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None
    dataset = build_dataset(cfg.data.test)
    model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
    checkpoint = load_checkpoint(model, args.checkpoint)
    model = model.to(device)
    model.eval()
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    class_num = 1000  # ins
    colors = [(np.random.random((1, 3)) * 255).tolist()[0] for i in range(class_num)]

    img = mmcv.imread(args.file)
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)
    data = scatter(collate([data], samples_per_gpu=1), [device])[0]
    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    filename, _ = os.path.splitext(args.file)
    vis_seg(data, result, cfg.img_norm_cfg, data_id='seg', colors=colors, score_thr=args.score_thr,
            score_thr_min=args.score_thr_min, save_dir=filename)



if __name__ == '__main__':
    main()
