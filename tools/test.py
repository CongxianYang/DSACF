# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import cv2
import torch
import numpy as np
import math
import sys
import shutil
sys.path.append('../')

from pysot.core.config import cfg
from pysot.tracker.siamcar_tracker import SiamCARTracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from pysot.models.model_builder import ModelBuilder

from toolkit.datasets import DatasetFactory
from tqdm import tqdm
parser = argparse.ArgumentParser(description='siamcar tracking')

parser.add_argument('--video', default='', type=str,
        help='eval one special video')
parser.add_argument('--dataset', type=str, default='GTOT',
        help='datasets')#OTB100 LaSOT UAV123 GOT-10k  RGBT234 LasHer
parser.add_argument('--vis', action='store_true',default=True,
        help='whether visualzie result')
parser.add_argument('--snapshot', type=str, default='/home/xiancong/Project_all/SiamDSACF/tools/snapshot/checkpoint_e13.pth',
        help='snapshot of models to eval')

parser.add_argument('--config', type=str, default='../experiments/siamDSACF_r50/config.yaml',
        help='config file')

args = parser.parse_args()

torch.set_num_threads(1)


def main():
    # load config
    cfg.merge_from_file(args.config)

    # hp_search
    params = getattr(cfg.HP_SEARCH,args.dataset)
    hp = {'lr': params[0], 'penalty_k':params[1], 'window_lr':params[2]}
    if(args.dataset=='LasHeR'):
        dataset_root='/media/xiancong/DataPlus/DataSets/LasHeR0428/LasHeR_Divided_TraningSet&TestingSet/TestingSet/testingset'
    else:
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        dataset_root = os.path.join(cur_dir,'/home/xiancong/Data_set', args.dataset)

    model = ModelBuilder()


    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval()

    # build tracker
    tracker = SiamCARTracker(model, cfg.TRACK)

    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    model_name = args.snapshot.split('/')[-1] + str(hp['lr']) + '_' + str(hp['penalty_k']) + '_' + str(hp['window_lr'])
    print('Running tracker %s on %s...' % (tracker.name, args.dataset))
    # OPE tracking
    for v_idx, (rgb_img_files, t_img_files, groundtruth, gt_bbox_r, gt_bbox_t) in tqdm(enumerate(dataset),total=len(dataset)):
        # GTOT convert left_top  w,h
        # if args.dataset == 'GTOT':
        #    gt_bbox_t[0]= (gt_bbox_t[0][0],gt_bbox_t[0][1],gt_bbox_t[0][2] - gt_bbox_t[0][0], gt_bbox_t[0][3] - gt_bbox_t[0][1])
        #    gt_bbox_r[0]= (gt_bbox_r[0][0],gt_bbox_r[0][1],gt_bbox_r[0][2] - gt_bbox_r[0][0], gt_bbox_r[0][3] - gt_bbox_r[0][1])##x,y,w,h
        video_name = dataset.seq_names[v_idx]
        if args.video != '':
            # test one special video
            if video_name != args.video:
                continue
        toc = 0
        pred_bboxes = []
        track_times = []
        for idx, rgb_img_file in enumerate(rgb_img_files):
            tic = cv2.getTickCount()
            t_img_file = t_img_files[idx]
            rgb_img = cv2.imread(rgb_img_file, cv2.IMREAD_COLOR)
            t_img = cv2.imread(t_img_file, cv2.IMREAD_GRAYSCALE)
            t_img = cv2.cvtColor(t_img, cv2.COLOR_GRAY2RGB)
            if idx == 0:
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox_r[idx]))##统一转化为中心点和宽高
                gt_bbox_r_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                groundtruth_cv=gt_bbox_r_
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox_t[idx]))
                gt_bbox_t_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]##x,y,w,h
                tracker.init(rgb_img,t_img,groundtruth_cv,gt_bbox_r_,gt_bbox_t_)
                pred_bbox = groundtruth[0]
                pred_bboxes.append(pred_bbox)
            else:
                outputs = tracker.track(rgb_img,t_img, hp)
                pred_bbox = outputs['bbox']
                pred_bboxes.append(pred_bbox)
            toc += cv2.getTickCount() - tic
            track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
            if idx == 0:
                cv2.destroyAllWindows()
            if args.vis and idx > 0:
                  show_rgbt_image(rgb_img, pred_bbox, gt=groundtruth[idx], fig_n=1, cvt_code=cv2.COLOR_RGB2BGR,idx=idx)
                  show_rgbt_image(t_img, pred_bbox, gt=groundtruth[idx], fig_n=2, cvt_code=cv2.COLOR_RGB2GRAY,idx=idx)
        # # save results
        toc /= cv2.getTickFrequency()
        model_path = os.path.join('results', args.dataset, model_name)
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        result_path = os.path.join(model_path, '{}.txt'.format(video_name))
        with open(result_path, 'w') as f:
            for x in pred_bboxes:
                f.write(','.join([str(i) for i in x])+'\n')
        print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
            v_idx+1, video_name, toc, idx / toc))
    os.chdir(model_path)
    save_file = '../%s' % dataset
    shutil.make_archive(save_file, 'zip')
    print('Records saved at', save_file + '.zip')


def show_rgbt_image(img, boxes=None, gt=None, box_fmt='ltwh', colors=None,
               thickness=3, fig_n=1, delay=1, visualize=True,
               cvt_code=cv2.COLOR_RGB2BGR,idx=0):
    if cvt_code is not None:
        img = cv2.cvtColor(img, cvt_code)
    idx=idx
    # resize img if necessary
    max_size = 960
    if max(img.shape[:2]) > max_size:
        scale = max_size / max(img.shape[:2])
        out_size = (
            int(img.shape[1] * scale),
            int(img.shape[0] * scale))
        img = cv2.resize(img, out_size)
        if boxes is not None:
            boxes = np.array(boxes, dtype=np.float32) * scale
        if gt is not None:
            gt = np.array(gt, dtype=np.float32) * scale

    if boxes is not None:
        assert box_fmt in ['ltwh', 'ltrb']
        boxes = np.array(boxes, dtype=np.int32)
        if boxes.ndim == 1:
            boxes = np.expand_dims(boxes, axis=0)
        if box_fmt == 'ltrb':
            boxes[:, 2:] -= boxes[:, :2]
    if gt is not None:
        assert box_fmt in ['ltwh', 'ltrb']
        gt = np.array(gt, dtype=np.int32)
        if gt.ndim == 1:
            gt = np.expand_dims(gt, axis=0)
        if box_fmt == 'ltrb':
            gt[:, 2:] -= gt[:, :2]

        # clip bounding boxes
        bound = np.array(img.shape[1::-1])[None, :]
        boxes[:, :2] = np.clip(boxes[:, :2], 0, bound)
        boxes[:, 2:] = np.clip(boxes[:, 2:], 0, bound - boxes[:, :2])
        gt[:, :2] = np.clip(gt[:, :2], 0, bound)
        gt[:, 2:] = np.clip(gt[:, 2:], 0, bound - gt[:, :2])

        if colors is None:
            colors = [
                (0, 0, 255),
                (0, 255, 0),
                (255, 0, 0),
                (0, 255, 255),
                (255, 0, 255),
                (255, 255, 0),
                (0, 0, 128),
                (0, 128, 0),
                (128, 0, 0),
                (0, 128, 128),
                (128, 0, 128),
                (128, 128, 0)]
        colors = np.array(colors, dtype=np.int32)
        if colors.ndim == 1:
            colors = np.expand_dims(colors, axis=0)

        for i, box in enumerate(boxes):
            color = colors[i % len(colors)]
            pt1 = (box[0], box[1])
            pt2 = (box[0] + box[2], box[1] + box[3])
            gt1 = (gt[i][0], gt[i][1])
            gt2 = (gt[i][0] + gt[i][2], gt[i][1] + gt[i][3])
            img = cv2.rectangle(img, pt1, pt2, color.tolist(), thickness)
            img = cv2.rectangle(img, gt1, gt2, (255, 215, 0), 2)

    if visualize:
        winname = 'window_{}'.format(fig_n)
        cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow(winname, img)
        cv2.waitKey(delay)

    return img

if __name__ == '__main__':
    main()
