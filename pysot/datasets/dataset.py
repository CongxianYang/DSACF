# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import random
import json
import logging
import sys
import os
from collections import namedtuple
Corner = namedtuple('Corner', 'x1 y1 x2 y2')
import cv2
import numpy as np
from torch.utils.data import Dataset
from pysot.utils.bbox import center2corner, Center
from pysot.datasets.augmentation import Augmentation
from pysot.core.config import cfg
logger = logging.getLogger("global")
import glob
import six



# setting opencv
pyv = sys.version[0]
if pyv[0] == '3':
    cv2.ocl.setUseOpenCL(False)


class SubDataset(object):
    def __init__(self, name, root,list, frame_range, num_use, start_idx):
        cur_path = os.path.dirname(os.path.realpath(__file__))
        self.name = name
        self.root = root
        self.list = list
        self.max_inter=100#like T in SiamFC,The images are extracted from two frames of a video that both contain the object and are at most T frames apart
        self.frame_range = frame_range
        self.num_use = num_use
        self.start_idx = start_idx
        logger.info("loading " + name)
        # with open(self.anno, 'r') as f:
        #     meta_data = json.load(f)
        #     meta_data = self._filter_zero(meta_data)
        #
        # for video in list(meta_data.keys()):
        #     for track in meta_data[video]:
        #         frames = meta_data[video][track]
        #         frames = list(map(int,
        #                       filter(lambda x: x.isdigit(), frames.keys())))
        #         frames.sort()
        #         meta_data[video][track]['frames'] = frames
        #         if len(frames) <= 0:
        #             logger.warning("{}/{} has no frames".format(video, track))
        #             del meta_data[video][track]
        #
        # for video in list(meta_data.keys()):
        #     if len(meta_data[video]) <= 0:
        #         logger.warning("{} has no tracks".format(video))
        #         del meta_data[video]
        self._check_integrity(root, list)
        list_file = list
        with open(list_file, 'r') as f:
            self.seq_names = f.read().strip().split('\n')
        # if(name=='GTOT'):
        #     self.seq_dirs_rgb = [os.path.join(root, s, 'v')
        #                          for s in self.seq_names]
        #     self.seq_dirs_t = [os.path.join(root, s, 'i')
        #                        for s in self.seq_names]
        # else:
        self.seq_dirs_rgb = [os.path.join(root, s, 'visible')
                              for s in self.seq_names]
        self.seq_dirs_t = [os.path.join(root, s, 'infrared')
                            for s in self.seq_names]
        self.rgb_anno_files = [os.path.join(root, s, 'visible.txt')
                               for s in self.seq_names]
        self.t_anno_files = [os.path.join(root, s, 'infrared.txt')  ##跟init一样
                             for s in self.seq_names]

        #self.labels = meta_data
        self.num = len(self.seq_names)
        self.num_use = self.num if self.num_use == -1 else self.num_use
        #self.videos = list(meta_data.keys())
        logger.info("{} loaded".format(self.name))
        self.path_format = '{}.{}.{}.jpg'
        self.pick = self.shuffle()

    def _check_integrity(self, root_dir, list='rgbt234.txt'):
        list_file = list
        if os.path.isfile(list_file):
            with open(list_file, 'r') as f:
                seq_names = f.read().strip().split('\n')
            for seq_name in seq_names:
                seq_dir = os.path.join(root_dir, seq_name)
                if not os.path.isdir(seq_dir):
                    print('Warning: sequence %s not exists.' % seq_name)
        else:
            raise Exception('Dataset not found or corrupted.')

    def _filter_zero(self, meta_data):
        meta_data_new = {}
        for video, tracks in meta_data.items():
            new_tracks = {}
            for trk, frames in tracks.items():
                new_frames = {}
                for frm, bbox in frames.items():
                    if not isinstance(bbox, dict):
                        if len(bbox) == 4:
                            x1, y1, x2, y2 = bbox
                            w, h = x2 - x1, y2 - y1
                        else:
                            w, h = bbox
                        if w <= 0 or h <= 0:
                            continue
                    new_frames[frm] = bbox
                if len(new_frames) > 0:
                    new_tracks[trk] = new_frames
            if len(new_tracks) > 0:
                meta_data_new[video] = new_tracks
        return meta_data_new

    def log(self):
        logger.info("{} start-index {} select [{}/{}] path_format {}".format(
            self.name, self.start_idx, self.num_use,
            self.num, self.path_format))

    def shuffle(self):
        lists = list(range(self.start_idx, self.start_idx + self.num))
        pick = []
        while len(pick) < self.num_use:
            np.random.shuffle(lists)
            pick += lists
        return pick[:self.num_use]
##
    # def get_image_anno(self, video, track, frame):
    #     image_path = os.path.join(self.root, video,
    #                               self.path_format.format(frame, track, 'x'))
    #     image_anno = self.labels[video][track][frame]
    #     return image_path, image_anno
##
    # def get_positive_pair(self, index):
    #     video_name = self.videos[index]
    #     video = self.labels[video_name]
    #     track = np.random.choice(list(video.keys()))
    #     track_info = video[track]
    #
    #     frames = track_info['frames']
    #     template_frame = np.random.randint(0, len(frames))
    #     left = max(template_frame - self.frame_range, 0)
    #     right = min(template_frame + self.frame_range, len(frames)-1) + 1
    #     search_range = frames[left:right]
    #     template_frame = frames[template_frame]
    #     search_frame = np.random.choice(search_range)
    #     return self.get_image_anno(video_name, track, template_frame), \
    #         self.get_image_anno(video_name, track, search_frame)
##
    # def get_random_target(self, index=-1):
    #     if index == -1:
    #         index = np.random.randint(0, self.num)
    #     video_name_rgb= self.seq_dirs_rgb[index]
    #     video_name_t = self.seq_dirs_t[index]
    #     video_num=len(video_name_rgb)
    #     video_gt_rgb = self.rgb_anno_files[index]
    #     video_gt_t = self.t_anno_files[index]
    #     track = np.random.choice(video_num)
    #     track_info = [track]
    #     frames = track_info['frames']
    #     frame = np.random.choice(frames)
    #     return self.get_image_anno(video_name, track, frame)

    def __len__(self):
        return self.num
##
    def _pick_rgb_and_t_pairs(self, index):
        assert index < len(self.seq_names), 'index_of_subclass should less than total classes'
        # video_name_rgb = self.seq_dirs_rgb[index_of_subclass]
        # video_name_t = self.seq_dirs_t[index_of_subclass]
        # video_num = len(video_name_rgb)
        # video_gt_rgb = self.rgb_anno_files[index_of_subclass]
        # video_gt_t = self.t_anno_files[index_of_subclass]
        if isinstance(index, six.string_types):
            if not index in self.seq_names:
                raise Exception('Sequence {} not found.'.format(index))
            index =self.seq_names.index(index)

        img_files_rgb = sorted(glob.glob(os.path.join(
                   self.seq_dirs_rgb[index], '*.*')))
        img_files_t = sorted(glob.glob(os.path.join(
                   self.seq_dirs_t[index], '*.*')))

        rgb_anno = np.loadtxt(self.rgb_anno_files[index], delimiter=',')
        t_anno = np.loadtxt(self.t_anno_files[index], delimiter=',')
        assert len(img_files_rgb) == len(img_files_t) and len(rgb_anno) == len(t_anno) \
               and len(img_files_t) == len(rgb_anno)
        video_num = len(img_files_rgb)
        status = True
        while status:
            if self.max_inter >= video_num - 1:
                self.max_inter = video_num // 2
            template_index = np.clip(random.choice(range(0, max(1, video_num - self.max_inter))), 0, video_num - 1) # limit template_index from 0 to video_num - 1
            detection_index= np.clip(random.choice(range(1, max(2, self.max_inter))) + template_index, 0, video_num - 1)# limit detection_index from 0 to video_num - 1
            template_path_rgb, detection_path_rgb  =  img_files_rgb [template_index],  img_files_rgb [detection_index]
            template_path_t, detection_path_t =  img_files_t[template_index],  img_files_t[detection_index]

            # print(template_path_rgb)
            # print(detection_path_rgb)

            template_gt_rgb  =  rgb_anno[template_index]
            detection_gt_rgb =  rgb_anno[detection_index]
            template_gt_t = t_anno[template_index]
            detection_gt_t = t_anno[detection_index]

            if template_gt_rgb[2] * template_gt_rgb[3] * detection_gt_rgb[2] * detection_gt_rgb[3] \
                    * template_gt_t[2] * template_gt_t[3] * detection_gt_t[2] * detection_gt_t[3] != 0:
                status = False

        # load infomation of template and detection
        template_target_anno_rgb=template_gt_rgb
        detection_target_anno_rgb=detection_gt_rgb
        template_target_anno_t=template_gt_t
        detection_target_anno_t=detection_gt_t

        return template_path_rgb, detection_path_rgb,template_path_t, detection_path_t,\
            template_target_anno_rgb,detection_target_anno_rgb, template_target_anno_t,detection_target_anno_t

class TrkDataset(Dataset):
    def __init__(self,):
        super(TrkDataset, self).__init__()

        # create sub dataset
        self.all_dataset = []
        start = 0
        self.num = 0
        for name in cfg.DATASET.NAMES:
            subdata_cfg = getattr(cfg.DATASET, name)
            sub_dataset = SubDataset(
                    name,
                    subdata_cfg.ROOT,
                    subdata_cfg.LIST,
                    subdata_cfg.FRAME_RANGE,
                    subdata_cfg.NUM_USE,
                    start
                )
            start += sub_dataset.num
            self.num += sub_dataset.num_use

            sub_dataset.log()
            self.all_dataset.append(sub_dataset)

        # data augmentation
        self.template_aug = Augmentation(
                cfg.DATASET.TEMPLATE.SHIFT,
                cfg.DATASET.TEMPLATE.SCALE,
                cfg.DATASET.TEMPLATE.BLUR,
                cfg.DATASET.TEMPLATE.FLIP,
                cfg.DATASET.TEMPLATE.COLOR
            )
        self.search_aug = Augmentation(
                cfg.DATASET.SEARCH.SHIFT,
                cfg.DATASET.SEARCH.SCALE,
                cfg.DATASET.SEARCH.BLUR,
                cfg.DATASET.SEARCH.FLIP,
                cfg.DATASET.SEARCH.COLOR
            )
        videos_per_epoch = cfg.DATASET.VIDEOS_PER_EPOCH
        self.num = videos_per_epoch if videos_per_epoch > 0 else self.num
        self.num *= cfg.TRAIN.EPOCH
        self.pick = self.shuffle()

    def shuffle(self):
        pick = []
        m = 0
        while m < self.num:
            p = []
            for sub_dataset in self.all_dataset:
                sub_p = sub_dataset.pick
                p += sub_p
            np.random.shuffle(p)
            pick += p
            m = len(pick)
        logger.info("shuffle done!")
        logger.info("dataset length {}".format(self.num))
        return pick[:self.num]

    def _find_dataset(self, index):
        for dataset in self.all_dataset:
            if dataset.start_idx + dataset.num > index:
                return dataset, index - dataset.start_idx

    def _get_bbox(self, image, shape):
        imh, imw = image.shape[:2]
        if len(shape) == 4:
           #w, h = shape[2]-shape[0], shape[3]-shape[1]
           w, h = shape[2], shape[3]
        else:
            w, h = shape
        context_amount = 0.5
        exemplar_size = cfg.TRAIN.EXEMPLAR_SIZE
        wc_z = w + context_amount * (w+h)
        hc_z = h + context_amount * (w+h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        w = w*scale_z
        h = h*scale_z
        cx, cy = imw//2, imh//2
        #cx=shape[0]+w//2
        #cy=shape[1]+h//2
        bbox = center2corner(Center(cx, cy, w, h))
        return bbox
    def crop_hwc(self,image, bbox, out_sz, padding=(0, 0, 0)):
        a = (out_sz - 1) / (bbox[2] - bbox[0])
        b = (out_sz - 1) / (bbox[3] - bbox[1])
        c = -a * bbox[0]
        d = -b * bbox[1]
        mapping = np.array([[a, 0, c],
                            [0, b, d]]).astype(np.float)
        crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
        return crop

    def pos_s_2_bbox(self,pos, s):
        return [pos[0] - s / 2, pos[1] - s / 2, pos[0] + s / 2, pos[1] + s / 2]

    def crop_like_SiamFC(self,image, bbox, context_amount=0.5, exemplar_size=127, instanc_size=255, padding=(0, 0, 0)):
        target_pos = [bbox[0]+(bbox[2] / 2.),  bbox[1]+(bbox[3] / 2.)]
        target_size = [bbox[2] , bbox[3]]
        wc_z = target_size[1] + context_amount * sum(target_size)
        hc_z = target_size[0] + context_amount * sum(target_size)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        d_search = (instanc_size - exemplar_size) / 2
        pad = d_search / scale_z
        s_x = s_z + 2 * pad
       ## z = self.crop_hwc(image, self.pos_s_2_bbox(target_pos, s_z), exemplar_size, padding)
        x = self.crop_hwc(image, self.pos_s_2_bbox(target_pos, s_x), instanc_size, padding)
        return  x
    def __len__(self):
        return self.num

    def __getitem__(self, index):
        index = self.pick[index]
        dataset, index = self._find_dataset(index)
        gray = cfg.DATASET.GRAY and cfg.DATASET.GRAY > np.random.random()
        neg = cfg.DATASET.NEG and cfg.DATASET.NEG > np.random.random()
        # get one dataset
        if neg:
           template = dataset.get_random_target(index)
           search = np.random.choice(self.all_dataset).get_random_target()
        else:
           template_path_rgb, detection_path_rgb, template_path_t, detection_path_t, \
           template_target_anno_rgb, detection_target_anno_rgb, template_target_anno_t, detection_target_anno_t = dataset._pick_rgb_and_t_pairs(index)
      ##anno 标注序列第一帧的真值格式(x0,y0,box-width,box-height) 426,185,48,119 x0:left_top,  y0;right_bootom
        # get image
        template_image_rgb = cv2.imread(template_path_rgb)
        search_image_rgb = cv2.imread(detection_path_rgb)
        template_image_t = cv2.imread(template_path_t)
        search_image_t = cv2.imread(detection_path_t)
        if template_image_rgb is None:
            print('error image:',template_path_rgb)

        ##  crop image like siamFC
        ##  mean padding
        img_mean_rgb_tem = np.mean(template_image_rgb, axis=(0, 1))
        img_mean_t_tem = np.mean(template_image_t, axis=(0, 1))
        img_mean_rgb_sear = np.mean(search_image_rgb, axis=(0, 1))
        img_mean_t_sear=np.mean(search_image_t, axis=(0, 1))

        tem_img_rgb=self.crop_like_SiamFC(template_image_rgb,template_target_anno_rgb,instanc_size=511,padding=img_mean_rgb_tem)
        tem_img_t=self.crop_like_SiamFC(template_image_t,template_target_anno_t,instanc_size=511,padding=img_mean_t_tem)
        sear_img_rgb=self.crop_like_SiamFC(search_image_rgb,detection_target_anno_rgb,instanc_size=511,padding=img_mean_rgb_sear)
        sear_img_t=self.crop_like_SiamFC(search_image_t,detection_target_anno_t,instanc_size=511,padding=img_mean_t_sear)

        # get bounding box
        template_box_rgb = self._get_bbox(tem_img_rgb, template_target_anno_rgb)
        template_box_t=self._get_bbox(tem_img_t, template_target_anno_t)
        search_box_rgb = self._get_bbox(sear_img_rgb, detection_target_anno_rgb)
        search_box_t=self._get_bbox(sear_img_t,detection_target_anno_t)

        # bbox1 = template_box_rgb
        # cv2.rectangle(tem_img_rgb, (int(bbox1[0]), int(bbox1[1])),
        #               (int(bbox1[2]), int(bbox1[3])), (0, 0, 255), 2)
        # cv2.imwrite('/home/xiancong/桌面/cv_drwn0.jpg', tem_img_rgb)
        # bbox1 = search_box_rgb
        # cv2.rectangle(sear_img_rgb, (int(bbox1[0]), int(bbox1[1])),
        #               (int(bbox1[2]), int(bbox1[3])), (0, 0, 255), 2)
        # cv2.imwrite('/home/xiancong/桌面/cv_drwn00.jpg', sear_img_rgb)


        # augmentation
        #crop and  aug
        template_rgb, _ = self.template_aug(tem_img_rgb,
                                        template_box_rgb,
                                        cfg.TRAIN.EXEMPLAR_SIZE,
                                        gray=gray)
        # bbox2=_
        # cv2.rectangle(template_rgb, (int(bbox2[0]), int(bbox2[1])), (int(bbox2[2]), int(bbox2[3])), (0, 0, 255), 2)
        # cv2.imwrite('/home/xiancong/桌面/cv_drwn1.jpg', template_rgb)
        template_t, _ = self.template_aug(tem_img_t,
                                            template_box_t,
                                            cfg.TRAIN.EXEMPLAR_SIZE,
                                            gray=gray)
        # bbox2 = _
        # cv2.rectangle(template_t, (int(bbox2[0]), int(bbox2[1])), (int(bbox2[2]), int(bbox2[3])), (0, 0, 255), 2)
        # cv2.imwrite('/home/xiancong/桌面/cv_drwn11.jpg', template_t)
        ##
        # template_t, _ = self.search_aug(tem_img_t,
        #                                   template_box_t,
        #                                   cfg.TRAIN.EXEMPLAR_SIZE,
        #                                   gray=gray)
        search_rgb, bbox_rgb = self.search_aug(sear_img_rgb,
                                       search_box_rgb,
                                       cfg.TRAIN.SEARCH_SIZE,
                                       gray=gray)
        search_t, bbox_t = self.search_aug(sear_img_t,
                                       search_box_t,
                                       cfg.TRAIN.SEARCH_SIZE,
                                       gray=gray)
        # bbox2 = bbox_rgb
        # cv2.rectangle(search_rgb, (int(bbox2[0]), int(bbox2[1])), (int(bbox2[2]), int(bbox2[3])), (0, 0, 255), 2)
        # cv2.imwrite('/home/xiancong/桌面/cv_drwn3.jpg', search_rgb)
        # bbox2 = bbox_t
        # cv2.rectangle(search_t, (int(bbox2[0]), int(bbox2[1])), (int(bbox2[2]), int(bbox2[3])), (0, 0, 255), 2)
        # cv2.imwrite('/home/xiancong/桌面/cv_drwn33.jpg', search_t)
        # s=search_rgb
        cls = np.zeros((cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.OUTPUT_SIZE), dtype=np.int64)
        template_rgb =  template_rgb.transpose((2, 0, 1)).astype(np.float32)
        template_t  =  template_t.transpose((2, 0, 1)).astype(np.float32)
        search_rgb = search_rgb.transpose((2, 0, 1)).astype(np.float32)
        search_t = search_t.transpose((2, 0, 1)).astype(np.float32)
        #new_bbox=Corner((bbox_rgb.x1+bbox_t.x1)//2,(bbox_rgb.y1+bbox_t.y1)//2,(bbox_rgb.x2+bbox_t.x2)//2,(bbox_rgb.y2+bbox_t.y2)//2,)
        new_bbox=bbox_rgb  ##select rgb axis
        # bbox2 = new_bbox
        # cv2.rectangle(s, (int(bbox2[0]+1), int(bbox2[1]+1)), (int(bbox2[2]), int(bbox2[3])), (0, 255, 0), 2)
        # cv2.imwrite('/home/xiancong/桌面/cv_drwnff.jpg', s)
        return {
                'template_rgb': template_rgb,
                'template_t': template_t,
                'search_rgb': search_rgb,
                'search_t': search_t,
                'label_cls': cls,
                'bbox': np.array([new_bbox.x1,new_bbox.y1,new_bbox.x2,new_bbox.y2])
                }

