from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import cv2

from snot.core.config_car import cfg
from snot.models.siamcar_model import ModelBuilderCAR
from snot.trackers.siamcar_tracker import SiamCARTracker
from snot.utils.bbox import get_axis_aligned_bbox
from snot.utils.model_load import load_pretrain
from snot.utils.misc import bbox_clip


class SiamCARPipeline():
    def __init__(self, args, enhancer=None, denoiser=None):
        super(SiamCARPipeline, self).__init__()
        if not args.config:
            args.config = './experiments/SiamCAR/config.yaml'
        if not args.snapshot:
            args.snapshot = './experiments/SiamCAR/model.pth'

        cfg.merge_from_file(args.config)
        self.model = ModelBuilderCAR()
        self.model = load_pretrain(self.model, args.snapshot).cuda().eval()
        self.enhancer = enhancer
        self.denoiser = denoiser
        self.tracker = DNS_SiamCARTracker(self.model, self.enhancer, self.denoiser, cfg.TRACK)

        self.hp = {'lr': 0.4, 'penalty_k': 0.2, 'window_lr': 0.3}

    def init(self, img, gt_bbox):
        cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
        gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
        self.tracker.init(img, gt_bbox_)
        pred_bbox = gt_bbox_

        return pred_bbox
    
    def track(self, img):
        outputs = self.tracker.track(img, self.hp)  
        pred_bbox = outputs['bbox']

        return pred_bbox

class DNS_SiamCARTracker(SiamCARTracker):
    def __init__(self, model, enhancer, denoiser, cfg):
        super(DNS_SiamCARTracker, self).__init__(model, cfg)
        
        self.model = model
        self.model.eval()

        self.denoiser = denoiser
        self.enhancer = enhancer

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        if self.denoiser is not None:
            z_crop = self.denoiser.denoise(z_crop)
        if self.enhancer is not None:
            z_crop = self.enhancer.enhance(z_crop)
        self.model.template(z_crop)
        
    def track(self, img, hp):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        self.scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)
        if self.denoiser is not None:
            x_crop = self.denoiser.denoise(x_crop)
        if self.enhancer is not None:
            x_crop = self.enhancer.enhance(x_crop)
        
        outputs = self.model.track(x_crop)
        cls = self._convert_cls(outputs['cls']).squeeze()
        cen = outputs['cen'].data.cpu().numpy()
        cen = (cen - cen.min()) / cen.ptp()
        cen = cen.squeeze()
        lrtbs = outputs['loc'].data.cpu().numpy().squeeze()

        upsize = (cfg.TRACK.SCORE_SIZE-1) * cfg.TRACK.STRIDE + 1
        penalty = self.cal_penalty(lrtbs, hp['penalty_k'])
        p_score = penalty * cls * cen
        if cfg.TRACK.hanming:
            hp_score = p_score*(1 - hp['window_lr']) + self.window * hp['window_lr']
        else:
            hp_score = p_score

        hp_score_up = cv2.resize(hp_score, (upsize, upsize), interpolation=cv2.INTER_CUBIC)
        p_score_up = cv2.resize(p_score, (upsize, upsize), interpolation=cv2.INTER_CUBIC)
        cls_up = cv2.resize(cls, (upsize, upsize), interpolation=cv2.INTER_CUBIC)
        lrtbs = np.transpose(lrtbs,(1,2,0))
        lrtbs_up = cv2.resize(lrtbs, (upsize, upsize), interpolation=cv2.INTER_CUBIC)

        scale_score = upsize / cfg.TRACK.SCORE_SIZE
        # get center
        max_r_up, max_c_up, new_cx, new_cy = self.getCenter(hp_score_up, p_score_up, scale_score, lrtbs)
        # get w h
        ave_w = (lrtbs_up[max_r_up,max_c_up,0] + lrtbs_up[max_r_up,max_c_up,2]) / self.scale_z
        ave_h = (lrtbs_up[max_r_up,max_c_up,1] + lrtbs_up[max_r_up,max_c_up,3]) / self.scale_z

        s_c = self.change(self.sz(ave_w, ave_h) / self.sz(self.size[0]*self.scale_z, self.size[1]*self.scale_z))
        r_c = self.change((self.size[0] / self.size[1]) / (ave_w / ave_h))
        penalty = np.exp(-(r_c * s_c - 1) * hp['penalty_k'])
        lr = penalty * cls_up[max_r_up, max_c_up] * hp['lr']
        new_width = lr*ave_w + (1-lr)*self.size[0]
        new_height = lr*ave_h + (1-lr)*self.size[1]

        # clip boundary
        cx = bbox_clip(new_cx,0,img.shape[1])
        cy = bbox_clip(new_cy,0,img.shape[0])
        width = bbox_clip(new_width,0,img.shape[1])
        height = bbox_clip(new_height,0,img.shape[0])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])
        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]

        return {
                'bbox': bbox,
               }
