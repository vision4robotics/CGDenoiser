from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torchvision
import torch


from snot.core.config import cfg
from snot.models.model_builder import ModelBuilder
from snot.trackers.siammask_tracker import SiamMaskTracker
from snot.utils.bbox import get_axis_aligned_bbox
from snot.utils.model_load import load_pretrain


class DNS_SiamMaskTracker(SiamMaskTracker):
    def __init__(self, model, enhancer=None, denoiser=None):
        super(DNS_SiamMaskTracker, self).__init__(model)
        
        self.model = model
        self.model.eval()

        self.enhancer = enhancer
        self.denoiser = denoiser

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

    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        
        #test = torchvision.io.read_image('/home/yucheng/project/CDT/demo/img_ori/00001.jpg')
        #test = torch.unsqueeze(test,0).cuda()
        #testd = self.denoiser.denoise(test)
        #torchvision.utils.save_image(testd/255,'denoised.png')

        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)
        
        #torchvision.utils.save_image(x_crop/255,'origin.png')
        if self.denoiser is not None:
            x_crop = self.denoiser.denoise(x_crop)
        #torchvision.utils.save_image(x_crop/255,'denoised.png')
        if self.enhancer is not None:
            x_crop = self.enhancer.enhance(x_crop)
        #torchvision.utils.save_image(x_crop/255,'denhanced.png')

        outputs = self.model.track(x_crop)
        score = self._convert_score(outputs['cls'])
        '''
        from heatmap import draw_feature_map
        feat = torch.from_numpy(score.reshape(1,5,25,25))
        image = x_crop.permute(0,2,3,1).cpu().detach().numpy()
        heat = draw_feature_map(feat, image)
        '''

        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0]*scale_z, self.size[1]*scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0]/self.size[1]) /
                     (pred_bbox[2, :]/pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            self.window * cfg.TRACK.WINDOW_INFLUENCE
        

        best_idx = np.argmax(pscore)

        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = score[best_idx]
        return {
                'bbox': bbox,
                'best_score': best_score
               }

class SiamMaskPipeline():
    def __init__(self, args, enhancer=None, denoiser=None):
        super(SiamMaskPipeline, self).__init__()
        if not args.config:
            args.config = './experiments/SiamMask_r50/config.yaml'
        if not args.snapshot:
            args.snapshot = './experiments/SiamMask_r50/model.pth'

        cfg.merge_from_file(args.config)
        self.model = ModelBuilder()
        self.model = load_pretrain(self.model, args.snapshot).cuda().eval()
        self.enhancer = enhancer
        self.denoiser = denoiser
        self.tracker = DNS_SiamMaskTracker(self.model, self.enhancer, self.denoiser)

    def init(self, img, gt_bbox):
        cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
        gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
        self.tracker.init(img, gt_bbox_)
        pred_bbox = gt_bbox_

        return pred_bbox
    
    def track(self, img):
        outputs = self.tracker.track(img)  
        pred_bbox = outputs['bbox']

        return pred_bbox

