from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import argparse
import os
import faulthandler
faulthandler.enable()

import torch
import torch.nn.functional as F

from snot.pipelines.pipeline_builder import build_pipeline
from snot.datasets import DatasetFactory, datapath
from denoiser.denoiser_builder import build_denoiser
from enhancer.enhancer_builder import build_enhancer

torch.set_num_threads(1) 

parser = argparse.ArgumentParser(description='siamese tracking')
parser.add_argument('--dataset', default='UAVDark135', type=str,
                    help='datasets')
parser.add_argument('--datasetpath', default='/media/yucheng/Elements', type=str,
                    help='the path of datasets')
parser.add_argument('--config', default='', type=str,
                    help='config file')
parser.add_argument('--snapshot', default='', type=str,
                    help='snapshot of models to eval')
parser.add_argument('--trackername', default='SiamRPN++', type=str,
                    help='name of tracker')

parser.add_argument('--e_weights', default='./checkponit/DCE/model.pth', type=str,
                    help='weights')
parser.add_argument('--enhancername', default='DCE', type=str,
                    help='name of enhancer')

parser.add_argument('--d_weights', default='./checkponit/CGD/model.pth', type=str,
                    help='weights')
parser.add_argument('--denoisername', default='CGD', type=str,
                    help='name of denoiser')
parser.add_argument('--seed', default='6666', type=str,
                    help='random seed')


parser.add_argument('--video', default='girl5', type=str,
                    help='eval one special video')
parser.add_argument('--vis', default=True, action='store_true',
                    help='whether visualzie result')
parser.add_argument('--save_fig', default=True, action='store_true',
                    help='whether save result as image')
args = parser.parse_args()



def main():

    if args.seed:
        seed = args.seed
        torch.manual_seed(seed)
    if args.enhancername.split('-')[0]:
        enhancer = build_enhancer(args)
    else:
        enhancer = None
    if args.denoisername.split('-')[0]:
        denoiser = build_denoiser(args)
    else:
        denoiser = None
    pipeline = build_pipeline(args, enhancer=enhancer, denoiser=denoiser)


    for dataset_name in args.dataset.split(','):
        # create dataset
        try:
            dataset_root = args.datasetpath + datapath[dataset_name]
        except:
            print('datasetpath?')
        dataset = DatasetFactory.create_dataset(name=dataset_name,
                                            dataset_root=dataset_root,
                                            load_img=False)
        model_name = args.trackername+args.enhancername+args.denoisername
        
        # OPE tracking
        IDX = 0
        TOC = 0
        model_path = os.path.join('result', dataset_name, model_name)

        for v_idx, video in enumerate(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            toc = 0
            pred_bboxes = []
            for idx, (img, gt_bbox) in enumerate(video):
                tic = cv2.getTickCount()
                if idx == 0:
                    pred_bbox = pipeline.init(img, gt_bbox)
                    pred_bboxes.append(pred_bbox)
                else:
                    pred_bbox = pipeline.track(img)
                    pred_bboxes.append(pred_bbox)
                toc += cv2.getTickCount() - tic

                if args.vis and idx > 0:
                    try:
                        gt_bbox = list(map(int, gt_bbox))
                        cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 3)
                    except:
                        pass
                    pred_bbox = list(map(int, pred_bbox))
                    cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 3)
                    cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                                (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                    img_folder_path = os.path.join(model_path, "img", video.name)
                    if not os.path.exists(img_folder_path):
                        os.makedirs(img_folder_path)
                    img_path = os.path.join(img_folder_path, str(idx)+".png")
                    cv2.imwrite(img_path, img)

            toc /= cv2.getTickFrequency()
            # save results 
            if not os.path.isdir(model_path):
                os.makedirs(model_path)
            result_path = os.path.join(model_path, '{}.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    f.write(','.join([str(i) for i in x])+'\n')
            print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(v_idx+1, video.name, toc, idx / toc))
            IDX += idx
            TOC += toc
        print('Total Time: {:5.1f}s Average Speed: {:3.1f}fps'.format(TOC, IDX / TOC))

if __name__ == '__main__':
    main()