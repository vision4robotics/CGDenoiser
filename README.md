# [CGDenoiser]
# Conditional Generative Denoiser for Nighttime UAV Tracking

### Yucheng Wang, Changhong Fu, Kunhan Lu, Liangliang Yao, and Haobo Zuo

This is the official code for the paper "Conditional Generative Denoiser for Nighttime UAV Tracking".

<img src="./CGD.pdf">

# Abstract

Abstract— State-of-the-art (SOTA) visual object tracking methods have greatly enhanced the autonomy of unmanned aerial vehicles (UAVs) in robotics applications. However, in low-light conditions, the presence of irregular real noise from the environments severely impacts the performance of these SOTA methods. Moreover, existing SOTA denoising techniques often fail to meet real-time processing requirements due to slow speeds. To design a more practical denoiser for UAV nighttime tracking, this work initiatively breaks free from traditional deterministic paradigm limitation and proposes a conditional generative denoiser (CGDenoiser), which instead generates the noise conditioning on the input and remove it. Specifically, a novel nested residual Transformer conditionalizer is developed to better align the dimension of inputs as well as quicken the rate of inference. Furthermore, an innovative multi-kernel conditional refiner is designed to pertinently refine the denoised output. Extensive experiments show that CGDenoiser promotes the tracking precision of the SOTA tracker by 18.18% on DarkTrack2021 whereas working 5.8 times faster than the second well-performed denoiser. Real-world tests with complex challenges also prove the effectiveness and practicality of CGDenoiser.


## Environment Preparing

```
python 3.7.3
pytorch 1.10.1
```

## Demo
### Demo Video
[![CDT](https://res.cloudinary.com/marcomontalbano/image/upload/v1680953500/video_to_markdown/images/youtube--tuoebVKYEVg-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://youtu.be/tuoebVKYEVg "CDT")


## Test

Download our pretrained model by [CDT](https://pan.baidu.com/s/1mgQI_wzjod9ElYklbxYwBQ?pwd=cdtn)(code:cdtn).

To evaluate the ability of CDT to overcome noise to improve UAV night tracking performance, you need to meet the enviroment requirements of base trackers and enhancers, as well as download their snapshots to corresponding folders at first. Details can be found in their repos. Currently supporting trackers including [SiamAPN](https://github.com/vision4robotics/SiamAPN), [SiamAPN++](https://github.com/vision4robotics/SiamAPN), [SiamRPN++](), [SiamBAN](https://github.com/hqucv/siamban), and [SiamGAT](https://github.com/ohhhyeahhh/SiamGAT), enhancers including [SCT](https://github.com/vision4robotics/SCT) and [DCE++](https://github.com/Li-Chongyi/Zero-DCE_extension).

Take the test of SiamAPN_SCT_CDT as an example:

```
python test.py                                    \
  --dataset UAVDark135                            \ # dataset_name
  --datasetpath ./test_dataset                    \ # dataset_path
  --config ./experiments/SiamAPN/config.yaml      \ # tracker_config
  --snapshot ./experiments/SiamAPN/model.pth      \ # tracker_model
  --trackername SiamAPN                           \ # tracker_name

  --e_weights ./experiments/SCT/model.pth         \ # enhancer_model
  --enhancername SCT                              \ # enhancer_name

  --d_weights ./experiments/CDT/model.pth         \ # denoiser_model
  --denoisername CDT                                # denoiser_name

```

## Train

Before training, you need to prepare the training set of the [SIDD](https://www.eecs.yorku.ca/~kamel/sidd/index.php) dataset. Run train.py.

```
python -m torch.distributed.launch --nproc_per_node=4 --master_port=12345 train.py -opt Denoising/Options/RealDenoising_CDT.yml --launcher pytorch
```



## Acknowledgements
- The code is implemented based on [SNOT](https://github.com/vision4robotics/SiameseTracking4UAV), [BasicSR](https://github.com/XPixelGroup/BasicSR), and [Restormer](https://github.com/swz30/Restormer). We would like to express our sincere thanks to the contributors.

- We would also like to thank [SiamMask](https://github.com/foolwood/SiamMask), [SiamCAR](https://github.com/ohhhyeahhh/SiamCAR), [SiamRPN++](), [SiamBAN](https://github.com/hqucv/siamban), and [SiamGAT](https://github.com/ohhhyeahhh/SiamGAT) trackers and [SCT](https://github.com/vision4robotics/SCT) and [DCE++](https://github.com/Li-Chongyi/Zero-DCE_extension) enhancers for their efforts.
