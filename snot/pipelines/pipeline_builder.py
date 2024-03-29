from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals



from snot.pipelines.siamapn_pipeline import SiamAPNPipeline
from snot.pipelines.siamapnpp_pipeline import SiamAPNppPipeline
from snot.pipelines.siamban_pipeline import SiamBANPipeline
from snot.pipelines.siamgat_pipeline import SiamGATPipeline
from snot.pipelines.siamrpn_pipeline import SiamRPNppPipeline
from snot.pipelines.siamcar_pipeline import SiamCARPipeline
from snot.pipelines.siammask_pipeline import SiamMaskPipeline


TRACKERS =  {
          'SiamAPN': SiamAPNPipeline,
          'SiamAPN++': SiamAPNppPipeline,
          'SiamRPN++': SiamRPNppPipeline,
          'SiamBAN': SiamBANPipeline,
          'SiamGAT': SiamGATPipeline,
          'SiamCAR': SiamCARPipeline,
          'SiamMask': SiamMaskPipeline,
          }

def build_pipeline(args, enhancer, denoiser):
    return TRACKERS[args.trackername.split('_')[0]](args, enhancer, denoiser)

