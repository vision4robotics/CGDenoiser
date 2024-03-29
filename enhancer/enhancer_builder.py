import torch
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
import os

from .DCE_model import enhance_net_nopool as DCE

ENHANCERS = {
          'DCE': DCE,
         }

class Enhancer():
    def __init__(self, args):
        super(Enhancer, self).__init__()
        self.args = args

        if args.enhancername.split('-')[0]=='DCE':
            self.model = DCE(scale_factor=12) 
            self.model.load_state_dict(torch.load(args.e_weights))
            self.model.cuda().eval()
            
            
            

        
        
        
    def enhance(self, img):

        input_ = torch.div(img, 255.)
        if self.args.enhancername.split('-')[0]=='DCE':
            self.multiples = 12

            h,w = input_.shape[2], input_.shape[3]
            H,W = ((h+self.multiples)//self.multiples)*self.multiples, ((w+self.multiples)//self.multiples)*self.multiples
            padh = H-h if h%self.multiples!=0 else 0
            padw = W-w if w%self.multiples!=0 else 0
            input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

            enhanced,_ = self.model(input_)
            enhanced = enhanced[:,:,:h,:w]

        enhanced = torch.clamp(enhanced, 0, 1)

        return torch.mul(enhanced, 255.)


def build_enhancer(args):
    return Enhancer(args)

