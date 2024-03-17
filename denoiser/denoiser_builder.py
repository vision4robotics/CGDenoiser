import torch
import torch.nn.functional as F
import numpy as np
import cv2


from basicsr.models.archs.CGD_arch import CGD


DENOISERS = {
          'CGD':CGD,
         }

class Denoiser():
    def __init__(self, model, args):
        super(Denoiser, self).__init__()
        
        self.model = model
        self.model.cuda()
        self.denoisername = args.denoisername

        checkpoint = torch.load(args.d_weights)
        
        
        self.model.load_state_dict(checkpoint['params'])
        model.eval()

        self.multiples = 8

    def denoise(self, img):
            
        with torch.no_grad():
            input_ = torch.div(img, 255.)

            # Pad the input if not_multiple_of 8
            h,w = input_.shape[2], input_.shape[3]
            H,W = ((h+self.multiples)//self.multiples)*self.multiples, ((w+self.multiples)//self.multiples)*self.multiples
            padh = H-h if h%self.multiples!=0 else 0
            padw = W-w if w%self.multiples!=0 else 0
            input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

            _ = torch.randn_like(input_)
            restored = self.model(_,input_)

            restored = torch.clamp(restored, 0, 1)

            # Unpad the output
            restored = restored[:,:,:h,:w]

            return torch.mul(restored, 255.)
    
    def single_denoise(self, img):
        img = (np.asarray(img)/255.0)
        img = torch.from_numpy(img).float()
        img = img.permute(2,0,1)
        input_ = img.cuda().unsqueeze(0)

        # Pad the input if not_multiple_of 8
        h,w = input_.shape[2], input_.shape[3]
        H,W = ((h+self.multiples)//self.multiples)*self.multiples, ((w+self.multiples)//self.multiples)*self.multiples
        padh = H-h if h%self.multiples!=0 else 0
        padw = W-w if w%self.multiples!=0 else 0
        

        restored = self.model(F.pad(input_, (0,padw,0,padh), 'reflect'))

        restored = torch.clamp(restored, 0, 1)

        # Unpad the output
        return restored[:,:,:h,:w]


def build_denoiser(args):

    parameters = {
                  'CGD':{'for_train':False,'dim':16,'kernel_size':3,'kernel_num':20},
                  }
    
    model = DENOISERS[args.denoisername.split('-')[0]](**parameters[args.denoisername.split('-')[0]])
    return Denoiser(model, args)

