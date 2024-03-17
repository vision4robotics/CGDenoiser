import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
import torchvision
from torchvision.transforms import Resize
torch_sizeup = Resize([256,256])

from einops import rearrange

##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

def reparametrize(mu,sigma):
    std = torch.exp(sigma)
    eps = torch.randn_like(std)
    z = mu+eps*std
    return z

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class Downsample(nn.Module):
    def __init__(self,dim):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(dim, dim,kernel_size=3, padding=1, stride=1),
            nn.PixelUnshuffle(2),
            nn.Conv2d(4*dim, 4*dim, kernel_size=3, padding=1, stride=1)
            )

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self,dim):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(dim, dim,kernel_size=3, padding=1, stride=1),
            nn.PixelShuffle(2),
            nn.Conv2d(dim//4, dim//4, kernel_size=3, padding=1, stride=1)
            )
        
    def forward(self, x):
        return self.body(x)

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

class CEB(nn.Module):
    def __init__(self, dim, num_blocks, ffn_expansion_factor = 2.66):
        super(CEB, self).__init__()

        self.extract = nn.Sequential(
            *[TransformerBlock(dim=dim, num_heads=4, ffn_expansion_factor=ffn_expansion_factor, bias=False, LayerNorm_type = 'WithBias') for i in range(num_blocks)]
            )

    def forward(self, x):
        x = x + self.extract(x)
        return x
    
class CE(nn.Module):

    def __init__(self, dim, num_blocks=[2,4], ffn_expansion_factor = 2.66):
        super(CE, self).__init__()
        
        self.expand = nn.Conv2d(3,dim,3,padding=1)
        self.ce1 = CEB(dim,num_blocks[0], ffn_expansion_factor)
        self.down1 = Downsample(dim)
        self.ce2 = CEB(dim*2**2, num_blocks[1],ffn_expansion_factor)
        self.down2 = Downsample(4*dim)
        self.inputdown1 = nn.PixelUnshuffle(2)
        self.inputdown2 = nn.PixelUnshuffle(4)


    def forward(self, x):
        x_in = self.expand(x)
        x1 = self.down1(self.ce1(x_in))
        x1 = self.inputdown1(x_in) + x1
        x2 = self.down2(self.ce2(x1))
        x2 = self.inputdown1(x1) + x2
        out = self.inputdown2(x_in)+x2

        return out

##########################################################################
## Residual decoder
class RDB(nn.Module):
    def __init__(self, dim, num_blocks, ffn_expansion_factor = 2.66):
        super(RDB, self).__init__()

        self.extract = nn.Sequential(
            *[TransformerBlock(dim=dim, num_heads=4, ffn_expansion_factor=ffn_expansion_factor, bias=False, LayerNorm_type = 'WithBias') for i in range(num_blocks)]
            )

    def forward(self, x):
        x = x + self.extract(x)
        return x

class Res_decoder(nn.Module):

    def __init__(self, dim, num_blocks=[4,2], ffn_expansion_factor = 2.66):
        super(Res_decoder, self).__init__()
        
        self.align = nn.Conv2d(16*dim+48, 16*dim, 3, padding=1)
        self.rd1 = RDB(dim*16,num_blocks[0], ffn_expansion_factor)
        self.up1 = Upsample(dim*16)
        self.rd2 = RDB(dim*4, num_blocks[1],ffn_expansion_factor)
        self.up2 = Upsample(dim*4)
        self.xup1 = nn.PixelShuffle(2)
        self.xup2 = nn.PixelShuffle(4)
        self.compress = nn.Conv2d(dim,3,3,padding=1)


    def forward(self, x):
        x_in = self.align(x)
        x1 = self.up1(self.rd1(x_in))
        x1 = self.xup1(x_in) + x1
        x2 = self.up2(self.rd2(x1))
        x2 = self.xup1(x1) + x2
        out = self.compress(self.xup2(x_in)+x2)

        return out
    

##########################################################################
## Attributive kernel learner

class Ak_learner(nn.Module):
    def __init__(self, kernel_size, kernel_num,dim):
        super(Ak_learner, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_size = kernel_size
        self.fc1 = nn.Linear(3*(kernel_num+1)*kernel_size**2, 3*(kernel_num+1)*kernel_size**2)
        self.fc2 = nn.Linear(3*(kernel_num+1)*kernel_size**2, 3*kernel_num*kernel_size**2)
        self.condition_kernelize = nn.Sequential( #64
            nn.Conv2d(16*dim, 3, kernel_size=6,stride=2), #30
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 3, kernel_size=6,stride=2), #13
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 3, kernel_size=3,stride=2), #6
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 3, kernel_size=4,stride=1), #3
            nn.ReLU(inplace=True),
        )
        
    
    def forward(self, poskernel, condition_map):   #x:[8*(3*num)*size*size]
        down_condition = self.condition_kernelize(condition_map)  #[8*3*size*size]
        x_y = torch.cat((poskernel,down_condition),1)   #[8,(3*(num+1)),size,size]

        inputs = x_y.reshape(x_y.shape[0], -1)
        h1 = F.relu(self.fc1(inputs))
        h2 = self.fc2(h1)
        kernels = h2.reshape(x_y.shape[0], 3*self.kernel_num, self.kernel_size, self.kernel_size)

        return kernels


##########################################################################
## Muti-kernel attributive refinement(MKAR)

class MKAR(nn.Module):
    def __init__(self, kernel_size, kernel_num):
        super(MKAR, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_size = kernel_size
        self.down = nn.ConvTranspose2d(3 * (1+kernel_num), 3, 1, 1)
    
    def forward(self, x, ak):
        aks = []
        for i in range(self.kernel_num):
            aks.append(ak[:, i*3:i*3+3])
        
        assert len(aks)==self.kernel_num
        
        xs = x
        for i in range(len(aks)):  
            x_new = xcorr_depthwise(x,aks[i])
            xs = torch.cat([xs,x_new],1)
        xs = self.down(xs)

        return xs

def xcorr_depthwise(x, kernel):
    """depthwise cross correlation
    """
    batch = kernel.size(0)
    channel = kernel.size(1)
    x = x.reshape(1, batch*channel, x.size(2), x.size(3))
    kernel = kernel.reshape(batch*channel, 1, kernel.size(2), kernel.size(3))
    out = F.conv2d(x, kernel,stride=1,padding='same', groups=batch*channel)
    out = out.reshape(batch, channel, out.size(2), out.size(3))
    return out


##########################################################################
## Posterior estimator

class PE(nn.Module):
    def __init__(self, kernel_num):
        super(PE, self).__init__()

        '''
        input: batch*3*256*256
        output: batch*48*64*64, batch*3num*3*3, kl_loss
        '''
        ak_miu_extract = nn.Sequential(
            nn.Conv2d(3*kernel_num, 3*kernel_num*2, kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(3*kernel_num*2, 3*kernel_num*4, kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(3*kernel_num*4, 3*kernel_num*2, kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(3*kernel_num*2, 3*kernel_num, kernel_size=3,padding=1),
            nn.ReLU(inplace=True)
            )
        miu_kernelize = nn.Sequential(
            nn.Conv2d(3*kernel_num, 3*kernel_num, kernel_size=6,stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(3*kernel_num, 3*kernel_num, kernel_size=6,stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(3*kernel_num, 3*kernel_num, kernel_size=3,stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(3*kernel_num, 3*kernel_num, kernel_size=3,stride=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(3*kernel_num, 3*kernel_num, kernel_size=4,stride=3),
            nn.ReLU(inplace=True),
            )
        ak_std_extract = nn.Sequential(
            nn.Conv2d(3*kernel_num, 3*kernel_num*2, kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(3*kernel_num*2, 3*kernel_num*4, kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(3*kernel_num*4, 3*kernel_num*2, kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(3*kernel_num*2, 3*kernel_num, kernel_size=3,padding=1),
            nn.ReLU(inplace=True)
            )
        std_kernelize = nn.Sequential(
            nn.Conv2d(3*kernel_num, 3*kernel_num, kernel_size=6,stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(3*kernel_num, 3*kernel_num, kernel_size=6,stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(3*kernel_num, 3*kernel_num, kernel_size=3,stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(3*kernel_num, 3*kernel_num, kernel_size=3,stride=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(3*kernel_num, 3*kernel_num, kernel_size=4,stride=3),
            nn.ReLU(inplace=True),
            )
        
        self.expand = nn.Sequential(nn.Conv2d(3, 3*kernel_num, 3, padding=1),
            nn.ReLU(inplace=True)
            )
        self.ak_estimator_miu = nn.Sequential(  #designed for image_size=256*256, kernel_size=3*3
            ak_miu_extract,
            miu_kernelize
        )
        self.ak_estimator_std = nn.Sequential(  #designed for image_size=256*256, kernel_size=3*3
            ak_std_extract,
            std_kernelize
        ) 

        self.map_extract_miu = nn.Sequential( 
            nn.Conv2d(3, 64, kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3,padding=1),
            nn.ReLU(inplace=True)
        ) 
        self.map_extract_std = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3,padding=1),
            nn.ReLU(inplace=True)
        ) 
        self.map_down = nn.PixelUnshuffle(4)
        

    def forward(self, clean_image):
        ak_x = self.expand(clean_image)
        miu_ak = self.ak_estimator_miu(ak_x)
        std_ak = self.ak_estimator_std(ak_x)
        miu_map = self.map_down(self.map_extract_miu(clean_image))
        std_map = self.map_down(self.map_extract_std(clean_image))
        
        ak_poster = reparametrize(miu_ak,std_ak)
        map_poster = reparametrize(miu_map,std_map)

        kl_loss_ak = -0.5 * (1 + 2*std_ak - miu_ak.pow(2) - torch.exp(2*std_ak))
        kl_loss_map = -0.5 * (1 + 2*std_map - miu_map.pow(2) - torch.exp(2*std_map))
        kl_loss = torch.sum(kl_loss_ak)+torch.sum(kl_loss_map)
        
        return map_poster, ak_poster, kl_loss








##---------------------------------- CGD ------------------------------------------##
class CGD(nn.Module):
    def __init__(self, 
                 for_train,
                 dim,
                 kernel_size,
                 kernel_num
    ):
        super(CGD, self).__init__()
        self.for_train = for_train
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.ce = CE(dim)
        self.residual_decoder = Res_decoder(dim)
        self.kernel_learner = Ak_learner(kernel_size, kernel_num,dim)
        self.refinement = MKAR(kernel_size, kernel_num)
        self.pe = PE(kernel_num)

    def forward(self, clean_img, inp_img):

        torch_sizedown = Resize([inp_img.shape[2],inp_img.shape[3]])
        resize = False
        if self.for_train == True:
            posterior_map, posterior_kernel, kl_loss = self.pe(clean_img)
        else:
            batch = inp_img.shape[0]
            origin_h = inp_img.shape[2]
            if origin_h != 256:
                inp_img = torch_sizeup(inp_img)
                resize = True
            posterior_map = torch.randn(batch,48,inp_img.shape[2]//4,inp_img.shape[3]//4).to(inp_img.device)
            posterior_kernel = torch.randn(batch, 3*self.kernel_num, self.kernel_size, self.kernel_size).to(inp_img.device)
        
        condition_map = self.ce(inp_img)
        noise_pred = self.residual_decoder(torch.cat([condition_map,posterior_map],dim=1))
        predeonised = inp_img-noise_pred
        ak = self.kernel_learner(posterior_kernel,condition_map)
        denoised = self.refinement(predeonised,ak)
        if resize == True:
            denoised = torch_sizedown(denoised)

        if self.for_train == True:
            return denoised, kl_loss
        else:
            return denoised
        


