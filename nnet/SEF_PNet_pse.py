"""
Created on Sun June 2  2024
@author: Ziling Huang
"""
import torch as th 
import torch.nn as nn 
import torch.nn.functional as nn_f 
from typing import Tuple, List 
from memonger import SublinearSequential
from libs.conv_stft import ConvSTFT, ConviSTFT

def param(nnet, Mb=True): 
    """
    Return number parameters(not bytes) in nnet
    """
    neles = sum([param.nelement() for param in nnet.parameters()])
    return neles / 10**6 if Mb else neles

class Conv1D(nn.Conv1d): 
    """
    1D conv in ConvTasNet
    """

    def __init__(self, *args, **kwargs):
        super(Conv1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        """
        x: N x L or N x C x L
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else th.unsqueeze(x, 1))
        if squeeze:
            x = th.squeeze(x)
        return x

class ChannelWiseLayerNorm(nn.LayerNorm):
    """
    Channel wise layer normalization
    """

    def __init__(self, *args, **kwargs):
        super(ChannelWiseLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        x: N x C x T
        """
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        # N x C x T => N x T x C
        x = th.transpose(x, 1, 2)
        # LN
        x = super().forward(x)
        # N x C x T => N x T x C
        x = th.transpose(x, 1, 2)
        return x

class GlobalChannelLayerNorm(nn.Module):
    """
    Global channel layer normalization
    """

    def __init__(self, dim, eps=1e-05, elementwise_affine=True):
        super(GlobalChannelLayerNorm, self).__init__()
        self.eps = eps
        self.normalized_dim = dim
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.beta = nn.Parameter(th.zeros(dim, 1))
            self.gamma = nn.Parameter(th.ones(dim, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        """
        x: N x C x T
        """
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        # N x 1 x 1
        mean = th.mean(x, (1, 2), keepdim=True)
        var = th.mean((x - mean)**2, (1, 2), keepdim=True)
        # N x T x C
        if self.elementwise_affine:
            x = self.gamma * (x - mean) / th.sqrt(var + self.eps) + self.beta
        else:
            x = (x - mean) / th.sqrt(var + self.eps)
        return x

    def extra_repr(self):
        return "{normalized_dim}, eps={eps}, " \
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)

def build_norm(norm, dim):
    """
    Build normalize layer
    LN cost more memory than BN
    """
    if norm not in ["cLN", "gLN", "BN"]:
        raise RuntimeError("Unsupported normalize layer: {}".format(norm))
    if norm == "cLN":
        return ChannelWiseLayerNorm(dim, elementwise_affine=True)
    elif norm == "BN":
        return nn.BatchNorm1d(dim)
    else:
        return GlobalChannelLayerNorm(dim, elementwise_affine=True)

class Conv2dBlock(nn.Module):
    def __init__(self, 
                 in_dims: int = 16,
                 out_dims: int = 32,
                 kernel_size: Tuple[int] = (3, 3),
                 stride: Tuple[int] = (1, 1),
                 padding: Tuple[int] = (1, 1)) -> None:
        super(Conv2dBlock, self).__init__() 
        self.conv2d = nn.Conv2d(in_dims, out_dims, kernel_size, stride, padding)     
        self.elu = nn.ELU()
        self.norm = nn.InstanceNorm2d(out_dims)
        
    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.conv2d(x)
        x = self.elu(x)
        return self.norm(x)

class ConvTrans2dBlock(nn.Module):
    def __init__(self, 
                 in_dims: int = 32,
                 out_dims: int = 16,
                 kernel_size: Tuple[int] = (3, 3),
                 stride: Tuple[int] = (1, 2),
                 padding: Tuple[int] = (1, 0),
                 output_padding: Tuple[int] = (0, 0)) -> None:
        super(ConvTrans2dBlock, self).__init__() 
        self.convtrans2d = nn.ConvTranspose2d(in_dims, out_dims, kernel_size, stride, padding, output_padding)     
        self.elu = nn.ELU()
        self.norm = nn.InstanceNorm2d(out_dims)
        
    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.convtrans2d(x)
        x = self.elu(x)
        return self.norm(x)
        
class DenseBlock(nn.Module):
    def __init__(self, in_dims, out_dims, mode = "enc", **kargs):
        super(DenseBlock, self).__init__()
        if mode not in ["enc", "dec"]:
            raise RuntimeError("The mode option must be 'enc' or 'dec'!")
            
        n = 1 if mode == "enc" else 2
        self.conv1 = Conv2dBlock(in_dims=in_dims*n, out_dims=in_dims, **kargs)
        self.conv2 = Conv2dBlock(in_dims=in_dims*(n+1), out_dims=in_dims, **kargs)
        self.conv3 = Conv2dBlock(in_dims=in_dims*(n+2), out_dims=in_dims, **kargs)
        self.conv4 = Conv2dBlock(in_dims=in_dims*(n+3), out_dims=in_dims, **kargs)
        self.conv5 = Conv2dBlock(in_dims=in_dims*(n+4), out_dims=out_dims, **kargs)
        
    def forward(self, x: th.Tensor) -> th.Tensor:
        y1 = self.conv1(x)
        y2 = self.conv2(th.cat([x, y1], 1))
        y3 = self.conv3(th.cat([x, y1, y2], 1))
        y4 = self.conv4(th.cat([x, y1, y2, y3], 1))
        y5 = self.conv5(th.cat([x, y1, y2, y3, y4], 1))
        return y5
           
class TCNBlock(nn.Module):
    """
    TCN block:
        IN - ELU - Conv1D - IN - ELU - Conv1D
    """

    def __init__(self,
                 in_dims: int = 384,
                 out_dims: int = 384,
                 kernel_size: int = 3,
                 stride: int = 1,
                 paddings: int = 1,
                 dilation: int = 1,
                 causal: bool = False) -> None:
        super(TCNBlock, self).__init__()
        self.norm1 = nn.InstanceNorm1d(in_dims)
        self.elu1 = nn.ELU()
        dconv_pad = (dilation * (kernel_size - 1)) // 2 if not causal else (
            dilation * (kernel_size - 1))
        # dilated conv
        self.dconv1 = nn.Conv1d(
            in_dims,
            out_dims,
            kernel_size,
            padding=dconv_pad,
            dilation=dilation,
            groups=in_dims,
            bias=True)
        
        self.norm2 = nn.InstanceNorm1d(in_dims)
        self.elu2 = nn.ELU()    
        self.dconv2 = nn.Conv1d(in_dims, out_dims, 1, bias=True)
        
        # different padding way
        self.causal = causal
        self.dconv_pad = dconv_pad

    def forward(self, x: th.Tensor) -> th.Tensor:
        y = self.elu1(self.norm1(x))
        y = self.dconv1(y)
        if self.causal:
            y = y[:, :, :-self.dconv_pad]
        y = self.elu2(self.norm2(y))
        y = self.dconv2(y)    
        x = x + y
        
        return x 


class LCA(nn.Module):
    def __init__(self, channels=64, r=4):
        super(LCA, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        return x * wei

class IFI(nn.Module):

    def __init__(self, channels=64, r=4):
        super(IFI, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x * wei + residual * (1 - wei)

        xl2 = self.local_att2(xi)
        xg2 = self.global_att(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + residual * (1 - wei2)
        return xo

class SEF_PNet(nn.Module):
    def __init__(self, 
                 win_len: int = 256,    # 32 ms
                 win_inc: int = 64,    # 8 ms
                 fft_len: int = 256,
                 win_type: str = "sqrthann",                 
                 kernel_size: Tuple[int] = (3, 3),
                 stride1: Tuple[int] = (1, 1),
                 stride2: Tuple[int] = (1, 2),
                 paddings: Tuple[int] = (1, 0),
                 output_padding: Tuple[int] = (0, 0),
                 tcn_dims: int = 384,
                 tcn_blocks: int = 10,
                 tcn_layers: int = 2,
                 causal: bool = False,     
                 pool_size: Tuple[int] = (4, 8, 16, 32),
                 num_spks: int = 1,
                 L: int = 20) -> None:
        super(SEF_PNet, self).__init__()
        
        self.L = L
        self.fft_len = fft_len
        self.num_spks = num_spks
        self.stft = ConvSTFT(win_len, win_inc, fft_len, win_type, 'complex')
        self.softmax = nn.Softmax(dim=-2)
        self.ifi = IFI(channels=2, r=1/32)
        self.upconv1 = nn.Conv2d(4, 64, 1, 1, 0)
        self.lca = LCA(64)
        self.conv2d = nn.Conv2d(64, 16, (1, 3), 1, 0)
        self.relu = nn.ReLU() 
        self.encoder = self._build_encoder(
                    kernel_size=kernel_size,
                    stride=stride2,
                    padding=paddings
                )
        self.tcn_layers = self._build_tcn_layers(
                    tcn_layers,
                    tcn_blocks,
                    in_dims=tcn_dims,
                    out_dims=tcn_dims,
                    causal=causal                         
                )
        self.decoder = self._build_decoder(
                    kernel_size=kernel_size,
                    stride=stride2,
                    padding=paddings,
                    output_padding=output_padding
                )
        self.avg_pool = self._build_avg_pool(pool_size)
        self.avg_proj = nn.Conv2d(64, 32, 1, 1)
        self.deconv2d = nn.ConvTranspose2d(32, 2*num_spks, kernel_size, stride1, paddings)
        self.istft = ConviSTFT(win_len, win_inc, fft_len, win_type, 'complex')
        
    def _build_encoder(self, **enc_kargs):
        """
        Build encoder layers 
        """
        encoder = nn.ModuleList()
        encoder.append(SublinearSequential(DenseBlock(16, 16, "enc"),LCA(16)))

        for i in range(3):
            encoder.append(
                    SublinearSequential(
                            Conv2dBlock(in_dims=16 if i==0 else 32, 
                                    out_dims=32, **enc_kargs),
                            DenseBlock(32, 32, "enc"),
                            LCA(32)
                            )
                    )
        encoder.append(
            SublinearSequential(
                Conv2dBlock(in_dims=32, out_dims=64, **enc_kargs),
                LCA(64)
            )
        )
        encoder.append(
            SublinearSequential(
                Conv2dBlock(in_dims=64, out_dims=128, **enc_kargs),
                LCA(128)
            )
        )
        encoder.append(
            SublinearSequential(
                Conv2dBlock(in_dims=128, out_dims=384, **enc_kargs),
                LCA(384)
            )
        )

        return encoder
    
    def _build_decoder(self, **dec_kargs):
        """
        Build decoder layers 
        """
        decoder = nn.ModuleList()
        decoder.append(ConvTrans2dBlock(in_dims=384*2, out_dims=128, **dec_kargs))
        decoder.append(ConvTrans2dBlock(in_dims=128*2, out_dims=64, **dec_kargs))
        decoder.append(ConvTrans2dBlock(in_dims=64*2, out_dims=32, **dec_kargs))        
        for i in range(3):
            decoder.append(
                    SublinearSequential(
                            DenseBlock(32, 64, "dec"),
                            ConvTrans2dBlock(in_dims=64, 
                                             out_dims=32  if i!=2 else 16,
                                             **dec_kargs)
                            )
                    )
        decoder.append(DenseBlock(16, 32, "dec"))                            
        
        return decoder    
    
    def _build_tcn_blocks(self, tcn_blocks, **tcn_kargs):
        """
        Build TCN blocks in each repeat (layer)
        """
        blocks = [
            TCNBlock(**tcn_kargs, dilation=(2**b))
            for b in range(tcn_blocks)
        ]
        
        return SublinearSequential(*blocks)
    
    def _build_tcn_layers(self, tcn_layers, tcn_blocks, **tcn_kargs):
        """
        Build TCN layers
        """
        layers = [
            self._build_tcn_blocks(tcn_blocks, **tcn_kargs)
            for _ in range(tcn_layers)
        ]
        
        return SublinearSequential(*layers)
    
    def _build_avg_pool(self, pool_size):
        """
        Build avg pooling layers
        """
        avg_pool = nn.ModuleList()
        for sz in pool_size:
            avg_pool.append(
                    SublinearSequential(
                            nn.AvgPool2d(sz),
                            nn.Conv2d(32, 8, 1, 1)                            
                            )
                )
        
        return avg_pool

    def wav2spec(self, x: th.Tensor, mags: bool = False) -> th.Tensor:
        """
        convert waveform to spectrogram
        """
        # print(x.shape)
        assert x.dim() == 2  
        # x = x / th.std(x, -1, keepdims=True)        # variance normalization
        specs = self.stft(x)
        real = specs[:,:self.fft_len//2+1]
        imag = specs[:,self.fft_len//2+1:]
        spec = th.stack([real,imag], 1) #[B,2,F,T]
        # spec = th.einsum("hijk->hikj", spec)    # batchsize, 2, T, F      
        if mags:
            return th.sqrt(real**2+imag**2+1e-8)
        else:
            return spec
        
    def FeaCompression(self, input, factor=0.5):
        input_change = input.float()
        complex_spectrum = th.complex(input_change[:, 0, :, :], input_change[:, 1, :, :])
        magnitude = th.abs(complex_spectrum).unsqueeze(1) ** factor
        phase = th.angle(complex_spectrum).unsqueeze(1)

        real = magnitude * th.cos(phase)
        imag = magnitude * th.sin(phase)
        output = th.cat((real, imag), dim=1)

        return output       
    
    def FeaDecompression(self, input, factor=0.5):
        input_change = input.float()
        complex_spectrum = th.complex(input_change[:, 0, :, :], input_change[:, 1, :, :])
        magnitude = th.abs(complex_spectrum).unsqueeze(1) ** (1 / factor)
        phase = th.angle(complex_spectrum).unsqueeze(1)

        real = magnitude * th.cos(phase)
        imag = magnitude * th.sin(phase)
        output = th.cat((real, imag), dim=1)

        return output

    def ComputeSimilarity(self, input, enrollment):
        att = enrollment.transpose(-2, -1) @ input
        att = self.softmax(att)
        output = enrollment @ att

        return output.unsqueeze(0).unsqueeze(0)
     
    def sep(self, spec: th.Tensor) -> List[th.Tensor]:
        """
        spec: (batchsize, 2, T, F)
        return [real, imag] or waveform
        """
        # spec = th.einsum("hijk->hikj", spec)        # (batchsize, 2, F, T)
        B, N, F, T = spec.shape
        est = th.chunk(spec, 2, 1)      # [(B, 1, F, T), (B, 1, F, T)]
        est = th.cat(est, 2).reshape(B, -1, T)      # B, 2F, T
        return th.squeeze(self.istft(est))
        
    def forward(self, 
                mix: th.Tensor, 
                enrollment: th.Tensor) -> th.Tensor:
        """
        if waveform = True, return both waveform and real & imag parts;
        else, only return real & imag parts
        """
        batch_size = mix.shape[0]
        if mix.dim() == 1:
            mix = th.unsqueeze(mix, 0)
            aux = th.unsqueeze(aux, 0)   
        mix_spec = self.wav2spec(mix, False)
        mix_spec_change = self.FeaCompression(mix_spec) #[B,2,F,T]
        similarity = []
        aux_drc = []
        for i in range(batch_size):
            aux = self.wav2spec(enrollment[i].unsqueeze(0), False)
            aux_spec_change = self.FeaCompression(aux)
            aux_drc.append(aux_spec_change)
            similarity.append(th.cat([self.ComputeSimilarity(mix_spec_change[i, 0, ...], aux_spec_change[0, 0, ...]), self.ComputeSimilarity(mix_spec_change[i, 1, ...], aux_spec_change[0, 1, ...])], dim=1))
        similarity = th.cat(similarity, dim=0)
        aux_drc = th.cat(aux_drc, dim=0)
        aux_drc = th.mean(aux_drc, dim=-1).unsqueeze(-1).repeat(1, 1,1, similarity.shape[-1])
        similarity = self.ifi(similarity, aux_drc)
        fus = th.cat((mix_spec_change, similarity), dim=1) #[1,4,129,251]
        fus = self.upconv1(fus)
        fus = self.lca(fus)
        # speech separation
        fus = fus.permute(0, 1, 3, 2)
        out = self.relu(self.conv2d(fus))     
        out_list = []
        out = self.encoder[0](out) 
        out_list.append(out) 
        for idx, enc in enumerate(self.encoder[1:]):
            out = enc(out)
            out_list.append(out)
        
        B, N, T, F = out.shape
        out = out.reshape(B, N, T*F)
        out = self.tcn_layers(out) 
        out = th.unsqueeze(out, -1)

        out_list = out_list[::-1] 
        for idx, dec in enumerate(self.decoder):
            decinput = th.cat([out_list[idx], out], 1)
            out = dec(decinput) 
          
        # Pyramidal pooling
        B, N, T, F = out.shape
        upsample = nn.Upsample(size=(T, F), mode='bilinear') 
        pool_list = []
        for avg in self.avg_pool:
            pool_list.append(upsample(avg(out)))
        out = th.cat([out, *pool_list], 1)
        out = self.avg_proj(out)
        out = self.deconv2d(out)
        out = out.permute(0, 1, 3, 2)
        out = self.FeaDecompression(out)
        out = self.sep(out)
        return out
    
       
def test_covn2d_block():
    x = th.randn(2, 16, 257, 200)
    conv = Conv2dBlock()
    y = conv(x)
    convtrans = ConvTrans2dBlock()
    z = convtrans(y)
    
def test_dense_block():
    x = th.randn(2, 16, 257, 200)
    dense = DenseBlock(16, 32, "enc")
    y = dense(x)
    
def test_tcn_block():
    x = th.randn(2, 384, 1000)
    tcn1 = TCNBlock(dilation=128)
    
if __name__ == "__main__":
    from thop import profile, clever_format
    nnet = SEF_PNet()
    mix = th.randn(2, 8000)
    aux = th.randn(2, 8000)
    est = nnet(mix, aux)
    macs, params = profile(nnet, inputs=(mix,aux))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs, params)
