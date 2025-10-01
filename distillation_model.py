import math
from dataclasses import dataclass
from typing import Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from mamba_ssm.modules.mamba2 import Mamba2

@dataclass
class MambaConfig:
    d_model: int  # D
    n_layers: int
    input_size: int = 256
    # scale: int
    in_channels: int = 3
    out_channels: int = 3
    # hide_channels: int


class Downsample(nn.Module):
    def __init__(self, config: MambaConfig, in_channels, out_channels,input_size,patch_size,layers):
        super().__init__()
        patch_size = 2
        self.in_channels = config.in_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
        self.x_embedder = PatchEmbed(input_size, patch_size, out_channels, config.d_model,bias=True)  
        self.pos_embed_x = nn.Parameter(torch.zeros(1, self.x_embedder.num_patches, config.d_model), requires_grad=False)
        self.layers = nn.ModuleList([MambaBlock(config) for _ in range(layers)])

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        self.final_layer = FinalLayer(config.d_model, patch_size, out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed_x = get_2d_sincos_pos_embed(self.pos_embed_x.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed_x.data.copy_(torch.from_numpy(pos_embed_x).float().unsqueeze(0))


        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Zero-out output layers:

        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        
    def unpatchify(self, x):
        c = self.in_channels
        p = self.x_embedder.patch_size[0] #2
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h*p, h*p))
        return imgs
        
    def forward(self, x):
        x = self.conv(x)  # [B, C, H, W] -> [B, C', H/2, W/2]
        x = self.norm(x)
        x = self.act(x)
        x = self.x_embedder(x) + self.pos_embed_x  # (N, T, D), where T = H * W / patch_size ** 
        for layer in self.layers:
            output = layer(x)
        x = self.final_layer(output)  # (N, T, patch_size ** 2 * out_channels)
        out = self.unpatchify(x)                   # (N, out_channels, H, W)
        
        return out



class Upsample(nn.Module):
    def __init__(self, config: MambaConfig, in_channels, out_channels,input_size,patch_size,layers):
        super().__init__()
        patch_size = 2
        self.in_channels = config.in_channels
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
        self.x_embedder = PatchEmbed(input_size, patch_size, out_channels*2, config.d_model,bias=True)  
        self.pos_embed_x = nn.Parameter(torch.zeros(1, self.x_embedder.num_patches, config.d_model), requires_grad=False)
        self.layers = nn.ModuleList([MambaBlock(config) for _ in range(layers)])

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        self.final_layer = FinalLayer(config.d_model, patch_size, out_channels)
        self.initialize_weights()
        
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed_x = get_2d_sincos_pos_embed(self.pos_embed_x.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed_x.data.copy_(torch.from_numpy(pos_embed_x).float().unsqueeze(0))


        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Zero-out output layers:

        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        
    def unpatchify(self, x):
        c = self.in_channels
        p = self.x_embedder.patch_size[0] #2
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h*p, h*p))
        return imgs
        
    def forward(self, x1, x2):
        x1 = self.conv(x1)  # [B, C, H, W] -> [B, C', 2H, 2W] 
        x1 = self.norm(x1)
        x1 = self.act(x1)
        x = torch.cat([x1,x2],dim=1)
        x = self.x_embedder(x) + self.pos_embed_x  # (N, T, D), where T = H * W / patch_size ** 2    torch.Size([10, 256, 16])# x : (B, L, D)
        for layer in self.layers:
            output = layer(x)
        x = self.final_layer(output)  # (N, T, patch_size ** 2 * out_channels)
        out = self.unpatchify(x)       # (N, out_channels, H, W)
        
        return out

class UMamba_512(nn.Module):
    def __init__(self, config: MambaConfig, patch_size=2):
        super().__init__()
        self.config = config
        self.input_size = config.input_size
        self.in_channels = config.in_channels
        self.x_embedder1 = PatchEmbed(32, patch_size, config.in_channels, config.d_model,bias=True)  
    
        self.x_embedder3 = PatchEmbed(32, patch_size, config.in_channels, config.d_model,bias=True)   
        self.layers1 = nn.ModuleList([MambaBlock(config) for _ in range(config.n_layers)])       
        self.layers3 = nn.ModuleList([MambaBlock(config) for _ in range(config.n_layers)])      
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.out_channels = config.out_channels
        self.pos_embed_x1 = nn.Parameter(torch.zeros(1, self.x_embedder1.num_patches, config.d_model), requires_grad=False)        
        self.pos_embed_x3 = nn.Parameter(torch.zeros(1, self.x_embedder3.num_patches, config.d_model), requires_grad=False)
        self.down1 = Downsample(config,config.in_channels,self.out_channels,128,2,3)
        self.down2 = Downsample(config,config.in_channels,self.out_channels,64,2,3)
        self.down3 = Downsample(config,config.in_channels,self.out_channels,32,2,3)
        self.up1 = Upsample(config,config.in_channels,self.out_channels,64,2,3)
        self.up2 = Upsample(config,config.in_channels,self.out_channels,128,2,3)
        self.up3 = Upsample(config,config.in_channels,self.out_channels,256,2,3)
        

        self.final_layer1 = FinalLayer(config.d_model, patch_size, config.in_channels)     
        self.final_layer3 = FinalLayer(config.d_model, patch_size, config.in_channels)
        self.enhance1 = HFNet(3, 64)
        self.initialize_weights()

    '''初始化权重'''
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed_x1 = get_2d_sincos_pos_embed(self.pos_embed_x1.shape[-1], int(self.x_embedder1.num_patches ** 0.5))
        self.pos_embed_x1.data.copy_(torch.from_numpy(pos_embed_x1).float().unsqueeze(0))
        pos_embed_x3 = get_2d_sincos_pos_embed(self.pos_embed_x3.shape[-1], int(self.x_embedder3.num_patches ** 0.5))
        self.pos_embed_x3.data.copy_(torch.from_numpy(pos_embed_x3).float().unsqueeze(0))


        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder1.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder1.proj.bias, 0)
        
        
        
        w = self.x_embedder3.proj.weight.data
        nn.init.constant_(self.x_embedder3.proj.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer1.linear.weight, 0)
        nn.init.constant_(self.final_layer1.linear.bias, 0)
        nn.init.constant_(self.final_layer3.linear.weight, 0)
        nn.init.constant_(self.final_layer3.linear.bias, 0)

        
    def unpatchify1(self, x):
        c = self.in_channels
        p = self.x_embedder1.patch_size[0] #2
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h*p, h*p))
        return imgs

    def unpatchify3(self, x):
        c = self.in_channels
        p = self.x_embedder3.patch_size[0] #2
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h*p, h*p))
        return imgs
    def forward(self, x,H1,pred_xstart=None,model_zero=None,**kwargs):
        LH, HL, HH = torch.unbind(H1, dim=2)
        h = torch.cat((LH, HL, HH), 0).cuda()
        hout = self.enhance1(h)    
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        
        x3 = self.x_embedder1(x3) + self.pos_embed_x1  
        for layer in self.layers1:
            x32 = layer(x3)
        x32 = self.final_layer1(x32)  
        x32 = self.unpatchify1(x32)                   

        x33 = self.x_embedder3(x32) + self.pos_embed_x3  
        for layer in self.layers3:
            x33 = layer(x33)
        x33 = self.final_layer3(x33)  
        x33 = self.unpatchify3(x33)                 

        x_3 = self.up1(x33,x2) 
        x_2 = self.up2(x_3,x1)
        out = self.up3(x_2,x)
        
        
        return hout,x32,x33,x_3,x_2,out
class UMamba(nn.Module):
    def __init__(self, config: MambaConfig, patch_size=2):
        super().__init__()
        self.config = config
        self.input_size = config.input_size
        self.in_channels = config.in_channels
        self.x_embedder1 = PatchEmbed(32, patch_size, config.in_channels, config.d_model,bias=True)  
        self.x_embedder3 = PatchEmbed(32, patch_size, config.in_channels, config.d_model,bias=True)  
        
        self.layers1 = nn.ModuleList([MambaBlock(config) for _ in range(config.n_layers)])

        self.layers3 = nn.ModuleList([MambaBlock(config) for _ in range(config.n_layers)])
        
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.out_channels = config.out_channels
        self.pos_embed_x1 = nn.Parameter(torch.zeros(1, self.x_embedder1.num_patches, config.d_model), requires_grad=False)
        
        self.pos_embed_x3 = nn.Parameter(torch.zeros(1, self.x_embedder3.num_patches, config.d_model), requires_grad=False)
        self.down1 = Downsample(config,config.in_channels,self.out_channels,128,2,2)
        self.down2 = Downsample(config,config.in_channels,self.out_channels,64,2,2)
        self.down3 = Downsample(config,config.in_channels,self.out_channels,32,1,2)
        self.up1 = Upsample(config,config.in_channels,self.out_channels,64,2,2)
        self.up2 = Upsample(config,config.in_channels,self.out_channels,128,2,2)
        self.up3 = Upsample(config,config.in_channels,self.out_channels,256,2,20)
        
        # Initialize (and freeze) pos_embed by sin-cos embedding:
        self.final_layer1 = FinalLayer(config.d_model, patch_size, config.in_channels)
        self.final_layer3 = FinalLayer(config.d_model, patch_size, config.in_channels)
        self.initialize_weights()

    '''初始化权重'''
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed_x1 = get_2d_sincos_pos_embed(self.pos_embed_x1.shape[-1], int(self.x_embedder1.num_patches ** 0.5))
        self.pos_embed_x1.data.copy_(torch.from_numpy(pos_embed_x1).float().unsqueeze(0))
        pos_embed_x3 = get_2d_sincos_pos_embed(self.pos_embed_x3.shape[-1], int(self.x_embedder3.num_patches ** 0.5))
        self.pos_embed_x3.data.copy_(torch.from_numpy(pos_embed_x3).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder1.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder1.proj.bias, 0)
        
  
        
        w = self.x_embedder3.proj.weight.data
        nn.init.constant_(self.x_embedder3.proj.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer1.linear.weight, 0)
        nn.init.constant_(self.final_layer1.linear.bias, 0)
        
        nn.init.constant_(self.final_layer3.linear.weight, 0)
        nn.init.constant_(self.final_layer3.linear.bias, 0)

    def unpatchify1(self, x):
        c = self.in_channels
        p = self.x_embedder1.patch_size[0] #2
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h*p, h*p))
        return imgs

    def unpatchify3(self, x):
        c = self.in_channels
        p = self.x_embedder3.patch_size[0] #2
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h*p, h*p))
        return imgs
    def forward(self, x ,pred_xstart=None,model_zero=None,**kwargs):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        
        x3 = self.x_embedder1(x3) + self.pos_embed_x1  # (N, T, D), where T = H * W / patch_size 
        for layer in self.layers1:
            x32 = layer(x3)
        x32 = self.final_layer1(x32)  # (N, T, patch_size ** 2 * out_channels)
        x32 = self.unpatchify1(x32)                   # (N, out_channels, H, W)

        x33 = self.x_embedder3(x32) + self.pos_embed_x3  
        for layer in self.layers3:
            x33 = layer(x33)
        x33 = self.final_layer3(x33)  # (N, T, patch_size ** 2 * out_channels)
        x33 = self.unpatchify3(x33)                   # (N, out_channels, H, W)

        x_3 = self.up1(x33,x2) 
        x_2 = self.up2(x_3,x1)
        out = self.up3(x_2,x)
        
        
        return x32,x33,x_3,x_2,out
"512need"
class cross_attention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.):
        super(cross_attention, self).__init__()
        if dim % num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (dim, num_heads)
            )
        self.num_heads = num_heads
        self.attention_head_size = int(dim / num_heads)

        self.query = Depth_conv(in_ch=dim, out_ch=dim)
        self.key = Depth_conv(in_ch=dim, out_ch=dim)
        self.value = Depth_conv(in_ch=dim, out_ch=dim)

        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        '''
        new_x_shape = x.size()[:-1] + (
            self.num_heads,
            self.attention_head_size,
        )
        print(new_x_shape)
        x = x.view(*new_x_shape)
        '''
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, ctx):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(ctx)
        mixed_value_layer = self.value(ctx)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        ctx_layer = torch.matmul(attention_probs, value_layer)
        ctx_layer = ctx_layer.permute(0, 2, 1, 3).contiguous()

        return ctx_layer

"512need"
class Depth_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Depth_conv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out
"512need"
class Dilated_Resblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Dilated_Resblock, self).__init__()

        sequence = list()
        sequence += [
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding=1, dilation=(1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding=2, dilation=(2, 2)),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, in_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding=3, dilation=(3, 3)),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, in_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding=2, dilation=(2, 2)),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, in_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding=1, dilation=(1, 1))

        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        out = self.model(x) + x

        return out
"512need"           
class HFNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HFNet, self).__init__()

        self.conv_head = Depth_conv(in_channels, out_channels)

        self.dilated_block_LH = Dilated_Resblock(out_channels, out_channels)
        self.dilated_block_HL = Dilated_Resblock(out_channels, out_channels)

        self.cross_attention0 = cross_attention(out_channels, num_heads=8)
        self.dilated_block_HH = Dilated_Resblock(out_channels, out_channels)
        self.conv_HH = nn.Conv2d(out_channels*2, out_channels, kernel_size=3, stride=1, padding=1)
        self.cross_attention1 = cross_attention(out_channels, num_heads=8)

        self.conv_tail = Depth_conv(out_channels, in_channels)

    def forward(self, x):

        residual = x

        x = self.conv_head(x)

        x_HL, x_LH, x_HH = torch.chunk(x, 3, dim=0)

        x_HH_LH = self.cross_attention0(x_LH, x_HH)
        x_HH_HL = self.cross_attention1(x_HL, x_HH)

        x_HL = self.dilated_block_HL(x_HL)
        x_LH = self.dilated_block_LH(x_LH)

        x_HH = self.dilated_block_HH(self.conv_HH(torch.cat((x_HH_LH, x_HH_HL), dim=1)))

        out = self.conv_tail(torch.cat((x_HL, x_LH, x_HH), dim=0))

        return out + residual


class MambaBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.Mamba = Mamba2(config.d_model)
        
    
    def forward(self, x):

        output = self.Mamba(x) + x 


        return output

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)


    def forward(self, x):
        x = self.linear(x)

        return x                 # (N, T, patch_size ** 2 * out_channels)










def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed
def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

