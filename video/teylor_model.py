import torch
from torch import nn
import sys

sys.path.append('../image')
sys.path.append('../image/modified')

from m_vqvae_multi_level import *
from pixelsnail import *


class Taylor_VQVAE(nn.Module):
    def __init__(
            self,
            n_frame=8,
            n_predic=8,
            shape=(256, 256),
            vqvae_in_channel=3,
            vqvae_channel=128,
            vqvae_n_res_block=2,
            vqvae_n_res_channel=32,
            vqvae_chanel_array=[64, 32, 16, 8, 8, 4],
            vqvae_n_embed=32,
            vqvae_n_level=4,
            vqvae_decay=0.99,
            pixel_channel=128,
            pixel_n_block=8,
            pixel_res_channel=256,
            pixel_cond_res_channel=256,
            pixel_kernel_size=3,
            pixel_cond_res_kernel=3,
            pixel_n_cond_res_block=8,
            pixel_n_res_block=4,
            pixel_attention=True,
            pixel_dropout=0.1,
            pixel_n_out_res_block=8,
    ):
        super().__init__()
        self.device = 'cuda'
        self.vqvae_chanel_array = vqvae_chanel_array
        vqvae_middle_dim = sum(vqvae_chanel_array)
        pixel_cond_channel = sum(vqvae_chanel_array[1:])
        self.n_predic = n_predic
        self.enc = Encoder(vqvae_in_channel * n_frame, vqvae_channel, vqvae_n_res_block, vqvae_n_res_channel, stride=4)
        self.enc_t = Encoder(vqvae_channel, vqvae_channel, vqvae_n_res_block, vqvae_n_res_channel, stride=2)
        self.quantize_conv = nn.Conv2d(vqvae_channel, vqvae_middle_dim, 1)
        self.n_level = vqvae_n_level
        self.quantizes = nn.ModuleList()
        self.quantizes_conv = nn.ModuleList()

        self.bns = nn.ModuleList()

        for i in range(vqvae_n_level):
            self.quantizes.append(Quantize(vqvae_middle_dim, vqvae_n_embed))
            self.quantizes_conv.append(nn.Conv2d(vqvae_middle_dim, vqvae_n_embed, 1))
            self.bns.append(nn.BatchNorm2d(vqvae_middle_dim))

        height, width = shape / 8

        self.n_class = vqvae_n_level * vqvae_n_embed

        if pixel_kernel_size % 2 == 0:
            kernel = pixel_kernel_size + 1

        else:
            kernel = pixel_kernel_size

        self.horizontal = CausalConv2d(
            self.n_class, pixel_channel, [kernel // 2, kernel], padding='down'
        )
        self.vertical = CausalConv2d(
            self.n_class, pixel_channel, [(kernel + 1) // 2, kernel // 2], padding='downright'
        )

        coord_x = (torch.arange(height).float() - height / 2) / height
        coord_x = coord_x.view(1, 1, height, 1).expand(1, 1, height, width)
        coord_y = (torch.arange(width).float() - width / 2) / width
        coord_y = coord_y.view(1, 1, 1, width).expand(1, 1, height, width)
        self.register_buffer('background', torch.cat([coord_x, coord_y], 1))

        self.blocks = nn.ModuleList()

        for i in range(pixel_n_block):
            self.blocks.append(
                PixelBlock(
                    pixel_channel,
                    pixel_res_channel,
                    pixel_kernel_size,
                    pixel_n_res_block,
                    attention=pixel_attention,
                    dropout=pixel_dropout,
                    condition_dim=pixel_cond_res_channel,
                )
            )

        if pixel_n_cond_res_block > 0:
            self.cond_resnet = CondResNet(
                pixel_cond_channel, pixel_cond_res_channel, pixel_cond_res_kernel, pixel_n_cond_res_block
            )

        out = []

        for i in range(pixel_n_out_res_block):
            out.append(GatedResBlock(pixel_channel, pixel_res_channel, 1))

        out.extend([nn.ELU(inplace=True), WNConv2d(pixel_channel, self.n_class, 1)])

        self.out = nn.Sequential(*out)

    def forward(self, input, background):

        quant_t, cond_mat, diff, _, _ = self.encode(input)

        out = quant_t
        outs = []
        for i in range(self.n_predic):
            end = sum(self.vqvae_chanel_array[1:min(i+2,self.n_predic)])
            cshape = cond_mat.shape
            condition = torch.zeros(cshape[0],sum(self.vqvae_chanel_array[1:]),cshape[2],cshape[3])
            condition[:,:end,:,:] = cond_mat[:,:end,:,:]
            for resblock in self.pixel_resblocks:
                out = resblock(out, condition=condition)

            if self.pixel_attention:
                key_cat = torch.cat([input, out, background], 1)
                key = self.pixel_key_resblock(key_cat)
                query_cat = torch.cat([out, background], 1)
                query = self.pixel_query_resblock(query_cat)
                attn_out = self.pixel_causal_attention(query, key)
                out = self.pixel_out_resblock(out, attn_out)

            else:
                bg_cat = torch.cat([out, background], 1)
                out = self.out(bg_cat)

            outs.append(out)

        return outs

    def encode(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)
        enc = enc_t[:,:self.vqvae_chanel_array[0],:,:]
        conds = enc_t[:, self.vqvae_chanel_array[0]:, :, :]
        bottleneck = self.quantize_conv(enc)
        ids = None
        quants = None
        diffs = None
        quant_sum = None
        for i,quantize in enumerate(self.quantizes):
            # print('bottleneck shape'.format(bottleneck.shape))
            quant, diff, id = quantize(bottleneck.permute(0, 2, 3, 1))
            # print(bottleneck.shape)
            # print(quant.shape)
            quant = quant.permute(0, 3, 1, 2)
            diff = diff.unsqueeze(0)

            if diffs is None:
                diffs = diff
                quant_sum = quant
                quants = quant.unsqueeze(1)
                ids = id.unsqueeze(1)
            else:
                diffs += diff
                quant_sum += quant
                quants = torch.cat((quants,quant.unsqueeze(1)),dim=1)
                ids = torch.cat((ids, id.unsqueeze(1)), dim=1)
            bottleneck -= quant
            # bottleneck = F.relu(self.bns[i](self.quantizes_conv[i](bottleneck)))
        return quant_sum,conds, diffs, quants, ids

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)

        return dec

    def decode_code(self, codes):
        quants = None
        for i, code in enumerate(codes):
            quant = self.quantizes.embed_code(code)
            quant = quant.permute(0, 3, 1, 2)
            quants += quant
        dec = self.decode(quants)
        return dec