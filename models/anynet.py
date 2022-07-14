import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from .submodule import *


class AnyNet(nn.Module):
    def __init__(self):
        super(AnyNet, self).__init__()
        self.feature_extraction = feature_extraction()
        self.conv3d_1 = make_conv3d_block(in_channels=1, hidden_channels=16, out_channels=1, num_layers=6)
        self.conv3d_2 = make_conv3d_block(in_channels=1, hidden_channels=4, out_channels=1, num_layers=6)
        self.conv3d_3 = make_conv3d_block(in_channels=1, hidden_channels=4, out_channels=1, num_layers=6)
        self.volume_regularization = nn.ModuleList([self.conv3d_1, self.conv3d_2, self.conv3d_3])

    def _build_volume(self, refimg, targetimg, maxdisp):
        B, C, H, W = refimg.shape
        cost = torch.zeros(B, 1, maxdisp, H, W, device='cuda')
        for i in range(maxdisp):
            cost[:, :, i, :, :i] = torch.norm(refimg[:, :, :, :i], p=1, dim=1, keepdim=True)
            if i == 0:
                cost[:, :, i, :, :] = torch.norm(refimg[:, :, :, :] - targetimg[:, :, :, :], p=1, dim=1, keepdim=True)
            else:
                cost[:, :, i, :, i:] = torch.norm(refimg[:, :, :, i:] - targetimg[:, :, :, :-i], p=1, dim=1, keepdim=True)
        return cost.contiguous()

    def _bulid_residual_volume(self, refimg, targetimg, maxdisp, disp):
        B, C, H, W = refimg.shape
        cost = torch.zeros(B, 1, 2*maxdisp+1, H, W, device='cuda')
        for i in range(-maxdisp, maxdisp+1):
            new_disp = disp + i
            reconimg = self._warp(targetimg, new_disp)
            cost[:, :, i+maxdisp, :, :] = torch.norm(refimg[:, :, :, :] - reconimg[:, :, :, :], p=1, dim=1, keepdim=True)
        return cost.contiguous()

    def _warp(self, x, disp):
        '''
        Warp an image tensor right image to left image, according to disparity
        x: [B, C, H, W] right image
        disp: [B, 1, H, W] horizontal shift
        '''
        B, C, H, W = x.shape
        # mesh grid
        '''
        for example: H=4, W=3
        xx =         yy =
        [[0 1 2],    [[0 0 0],    
         [0 1 2],     [1 1 1],
         [0 1 2],     [2 2 2],
         [0 1 2]]     [3 3 3]]
        '''
        xx = torch.arange(0, W, device='cuda').view(1,-1).repeat(H, 1)  # [H, W]
        yy = torch.arange(0, H, device='cuda').view(-1,1).repeat(1, W)  # [H, W]
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)  # [B, 1, H, W]
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)  # [B, 1, H, W]
        vgrid = torch.cat((xx, yy), dim=1).float()   # [B, 2, H, W]

        # the correspondence between left and right is that left (i, j) = right (i-d, j)
        vgrid[:, :1, :, :] = vgrid[:, :1, :, :] - disp
        # scale to [-1, 1]
        vgrid[:, 0, :, :] = vgrid[:, 0, :, :] * 2.0 / (W-1) - 1.0
        vgrid[:, 1, :, :] = vgrid[:, 1, :, :] * 2.0 / (H-1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)
        output = F.grid_sample(x, vgrid, align_corners=True)
        return output

    def forward(self, left, right):
        refimg = self.feature_extraction(left)  # [1/16, 1/8, 1/4]
        targetimg = self.feature_extraction(right)  # [1/16, 1/8, 1/4]
        scales = len(refimg)

        pred = []
        for i in range(scales):
            if i == 0:
                cost = self._build_volume(refimg[i], targetimg[i], maxdisp=12)
            else:
                down_scale = float(refimg[i].shape[2] / left.shape[2])
                # print(f'Down scale: {down_scale}')
                warp_disp = F.interpolate(pred[i-1], scale_factor=down_scale, mode='bilinear', align_corners=True) * down_scale
                cost = self._bulid_residual_volume(refimg[i], targetimg[i], maxdisp=2, disp=warp_disp)
            cost = self.volume_regularization[i](cost)
            cost = cost.squeeze(1)
            up_scale = float(left.shape[2] / refimg[i].shape[2])
            # print(f'Up scale: {up_scale}')
            if i == 0:
                # predict disparity
                pred_low_res = F.softmax(-cost, dim=1)
                pred_low_res = disparityregression(start=0, maxdisp=12)(pred_low_res)
                pred_low_res = pred_low_res * up_scale  # D/16 -> D
                pred_high_res = F.interpolate(pred_low_res, scale_factor=up_scale, mode='bilinear', align_corners=True)  # H/16 x W/16 -> H x W
                pred.append(pred_high_res)
            else:
                # predict residual
                pred_low_res = F.softmax(-cost, dim=1)
                pred_low_res = disparityregression(start=-2, maxdisp=2+1)(pred_low_res)
                pred_low_res = pred_low_res * up_scale  # D/8 or D/4 -> D
                pred_high_res = F.interpolate(pred_low_res, scale_factor=up_scale, mode='bilinear', align_corners=True)  # H/8 x W/8 or H/4 x W/4 -> H x W  
                pred.append(pred_high_res + pred[i-1])

        return pred[0], pred[1], pred[2]  # [stage1, stage2, stage3]
            

if __name__ == '__main__':
    device = 'cuda'
    summary(AnyNet().to(device), [(3, 256, 256), (3, 256, 256)])
