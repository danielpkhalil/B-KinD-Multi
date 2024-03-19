import torch
import torch.nn as nn

from torchvision import transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as nnf

from .resnet_updated import conv3x3
from .resnet_updated import resnetbank50all as resnetbank50
from .globalNet import globalNet

import math
# from .tusk import tusk
# from kornia.morphology import dilation


import torch
from PIL import Image
import numpy as np

import os
import cv2
from PIL import Image
import numpy as np
import torch
import imageio
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
import gc

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        # if norm_layer is None:
        #     norm_layer = nn.BatchNorm2d
        norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.upsample = upsample
        self.stride = stride
        self.norm_layer = norm_layer

    def forward(self, x):
        
        if self.upsample is not None:
            x = self.upsample(x)

        out = self.conv1(x)
        if self.norm_layer is not None:
            N, C, H, W = out.shape
            layer_norm = nn.LayerNorm([C, H, W]).cuda()
            out = layer_norm(out)
        else:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        
        if self.norm_layer is not None:
            N, C, H, W = out.shape
            layer_norm = nn.LayerNorm([C, H, W]).cuda()
            out = layer_norm(out)
        else:
            out = self.bn2(out)
        out = self.relu(out)

        return out
    

class Decoder(nn.Module):
    def __init__(self, in_planes=256, wh=14, n_kps=10, ratio=1.0, norm_layer=None):
        super(Decoder, self).__init__()
        
        self.K = n_kps
        
        w, h = wh, wh
        if ratio != 1.0:
            w = wh
            h = ratio * wh
            
        self.layer1 = BasicBlock(int(in_planes)+self.K, int(in_planes/2),
                                 upsample=nn.Upsample((int(h*2), int(w*2)), mode='bilinear'),
                                 norm_layer=norm_layer); in_planes /= 2
        self.layer2 = BasicBlock(int(in_planes)+self.K, int(in_planes/2), 
                                 upsample=nn.Upsample((int(h*4), int(w*4)), mode='bilinear'),
                                 norm_layer=norm_layer); in_planes /= 2
        self.layer3 = BasicBlock(int(in_planes)+self.K, int(in_planes/2), 
                                 upsample=nn.Upsample((int(h*8), int(w*8)), mode='bilinear'),
                                 norm_layer=norm_layer); in_planes /= 2
        self.layer4 = BasicBlock(int(in_planes)+self.K, int(in_planes/2),
                                 upsample=nn.Upsample((int(h*16), int(w*16)), mode='bilinear'),
                                 norm_layer=norm_layer); in_planes /= 2
        self.layer5 = BasicBlock(int(in_planes)+self.K, max(int(in_planes/2), 32),
                                 upsample=nn.Upsample((int(h*32), int(w*32)), mode='bilinear'),
                                 norm_layer=norm_layer)
        in_planes = max(int(in_planes/2), 32)
        
        self.conv_final = nn.Conv2d(int(in_planes), 3, kernel_size=1, stride=1)
        
    def forward(self, x, heatmap):
        
        x = torch.cat((x[0], heatmap[0]), dim=1)
        x = self.layer1(x)
        x = torch.cat((x, heatmap[1]), dim=1)
        x = self.layer2(x)
        x = torch.cat((x, heatmap[2]), dim=1)
        x = self.layer3(x)
        x = torch.cat((x, heatmap[3]), dim=1)
        x = self.layer4(x)
        
        x = torch.cat((x, heatmap[4]), dim=1)
        x = self.layer5(x)
        x = self.conv_final(x)
        
        return x
    
    
class Model(nn.Module):
    def __init__(self, n_kps=10, output_dim=200, pretrained=True, 
                 output_shape=(64, 64), num_agents=2, frame_gap=20):

        super(Model, self).__init__()
        self.K = n_kps
        self.num_agents = num_agents
        self.frame_gap = frame_gap
        
        channel_settings = [2048, 1024, 512, 256]
        self.output_shape = output_shape
        self.kptNet = globalNet(channel_settings, output_shape, n_kps)
        self.ch_softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # width == height for now
        self.decoder = Decoder(in_planes=2048, wh=int(output_shape[0]/8), 
                               n_kps=self.K*2*num_agents, ratio=1.0) #, norm_layer=nn.LayerNorm) self.K*2 --> *4

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        self.encoder = resnetbank50(pretrained=pretrained)

        #SAM ViT
        # CKPT_PATH = 'SAMT/ckpt/sam_vit_b_01ec64.pth'
        # checkpoint = torch.load(CKPT_PATH)
        # encoder_weights = {k: v for k, v in checkpoint.items() if k.startswith('image_encoder.') and not k.endswith(('rel_pos_h', 'rel_pos_w'))}
        # SAM_ViT = image_encoder.ImageEncoderViT2()
        # SAM_ViT.load_state_dict(encoder_weights, strict=False)
        # for param in SAM_ViT.parameters():
        #     param.requires_grad = False
        # SAM_ViT.to("cuda")
        # self.encoder = SAM_ViT
        #check input and output (embedd) dimmensions, also freeze for not backprop

    def get_keypoints(self, x):
        x_res = self.encoder(x)

        # Get keypoints of x
        kpt_feat, kpt_out = self.kptNet(x_res)  # keypoint for reconstruction
        
        # Reconstruction module
        heatmap = kpt_out[-1].view(-1, self.K, kpt_out[-1].size(2) * kpt_out[-1].size(3))
        heatmap = self.ch_softmax(heatmap)
        heatmap = heatmap.view(-1, self.K, kpt_out[-1].size(2), kpt_out[-1].size(3))
                
        u_x, u_y, covs = self._mapTokpt(heatmap)        

        return (u_x, u_y)        

    def get_all_keypoints(self, x_list):

        u_x_list = []
        u_y_list = []
        for item in x_list:
            u_x, u_y = self.get_keypoints(item)
            u_x_list.append(u_x)
            u_y_list.append(u_y)
            

        return (u_x_list, u_y_list)

    def forward(self, x, tr_x=None, gmtr_x1 = None, gmtr_x2 = None, gmtr_x3 = None,
                find_peaks=False, use_bbox=True, frame_idx=0, masks=[]):
        #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                std=[0.229, 0.224, 0.225])
        # use SAM normalization


        x_masks = self.getMask(masks[frame_idx])

        # h, w = x_norm.shape[-2:]
        # padh = self.encoder.img_size - h
        # padw = self.encoder.img_size - w
        # x_norm = nnf.pad(x_norm, (0, padw, 0, padh))

        x_res = self.encoder(x)
        
        if tr_x is not None:
            tr_x_masks = self.getMask(masks[frame_idx + self.frame_gap])

            # h, w = tr_x_norm.shape[-2:]
            # padh = self.encoder.img_size - h
            # padw = self.encoder.img_size - w
            # tr_x_norm = nnf.pad(tr_x_norm, (0, padw, 0, padh))

            tr_x_res = self.encoder(tr_x)
            tr_kpt_feat, tr_kpt_out = self.kptNet(tr_x_res)  # keypoint for reconstruction

            # Reconstruction module
            tr_heatmap = tr_kpt_out[-1].view(-1, self.K, tr_kpt_out[-1].size(2) * tr_kpt_out[-1].size(3))
            tr_heatmap = self.ch_softmax(tr_heatmap)
            tr_heatmap = tr_heatmap.view(-1, self.K, tr_kpt_out[-1].size(2), tr_kpt_out[-1].size(3))

            tr_confidence = tr_heatmap.max(dim=-1)[0].max(dim=-1)[0]
            
            # tusk(tr_heatmap)
            tr_u_x, tr_u_y, tr_covs, bbox, tr_confidence = self._mapTokpt(tr_heatmap, tr_x_masks, find_peaks=find_peaks, use_bbox=use_bbox)

        # Get keypoints of x
        kpt_feat, kpt_out = self.kptNet(x_res)  # keypoint for reconstruction
        
        # Reconstruction module
        heatmap = kpt_out[-1].view(-1, self.K, kpt_out[-1].size(2) * kpt_out[-1].size(3))
        heatmap  = self.ch_softmax(heatmap)
        heatmap = heatmap.view(-1, self.K, kpt_out[-1].size(2), kpt_out[-1].size(3))
                
        u_x, u_y, covs, _, confidence = self._mapTokpt(heatmap, x_masks, find_peaks=find_peaks, use_bbox=use_bbox)
        
        if tr_x is None:
            return (u_x, u_y), kpt_out[-1], confidence, x_masks

        tr_kpt_conds = []
        
        prev_w, prev_h = int(self.output_shape[0]/16), int(self.output_shape[1]/16)
        std_in = [0.1, 0.1, 0.01, 0.01, 0.001]
        
        for i in range(0, 5):
            prev_h *= 2;  prev_w *= 2
            
            # _We can concatenate both keypoint representation
            hmaps = self._kptTomap(tr_u_x, tr_u_y, H=prev_h, W=prev_w, inv_std=std_in[i], normalize=False)

            hmaps_2 = self._kptTomap(u_x, u_y, H=prev_h, W=prev_w, inv_std=std_in[i], normalize=False)

            hmaps = torch.cat([hmaps, hmaps_2], dim = 1)

            tr_kpt_conds.append(hmaps)
            
        recon = self.decoder(x_res, tr_kpt_conds)
        
        if gmtr_x1 is not None:  # Rotation loss
            out_h, out_w = int(self.output_shape[0]*2), int(self.output_shape[1]*2)

            gmtr_x1_masks = self.getMask(np.rot90(masks[frame_idx + self.frame_gap]))

            # h, w = gmtr_x1_norm.shape[-2:]
            # padh = self.encoder.img_size - h
            # padw = self.encoder.img_size - w
            # gmtr_x1_norm = nnf.pad(gmtr_x1_norm, (0, padw, 0, padh))

            gmtr_x_res = self.encoder(gmtr_x1)
            gmtr_kpt_feat, gmtr_kpt_out = self.kptNet(gmtr_x_res)
            
            gmtr_heatmap = gmtr_kpt_out[-1].view(-1, self.K, gmtr_kpt_out[-1].size(2) * gmtr_kpt_out[-1].size(3))
            gmtr_heatmap = self.ch_softmax(gmtr_heatmap)
            gmtr_heatmap = gmtr_heatmap.view(-1, self.K, gmtr_kpt_out[-1].size(2), gmtr_kpt_out[-1].size(3))
            
            gmtr_u_x, gmtr_u_y, gmtr_covs, _, _ = self._mapTokpt(gmtr_heatmap, gmtr_x1_masks, find_peaks=find_peaks, use_bbox=use_bbox)

            gmtr_kpt_conds_1 = self._kptTomap(gmtr_u_x, gmtr_u_y, H=out_h, W=out_w, inv_std=0.001, normalize=False)

            #################################################
            gmtr_x2_masks = self.getMask(np.rot90(masks[frame_idx + self.frame_gap], 2))

            # h, w = gmtr_x2_norm.shape[-2:]
            # padh = self.encoder.img_size - h
            # padw = self.encoder.img_size - w
            # gmtr_x2_norm = nnf.pad(gmtr_x2_norm, (0, padw, 0, padh))

            gmtr_x_res = self.encoder(gmtr_x2)
            gmtr_kpt_feat, gmtr_kpt_out = self.kptNet(gmtr_x_res)
            
            gmtr_heatmap = gmtr_kpt_out[-1].view(-1, self.K, gmtr_kpt_out[-1].size(2) * gmtr_kpt_out[-1].size(3))
            gmtr_heatmap = self.ch_softmax(gmtr_heatmap)
            gmtr_heatmap = gmtr_heatmap.view(-1, self.K, gmtr_kpt_out[-1].size(2), gmtr_kpt_out[-1].size(3))
            
            gmtr_u_x_2, gmtr_u_y_2, gmtr_covs, _, _ = self._mapTokpt(gmtr_heatmap, gmtr_x2_masks, find_peaks=find_peaks, use_bbox=use_bbox)

            gmtr_kpt_conds_2 = self._kptTomap(gmtr_u_x_2, gmtr_u_y_2, H=out_h, W=out_w, inv_std=0.001, normalize=False)

            ###########################################
            gmtr_x3_masks = self.getMask(np.rot90(masks[frame_idx + self.frame_gap], -1))

            # h, w = gmtr_x3_norm.shape[-2:]
            # padh = self.encoder.img_size - h
            # padw = self.encoder.img_size - w
            # gmtr_x3_norm = nnf.pad(gmtr_x3_norm, (0, padw, 0, padh))

            gmtr_x_res = self.encoder(gmtr_x3)
            gmtr_kpt_feat, gmtr_kpt_out = self.kptNet(gmtr_x_res)
            
            gmtr_heatmap = gmtr_kpt_out[-1].view(-1, self.K, gmtr_kpt_out[-1].size(2) * gmtr_kpt_out[-1].size(3))
            gmtr_heatmap = self.ch_softmax(gmtr_heatmap)
            gmtr_heatmap = gmtr_heatmap.view(-1, self.K, gmtr_kpt_out[-1].size(2), gmtr_kpt_out[-1].size(3))
            
            gmtr_u_x_3, gmtr_u_y_3, gmtr_covs, _, _ = self._mapTokpt(gmtr_heatmap, gmtr_x3_masks, find_peaks=find_peaks, use_bbox=use_bbox)

            gmtr_kpt_conds_3 = self._kptTomap(gmtr_u_x_3, gmtr_u_y_3, H=out_h, W=out_w, inv_std=0.001, normalize=False)

            return (recon, (tr_u_x, tr_u_y), (tr_kpt_conds[-1], gmtr_kpt_conds_1, gmtr_kpt_conds_2, gmtr_kpt_conds_3),
                    (tr_kpt_out[-1], gmtr_kpt_out[-1], bbox), (u_x, u_y), 
                    (gmtr_u_x, gmtr_u_y, gmtr_u_x_2, gmtr_u_y_2, gmtr_u_x_3, gmtr_u_y_3),
                    tr_confidence, bbox)
        
        
        return recon, (tr_u_x, tr_u_y), tr_kpt_conds[-1], tr_kpt_out[-1], (u_x, u_y), tr_confidence
    
        
#     def _mapTokpt(self, heatmap):
#         # heatmap: (N, K, H, W)    
            
#         H = heatmap.size(2)
#         W = heatmap.size(3)
        
#         s_y = heatmap.sum(3)  # (N, K, H)
#         s_x = heatmap.sum(2)  # (N, K, W)
        
#         y = torch.linspace(-1.0, 1.0, H).cuda()
#         x = torch.linspace(-1.0, 1.0, W).cuda()
        
#         u_y = (y * s_y).sum(2) / s_y.sum(2)  # (N, K)
#         u_x = (x * s_x).sum(2) / s_x.sum(2)
        
#         y = torch.reshape(y, (1, 1, H, 1))
#         x = torch.reshape(x, (1, 1, 1, W))
        
#         # Covariance
#         var_y = ((heatmap * y.pow(2)).sum(2).sum(2) - u_y.pow(2)).clamp(min=1e-6)
#         var_x = ((heatmap * x.pow(2)).sum(2).sum(2) - u_x.pow(2)).clamp(min=1e-6)
        
#         cov = ((heatmap * (x - u_x.view(-1, self.K, 1, 1)) * (y - u_y.view(-1, self.K, 1, 1))).sum(2).sum(2)) #.clamp(min=1e-6)
                
#         return u_x, u_y, (var_x, var_y, cov)
    
    
    def get_fly_bboxes(self, heatmap, find_peaks):
        # Given a rough agent size / identity-free
        ##################### Get two bboxes     
        # 9 x 5 x 64 x 64

        summed = torch.sum(heatmap, dim = 1, keepdim = True)
        summed = _nms(summed, kernel = 15) # changed 15 to 9

        scores, inds, ys, xs = _topk(summed, K=1)
        scores = scores.repeat((1,self.K, 1))
        ys = ys.repeat((1,self.K,1))
        xs = xs.repeat((1,self.K,1))
        
        #ys = 9x5x2
        # bboxes = torch.stack([torch.zeros(heatmap.size()), torch.zeros(heatmap.size())])
        bboxes = torch.stack([torch.zeros(heatmap.size()) for _ in range(self.num_agents)])

        ##############################
        batch_size = bboxes.size()[1]

        bbox_y = ys.view((-1, 1)).long()
        bbox_x = xs.view((-1, 1)).long()
        bbox_scores = scores.view(-1, 1)

        comparison_score = scores[:, :, 0].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.output_shape[0], self.output_shape[1]) #64, 64)

        def create_circular_mask(center_y, center_x, radius):

            # Y, X = torch.meshgrid(torch.arange(64), torch.arange(64))
            Y, X = torch.meshgrid(torch.arange(self.output_shape[0]), torch.arange(self.output_shape[0]))
            X = X.unsqueeze(0).repeat(batch_size*self.K, 1, 1).to(heatmap.device)
            Y = Y.unsqueeze(0).repeat(batch_size*self.K, 1, 1).to(heatmap.device)


            # to_sqrt = (X - center_x.unsqueeze(-1).unsqueeze(-1).repeat(1, 64, 64))**2 + \
            #          (Y-center_y.unsqueeze(-1).unsqueeze(-1).repeat(1, 64, 64))**2
            to_sqrt = (X - center_x.unsqueeze(-1).unsqueeze(-1).repeat(1, self.output_shape[0], self.output_shape[0]))**2 + \
                     (Y-center_y.unsqueeze(-1).unsqueeze(-1).repeat(1, self.output_shape[0], self.output_shape[0]))**2
            dist_from_center = torch.sqrt(to_sqrt.float())

            mask = dist_from_center <= radius
            return mask
        """If this mask is perfect, all the problems are solved?
        1. Generate rough mask
        2. Apply gaussian filter? or make the mask thicker by convolving dilation filter?
        3. Use this mask for keypoints"""
        rad = 8 # 10
        dist_mask = create_circular_mask(bbox_y[:, 0], bbox_x[:, 0], rad).view((batch_size, self.K, self.output_shape[0], self.output_shape[0])) # 64, 64))

        # dist_mask_small = create_circular_mask(bbox_y[:, 0], bbox_x[:, 0], rad).view((batch_size, 5, self.output_shape[0], self.output_shape[0])) #64, 64))


        tmp = dist_mask #torch.mul(keep, dist_mask)

        bboxes[0] = tmp
        without_top = summed
        
        ##############################
        for ii in range(1, self.num_agents):
            without_top = torch.mul(without_top, 1.0 - dist_mask[:, 0, :, :].detach().long().unsqueeze(1))
            #print(without_top.size(), summed.size(), dist_mask[:, 0, :, :].size())
            scores, inds, ys, xs = _topk(without_top, K=1)
            scores = scores.repeat((1,5, 1))
            ys = ys.repeat((1,self.K,1))
            xs = xs.repeat((1,self.K,1))        


            bbox_y = ys.view((-1, 1)).long()
            bbox_x = xs.view((-1, 1)).long()
            bbox_scores = scores.view(-1, 1)      

            comparison_score2 = scores[:, :, 0].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.output_shape[0], self.output_shape[0]) #64, 64)

            # rad = 10
            dist_mask = create_circular_mask(bbox_y[:, 0], bbox_x[:, 0], rad).view((batch_size, self.K, self.output_shape[0], self.output_shape[0])) #64, 64))

            tmp = dist_mask#torch.mul(keep2, dist_mask)

            bboxes[ii] = tmp
        
        return bboxes.to(heatmap.device)
    
#     # Mask from summed heatmap
#     def get_fly_bboxes(self, heatmap, find_peaks):
#         # Given a rough agent size / identity-free
#         ##################### Get two bboxes     
#         # 9 x 5 x 64 x 64

#         summed = torch.sum(heatmap, dim = 1, keepdim = True)
#         summed = _nms(summed, kernel = 15)
#         import pdb; pdb.set_trace()
        
#         # Threshold summed heatmap (to generate binary mask)
#         thr_summed = (summed > 1.0)
        
#         # Apply dilation filter?
        
        
#         # Detect box? segmentation mask? for each instance
        
        
#         # From each box, multiply thr_summed --> get mask for each instance
        
        
#         scores, inds, ys, xs = _topk(summed, K=1)
#         scores = scores.repeat((1,5, 1))
#         ys = ys.repeat((1,5,1))
#         xs = xs.repeat((1,5,1))

#         bbox_size = 2
        
#         bboxes = torch.stack([torch.zeros(heatmap.size()), torch.zeros(heatmap.size())])

#         ##############################
#         batch_size = bboxes.size()[1]

#         bbox_y = ys.view((-1, 1)).long()
#         bbox_x = xs.view((-1, 1)).long()
#         bbox_scores = scores.view(-1, 1)

#         comparison_score = scores[:, :, 0].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 64, 64)

#         def create_circular_mask(center_y, center_x, radius):

#             Y, X = torch.meshgrid(torch.arange(64), torch.arange(64))
#             X = X.unsqueeze(0).repeat(batch_size*5, 1, 1).to(heatmap.device)
#             Y = Y.unsqueeze(0).repeat(batch_size*5, 1, 1).to(heatmap.device)


#             to_sqrt = (X - center_x.unsqueeze(-1).unsqueeze(-1).repeat(1, 64, 64))**2 + \
#                      (Y-center_y.unsqueeze(-1).unsqueeze(-1).repeat(1, 64, 64))**2
#             dist_from_center = torch.sqrt(to_sqrt.float())

#             mask = dist_from_center <= radius
#             return mask
#         """If this mask is perfect, all the problems are solved?
#         1. Generate rough mask
#         2. Apply gaussian filter? or make the mask thicker by convolving dilation filter?
#         3. Use this mask for keypoints"""
#         rad = 10
#         dist_mask = create_circular_mask(bbox_y[:, 0], bbox_x[:, 0], rad).view((batch_size, 5, 64, 64))

#         dist_mask_small = create_circular_mask(bbox_y[:, 0], bbox_x[:, 0], 10).view((batch_size, 5, 64, 64))


#         tmp = dist_mask #torch.mul(keep, dist_mask)

#         bboxes[0] = tmp

#         ##############################

#         without_top = torch.mul(summed, 1.0 - dist_mask_small[:, 0, :, :].detach().long().unsqueeze(1))
#         #print(without_top.size(), summed.size(), dist_mask[:, 0, :, :].size())
#         scores, inds, ys, xs = _topk(without_top, K=1)
#         scores = scores.repeat((1,5, 1))
#         ys = ys.repeat((1,5,1))
#         xs = xs.repeat((1,5,1))        

        
#         bbox_y = ys.view((-1, 1)).long()
#         bbox_x = xs.view((-1, 1)).long()
#         bbox_scores = scores.view(-1, 1)      

#         comparison_score2 = scores[:, :, 0].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 64, 64)

#         rad = 10
#         dist_mask = create_circular_mask(bbox_y[:, 0], bbox_x[:, 0], rad).view((batch_size, 5, 64, 64))
    
#         tmp = dist_mask#torch.mul(keep2, dist_mask)
        
#         bboxes[1] = tmp

#         return bboxes.to(heatmap.device)
    
    
    def get_wing_bbox(self, heatmap):
        ##################### Get two bboxes     
        # 9 x 5 x 64 x 64
        heatmap = _nms(heatmap, kernel = 11)

        scores, inds, ys, xs = _topk(heatmap, K=2)

        
        bbox_size = 2 # changed 2 to 1

        #ys = 9x5x2
        bboxes = torch.stack([torch.zeros(heatmap.size()), torch.zeros(heatmap.size())])

        #print(bboxes.size(), heatmap.size())
        # bboes = 2x 9x5x64x64 

        ##############################
        batch_size = heatmap.size()[0]
        tmp = bboxes[0].view((-1, 64, 64))

        bbox_y = ys.view((-1, 2)).long()
        bbox_x = xs.view((-1, 2)).long()
        # print(bbox_y, bbox_x)
        y_min = torch.clamp(bbox_y[:, 0] - 1, min = 0, max = 63)
        y_max = torch.clamp(bbox_y[:, 0] + 1, min = 0, max = 63)
        x_min = torch.clamp(bbox_x[:, 0] - 1, min = 0, max = 63)
        x_max = torch.clamp(bbox_x[:, 0] + 1, min = 0, max = 63)        
        tmp[torch.arange(tmp.size(0)), bbox_y[:, 0], bbox_x[:, 0]] = 1
        tmp[torch.arange(tmp.size(0)), y_min, bbox_x[:, 0]] = 1
        tmp[torch.arange(tmp.size(0)), y_max, bbox_x[:, 0]] = 1
        tmp[torch.arange(tmp.size(0)), bbox_y[:, 0], x_min] = 1
        tmp[torch.arange(tmp.size(0)), bbox_y[:, 0], x_max] = 1
        tmp[torch.arange(tmp.size(0)), y_min, x_min] = 1
        tmp[torch.arange(tmp.size(0)), y_max, x_min] = 1
        tmp[torch.arange(tmp.size(0)), y_min, x_max] = 1
        tmp[torch.arange(tmp.size(0)), y_max, x_max] = 1

        #tmp[torch.arange(tmp.size(0)), bbox_y[:, 0], bbox_x[:, 0]] = 1

        bboxes[0] = tmp.view((batch_size, 5, 64, 64))

        ##############################
        tmp = bboxes[1].view((-1, 64, 64))

        bbox_scores = scores.view(-1, 2)
        # print(bbox_scores[:9])
        # print(bbox_scores[:9, 1]/bbox_scores[:9, 0])

        thresh = 0.001

        y_min = torch.clamp(bbox_y[:, 1] - 1, min = 0, max = 63)
        y_max = torch.clamp(bbox_y[:, 1] + 1, min = 0, max = 63)
        x_min = torch.clamp(bbox_x[:, 1] - 1, min = 0, max = 63)
        x_max = torch.clamp(bbox_x[:, 1] + 1, min = 0, max = 63)        
        tmp[torch.arange(tmp.size(0)), bbox_y[:, 1], bbox_x[:, 1]] = 1
        tmp[torch.arange(tmp.size(0)), y_min, bbox_x[:, 1]] = 1
        tmp[torch.arange(tmp.size(0)), y_max, bbox_x[:, 1]] = 1
        tmp[torch.arange(tmp.size(0)), bbox_y[:, 1], x_min] = 1
        tmp[torch.arange(tmp.size(0)), bbox_y[:, 1], x_max] = 1
        tmp[torch.arange(tmp.size(0)), y_min, x_min] = 1
        tmp[torch.arange(tmp.size(0)), y_max, x_min] = 1
        tmp[torch.arange(tmp.size(0)), y_min, x_max] = 1
        tmp[torch.arange(tmp.size(0)), y_max, x_max] = 1


        bboxes[1] = tmp.view((batch_size, 5, 64, 64))


        return bboxes.to(heatmap.device)
    
    
    def _mapTokpt(self, heatmap, masks, find_peaks, use_bbox):
        # heatmap: (N, K, H, W) 
        # if use_bbox:
        #     # Summed bbox for each fly
        #     bbox_masks = self.get_fly_bboxes(heatmap, find_peaks = find_peaks).detach()
        # else:
        #     # max locations
        #     bbox_masks = self.get_wing_bbox(heatmap).detach()
        if len(masks) > 0:
            heatmap_fly1 = masks[0]*heatmap

            # Step 1: Convert the PyTorch tensor to a NumPy array
            numpy_image = heatmap_fly1[0][0].cpu().detach().numpy()  # Convert GPU tensor to CPU tensor if necessary

            nump = heatmap[0][0].cpu().detach().numpy()

            npbmask = masks[0].cpu().detach().numpy()

            b1 = masks[0][0][0].cpu().detach().numpy()

            b2 = masks[0][0][1].cpu().detach().numpy()
        else:
            heatmap_fly1 = heatmap


        H = heatmap_fly1.size(2)
        W = heatmap_fly1.size(3)

        y = torch.linspace(-1.0, 1.0, H).cuda()
        x = torch.linspace(-1.0, 1.0, W).cuda()
        
        
        s_y_1 = heatmap_fly1.sum(3)  # (N, K, H)
        s_x_1 = heatmap_fly1.sum(2)  # (N, K, W)
        
        # u_y = (self.H_tensor * s_y).sum(2) / s_y.sum(2)  # (N, K)
        # u_x = (self.W_tensor * s_x).sum(2) / s_x.sum(2)
        u_y_1 = (y * s_y_1).sum(2) / s_y_1.sum(2)  # (N, K)
        u_x_1 = (x * s_x_1).sum(2) / s_x_1.sum(2)
        
        y = torch.reshape(y, (1, 1, H, 1))
        x = torch.reshape(x, (1, 1, 1, W))
        
        # Covariance
        var_y_1 = ((heatmap * y.pow(2)).sum(2).sum(2) - u_y_1.pow(2)).clamp(min=1e-6)
        var_x_1 = ((heatmap * x.pow(2)).sum(2).sum(2) - u_x_1.pow(2)).clamp(min=1e-6)
        
        cov_1 = ((heatmap_fly1 * (x - u_x_1.view(-1, self.K, 1, 1)) * (y - u_y_1.view(-1, self.K, 1, 1))).sum(2).sum(2)) #.clamp(min=1e-6)
        # bboxes = bbox_masks[0, :, :3]
        if len(masks) > 0:
            bboxes = masks[0, :, :]
        else:
            bboxes = []

        confidence = heatmap_fly1.max(dim=-1)[0].max(dim=-1)[0]
                
        # print(u_x[0], u_y[0])
        
        for ii in range(1, self.num_agents):
            #if find_peaks:
            if len(masks) > 0:
                heatmap_fly2 = masks[ii]*heatmap
            else:
                heatmap_fly2 = heatmap

            # if find_peaks:
            #     bbox_wing = self.get_wing_bbox(heatmap_fly2)

            #     heatmap_fly2 = heatmap_fly2 * bbox_wing[0]

            # else:
            #     heatmap_fly2 = heatmap

            H = heatmap_fly2.size(2)
            W = heatmap_fly2.size(3)

            y = torch.linspace(-1.0, 1.0, H).cuda()
            x = torch.linspace(-1.0, 1.0, W).cuda()


            s_y_2 = heatmap_fly2.sum(3)  # (N, K, H)
            s_x_2 = heatmap_fly2.sum(2)  # (N, K, W) 

            # u_y = (self.H_tensor * s_y).sum(2) / s_y.sum(2)  # (N, K)
            # u_x = (self.W_tensor * s_x).sum(2) / s_x.sum(2)
            u_y_2 = (y * s_y_2).sum(2) / s_y_2.sum(2)  # (N, K)
            u_x_2 = (x * s_x_2).sum(2) / s_x_2.sum(2)

            y = torch.reshape(y, (1, 1, H, 1))
            x = torch.reshape(x, (1, 1, 1, W))

            # Covariance
            var_y_2 = ((heatmap * y.pow(2)).sum(2).sum(2) - u_y_2.pow(2)).clamp(min=1e-6)
            var_x_2 = ((heatmap * x.pow(2)).sum(2).sum(2) - u_x_2.pow(2)).clamp(min=1e-6)

            cov_2 = ((heatmap_fly2 * (x - u_x_2.view(-1, self.K, 1, 1)) * (y - u_y_2.view(-1, self.K, 1, 1))).sum(2).sum(2)) #.clamp(min=1e-6)
            confidence_2 = heatmap_fly2.max(dim=-1)[0].max(dim=-1)[0]
            
            """
            u_x = torch.cat([u_x_1, u_x_2], dim = 1)
            u_y = torch.cat([u_y_1, u_y_2], dim = 1)
            var_x = torch.cat([var_x_1, var_x_2], dim = 1)        
            var_y = torch.cat([var_y_1, var_y_2], dim = 1)
            cov = torch.cat([cov_1, cov_2], dim = 1)
            bboxes = torch.cat([bbox_masks[0, :, :3], bbox_masks[1, :, :3]], dim = 1)
            """
            u_x_1 = torch.cat([u_x_1, u_x_2], dim = 1)
            u_y_1 = torch.cat([u_y_1, u_y_2], dim = 1)
            var_x_1 = torch.cat([var_x_1, var_x_2], dim = 1)        
            var_y_1 = torch.cat([var_y_1, var_y_2], dim = 1)
            cov_1 = torch.cat([cov_1, cov_2], dim = 1)
            # bboxes = torch.cat([bboxes, bbox_masks[ii, :, :3]], dim = 1)
            if len(masks) > 0:
                bboxes = torch.cat([bboxes, masks[ii, :, :]], dim = 1)

            confidence = torch.cat([confidence, confidence_2], dim=1)

        # return u_x, u_y, (var_x, var_y, cov) , bboxes #(heatmap, xs, ys)
        return u_x_1, u_y_1, (var_x_1, var_y_1, cov_1), bboxes, confidence
    
    
    def _kptTomap(self, u_x, u_y, inv_std=1.0/0.1, H=16, W=16, normalize=False):        
        mu_x = u_x.unsqueeze(2).unsqueeze(3)  # (N, K, 1, 1)
        mu_y = u_y.unsqueeze(2) 
        
        y = torch.linspace(-1.0, 1.0, H).cuda()
        x = torch.linspace(-1.0, 1.0, W).cuda()
        y = torch.reshape(y, (1, H))
        x = torch.reshape(x, (1, 1, W))
        
        g_y = (mu_y - y).pow(2)
        g_x = (mu_x - x).pow(2)
        
        g_y = g_y.unsqueeze(3)
        g_yx = g_y + g_x
        
        g_yx = torch.exp(- g_yx / (2 * inv_std) ) * 1 / math.sqrt(2 * math.pi * inv_std)
        
        if normalize:
            g_yx = g_yx / g_yx.sum(2, keepdim=True).sum(3, keepdim=True)
        
        return g_yx

    def getMask(self, merged_mask):
        pred_masks = []
        unique_labels = np.unique(merged_mask)
        unique_labels = unique_labels[unique_labels != 0]
        for label in unique_labels:
            single_mask = np.where(merged_mask == label, label, 0)
            pred_masks.append(single_mask)

        for iii in range(len(pred_masks)):
            pred_masks[iii] = np.repeat(pred_masks[iii][np.newaxis, :, :], self.K, axis=0)
        if len(pred_masks) > 0:
            while len(pred_masks) < self.num_agents:
                pred_masks.append(pred_masks[-1])
            del pred_masks[self.num_agents:]

            # Stack the arrays along a new axis
            masks = np.stack(pred_masks, axis=0)

            # Add an extra dimension to get the desired shape
            masks = np.expand_dims(masks, axis=1)
            tensor_masks = torch.from_numpy(masks)
            tensor_masks = tensor_masks.to(torch.float32)
            tensor_masks = tensor_masks.to('cuda')
            resized_tensor_masks = nnf.interpolate(tensor_masks, size=(self.K, self.output_shape[0], self.output_shape[0]), mode='trilinear', align_corners=False)
        else:
            resized_tensor_masks = []

        return resized_tensor_masks


def _nms(heat, kernel=11):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (heat == hmax).float()
    return heat * keep

def _topk(scores, K=40):
    batch, cat, height, width = scores.size()
      
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()

    return topk_scores, topk_inds, topk_ys, topk_xs


def tusk(heatmap):
    # heatmap: (N, K, H, W)
    import pdb; pdb.set_trace()
    nms = _nms(heatmap, kernel=11)
    
