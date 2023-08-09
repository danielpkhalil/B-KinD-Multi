import torch
import torch.nn as nn

from .resnet_updated import resnetbank50all as resnetbank50
from .globalNet import globalNet


import os
import cv2
from utils.visualize import show_heatmaps

    
class Model(nn.Module):
    def __init__(self, n_kpts=10, pretrained=True, output_shape=(64, 64)):
        super(Model, self).__init__()
        self.K = n_kpts
        
        channel_settings = [2048, 1024, 512, 256]
        self.output_shape = output_shape
        self.kptNet = globalNet(channel_settings, output_shape, n_kpts)
        self.ch_softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.num_agents = 11
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        self.encoder = resnetbank50(pretrained=pretrained)
        
        
    def _lateral(self, input_size, output_shape):
        out_dim = 256
        
        layers = []
        layers.append(nn.Conv2d(input_size, out_dim,
            kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(out_dim))
        layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.Conv2d(out_dim, out_dim,
            kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.Upsample(size=output_shape, mode='bilinear', align_corners=True))
        layers.append(nn.BatchNorm2d(out_dim))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)
        
    def forward(self, x):
                
        x_res = self.encoder(x)
        
        kpt_feat, kpt_out = self.kptNet(x_res)
        
        # Classification module
        # heatmap = kpt_out[-1].view(-1, self.K, kpt_out[-1].size(2) * kpt_out[-1].size(3))
        # heatmap = self.ch_softmax(heatmap) 
        # heatmap = heatmap.view(-1, self.K, kpt_out[-1].size(2), kpt_out[-1].size(3))

        # im_dir = os.path.join("heat")
        # heatmaps = show_heatmaps(heatmap)
        # heatmaps = (heatmaps.data.cpu().numpy() * 255).astype('uint8')
        # heatmaps = heatmaps.transpose((1,2,0))
        # cv2.imwrite(os.path.join(im_dir, 'heatmapsN_'+str(1)+'.png'), heatmaps)
        
        confidence = heatmap.max(dim=-1)[0].max(dim=-1)[0]
            
        u_x, u_y, covs, bbox, confidence = self._mapTokpt(heatmap)
        
        kpts = []
        for ii in range(1, len(kpt_out)):
            _heatmap = kpt_out[ii-1].view(-1, self.K, kpt_out[ii-1].size(2) * kpt_out[ii-1].size(3))
            _heatmap = self.ch_softmax(_heatmap) 
            _heatmap = _heatmap.view(-1, self.K, kpt_out[ii-1].size(2), kpt_out[ii-1].size(3))

            _u_x, _u_y, _covs, _bbox, _conf = self._mapTokpt(_heatmap)
            kpts.append((u_x, u_y))
        
        return (u_x, u_y), kpt_out[-1], heatmap, kpts, kpt_out, confidence, covs
        
    def get_fly_bboxes(self, heatmap, find_peaks):
        # Given a rough agent size / identity-free
        ##################### Get two bboxes     
        # 9 x 5 x 64 x 64

        summed = torch.sum(heatmap, dim = 1, keepdim = True)
        summed = _nms(summed, kernel = 15)

        scores, inds, ys, xs = _topk(summed, K=1)
        scores = scores.repeat((1,5, 1))
        ys = ys.repeat((1,5,1))
        xs = xs.repeat((1,5,1))
        
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
            X = X.unsqueeze(0).repeat(batch_size*5, 1, 1).to(heatmap.device)
            Y = Y.unsqueeze(0).repeat(batch_size*5, 1, 1).to(heatmap.device)


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
        rad = 10 # 10
        dist_mask = create_circular_mask(bbox_y[:, 0], bbox_x[:, 0], rad).view((batch_size, 5, self.output_shape[0], self.output_shape[0])) # 64, 64))

        dist_mask_small = create_circular_mask(bbox_y[:, 0], bbox_x[:, 0], rad).view((batch_size, 5, self.output_shape[0], self.output_shape[0])) #64, 64))


        tmp = dist_mask #torch.mul(keep, dist_mask)

        bboxes[0] = tmp
        without_top = summed
        
        ##############################
        for ii in range(1, self.num_agents):
            without_top = torch.mul(without_top, 1.0 - dist_mask[:, 0, :, :].detach().long().unsqueeze(1))
            #print(without_top.size(), summed.size(), dist_mask[:, 0, :, :].size())
            scores, inds, ys, xs = _topk(without_top, K=1)
            scores = scores.repeat((1,5, 1))
            ys = ys.repeat((1,5,1))
            xs = xs.repeat((1,5,1))        


            bbox_y = ys.view((-1, 1)).long()
            bbox_x = xs.view((-1, 1)).long()
            bbox_scores = scores.view(-1, 1)      

            comparison_score2 = scores[:, :, 0].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.output_shape[0], self.output_shape[0]) #64, 64)

            # rad = 10
            dist_mask = create_circular_mask(bbox_y[:, 0], bbox_x[:, 0], rad).view((batch_size, 5, self.output_shape[0], self.output_shape[0])) #64, 64))

            tmp = dist_mask#torch.mul(keep2, dist_mask)

            bboxes[ii] = tmp

        return bboxes.to(heatmap.device)
    
        
    def _mapTokpt(self, heatmap, find_peaks=False, use_bbox=True):
        # heatmap: (N, K, H, W) 
        if use_bbox:
            # Summed bbox for each fly
            bbox_masks = self.get_fly_bboxes(heatmap, find_peaks = find_peaks).detach()
        else:
            # max locations
            bbox_masks = self.get_wing_bbox(heatmap).detach()
            
        heatmap_fly1 = bbox_masks[0]*heatmap


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
        bboxes = bbox_masks[0, :, :3]
        confidence = heatmap_fly1.max(dim=-1)[0].max(dim=-1)[0]
                
        # print(u_x[0], u_y[0])
        
        for ii in range(1, self.num_agents):
            #if find_peaks:
            heatmap_fly2 = bbox_masks[ii]*heatmap
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
            bboxes = torch.cat([bboxes, bbox_masks[ii, :, :3]], dim = 1)
            confidence = torch.cat([confidence, confidence_2], dim=1)

        # return u_x, u_y, (var_x, var_y, cov) , bboxes #(heatmap, xs, ys)
        return u_x_1, u_y_1, (var_x_1, var_y_1, cov_1), bboxes, confidence
    
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
    
#     def _mapTokpt(self, heatmap):
#         # heatmap: (N, K, H, W)    
            
#         H = heatmap.size(2)
#         W = heatmap.size(3)
        
#         s_y = heatmap.sum(3)  # (N, K, H)
#         s_x = heatmap.sum(2)  # (N, K, W)
        
#         y = torch.linspace(-1.0, 1.0, H).cuda()
#         x = torch.linspace(-1.0, 1.0, W).cuda()
        
#         # u_y = (self.H_tensor * s_y).sum(2) / s_y.sum(2)  # (N, K)
#         # u_x = (self.W_tensor * s_x).sum(2) / s_x.sum(2)
#         u_y = (y * s_y).sum(2) / s_y.sum(2)  # (N, K)
#         u_x = (x * s_x).sum(2) / s_x.sum(2)
        
#         y = torch.reshape(y, (1, 1, H, 1))
#         x = torch.reshape(x, (1, 1, 1, W))
        
#         # Covariance
#         var_y = ((heatmap * y.pow(2)).sum(2).sum(2) - u_y.pow(2)).clamp(min=1e-6)
#         var_x = ((heatmap * x.pow(2)).sum(2).sum(2) - u_x.pow(2)).clamp(min=1e-6)
        
#         cov = ((heatmap * (x - u_x.view(-1, self.K, 1, 1)) * (y - u_y.view(-1, self.K, 1, 1))).sum(2).sum(2)) #.clamp(min=1e-6)
                
#         return u_x, u_y, (var_x, var_y, cov)
    