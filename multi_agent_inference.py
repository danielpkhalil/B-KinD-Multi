# Load images and run inference
# Add more images in the samples directory if you want to test BKinD on different samples.
import torch
import torchvision
import torchvision.transforms as transforms

from model.unsupervised_model_multi_agents import Model as BKinD_multi
#from model.kpt_detector import Model

import cv2
from PIL import Image
from utils import visualize_with_circles

from Tracking_Anything_with_DEVA.demo.run_with_text import run_demo

import numpy as np
import os

## Input parameters #################################
nkpts = 3
num_agents = 10
gpu = 0  # None if gpu is not available
resume = 'checkpoint/custom_dataset/checkpoint.pth.tar'
sample_path = 'sample_images'
image_size = 256
frame_gap=1
#####################################################

# SEGMENTATION HERE
#################################################################################
sample_masks = run_demo(4, sample_path, True, 'semionline', 400, './Tracking_Anything_with_DEVA/example/output',
                       'ant body', 0.1) #adjust as needed
#################################################################################

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=mean, std=std)
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    normalize,]) #remove normalize


## Load a model
# create model
output_shape = (int(image_size/4), int(image_size/4))
#model = Model(nKps, output_shape=output_shape, num_agents=num_agents, segtracker=segtracker, grounding_caption=grounding_caption, box_threshold=box_threshold, text_threshold=text_threshold, box_size_threshold=box_size_threshold, reset_image=reset_image)
bkind_model = BKinD_multi(nkpts, output_shape=output_shape, num_agents=num_agents, frame_gap=frame_gap)

#model = Model(nKps)
if os.path.isfile(resume):
  if gpu is not None:
    bkind_model = bkind_model.cuda(gpu)
    loc = 'cuda:{}'.format(gpu)
    checkpoint = torch.load(resume, map_location=loc)
  else:
    checkpoint = torch.load(resume)

  bkind_model.load_state_dict(checkpoint['state_dict'])
  bkind_model_dict = bkind_model.state_dict()

  # model_dict = model.state_dict()
  # pretrained_dict = {k: v for k, v in bkind_model_dict.items() if k in model_dict}
  # model_dict.update(pretrained_dict)
  # model.load_state_dict(model_dict)

  print("=> loaded checkpoint '{}' (epoch {})"
        .format(resume, checkpoint['epoch']))
else:
  print("=> no checkpoint found at '{}'".format(resume))

## Inference on sample images
sample_files = sorted(os.listdir(sample_path))
for i in range(len(sample_files)):
  if 'output' not in sample_files[i]:
    with open(os.path.join(sample_path, sample_files[i]), 'rb') as f:
      im = Image.open(f)
      im = im.convert('RGB')

    frame_idx = i
    im = torchvision.transforms.Resize((image_size, image_size))(im)
    im = transform(im)
    im = im.unsqueeze(0).cuda(gpu, non_blocking=True)

    output = bkind_model(im, frame_idx=frame_idx, masks=sample_masks)

    combined_mask = output[3][0][0][0].detach().cpu().numpy()
    for m in range(1, output[3].shape[0]):
        mask = output[3][m][0][0].detach().cpu().numpy()
        combined_mask += mask
    combined_mask = combined_mask*255
    combined_mask = combined_mask.astype('uint8')
    cv2.imwrite(os.path.join(sample_path, 'outputMASK_' + str(i + 1) + '.png'), combined_mask)


    pred_kps = torch.stack((output[0][0], output[0][1]), dim=2)
    pred_kps = pred_kps.data.cpu().numpy()

    im_with_kps = visualize_with_circles(im[0].data.cpu().numpy(), pred_kps[0]+1,
                                        output[2][0], mean=mean, std=std)
    im_with_kps = im_with_kps.astype('uint8')
    im_with_kps = cv2.cvtColor(im_with_kps, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(sample_path, 'output_'+str(i+1)+'.png'), im_with_kps)

    #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    #cv2.imshow('image', im_with_kps)


print("==> Output images saved in \'sample_images\' directory")