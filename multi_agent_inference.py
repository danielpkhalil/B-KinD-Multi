# Load images and run inference
# Add more images in the samples directory if you want to test BKinD on different samples.
import torch
import torchvision
import torchvision.transforms as transforms

from model.unsupervised_model_multi_agents import Model as BKinD_multi
#from model.kpt_detector import Model

from SAMT.SegTracker import SegTracker
from SAMT.model_args import aot_args, sam_args, segtracker_args

import cv2
from PIL import Image
from utils import visualize_with_circles

import numpy as np
import os

## Input parameters #################################
nKps = 10
num_agents = 4
gpu = 0  # None if gpu is not available
resume = 'checkpoint/custom_dataset/checkpoint.pth.tar'
sample_path = 'sample_images'
image_size = 256
#####################################################

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=mean, std=std)
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),]) #remove normalize

sam_args['generator_args'] = {
        'points_per_side': 30,
        'pred_iou_thresh': 0.8,
        'stability_score_thresh': 0.9,
        'crop_n_layers': 1,
        'crop_n_points_downscale_factor': 2,
        'min_mask_region_area': 200,
}
# Set Text args
'''
parameter:
grounding_caption: Text prompt to detect objects in key-frames
box_threshold: threshold for box 
text_threshold: threshold for label(text)
box_size_threshold: If the size ratio between the box and the frame is larger than the box_size_threshold, the box will be ignored. This is used to filter out large boxes.
reset_image: reset the image embeddings for SAM
'''
grounding_caption = "rat"
box_threshold, text_threshold, box_size_threshold, reset_image = 0.35, 0.5, 0.5, True
frame_idx = 0
segtracker = SegTracker(segtracker_args, sam_args, aot_args)
segtracker.restart_tracker()
## Load a model
# create model
output_shape = (int(image_size/4), int(image_size/4))
#model = Model(nKps, output_shape=output_shape, num_agents=num_agents, segtracker=segtracker, grounding_caption=grounding_caption, box_threshold=box_threshold, text_threshold=text_threshold, box_size_threshold=box_size_threshold, reset_image=reset_image)
bkind_model = BKinD_multi(nKps, output_shape=output_shape, num_agents=num_agents, segtracker=segtracker, grounding_caption=grounding_caption, box_threshold=box_threshold, text_threshold=text_threshold, box_size_threshold=box_size_threshold, reset_image=reset_image)

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

    im = torchvision.transforms.Resize((image_size, image_size))(im)
    im = transform(im)
    im = im.unsqueeze(0).cuda(gpu, non_blocking=True)

    output = bkind_model(im)

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