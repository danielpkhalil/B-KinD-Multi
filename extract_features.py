import argparse
import cv2

import numpy as np

import os

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import torchvision

from model.unsupervised_model_multi_agents import Model as BKinD_multi
#from model.kpt_detector import Model
from Tracking_Anything_with_DEVA.demo.run_with_text import run_demo

from PIL import Image
import seaborn as sns


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

# resume, checkpoint, num keypoints
def load_model(resume, output_dir, num_agents, frame_gap, image_size=256, num_keypoints = 10):
    output_shape = (int(image_size / 4), int(image_size / 4))
    gpu = 0
    #model = Model(num_keypoints)
    bkind_model = BKinD_multi(num_keypoints, output_shape=output_shape, num_agents=num_agents, frame_gap=frame_gap)

    save_dir = os.path.join(output_dir, 'keypoints_confidence')

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        os.mkdir(os.path.join(save_dir, 'train'))
        os.mkdir(os.path.join(save_dir, 'test'))

    if os.path.isfile(resume):
        if gpu is not None:
            bkind_model = bkind_model.cuda(gpu)
            loc = 'cuda:{}'.format(gpu)
            checkpoint = torch.load(resume, map_location=loc)
        else:
            checkpoint = torch.load(resume)

        bkind_model.load_state_dict(checkpoint['state_dict'])
        bkind_model_dict = bkind_model.state_dict()

        print("=> loaded checkpoint '{}' (epoch {})"
          .format(resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(resume))

    # model.eval()
    # org_model.eval()

    return bkind_model, save_dir


def get_image_tensor(frame, size = 256):

    current = Image.fromarray(frame)

    crop_percent = 1.0
    final_sz = size
    resize_sz = np.round(final_sz / crop_percent).astype(np.int32)

    current = torchvision.transforms.Resize((resize_sz, resize_sz))(current)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    to_tensor = transforms.Compose([
                          transforms.Resize(size),
                          transforms.CenterCrop(size),
                          transforms.ToTensor(),
                          normalize,])

    current_tensor = to_tensor(current)

    return current_tensor.unsqueeze(0)



def compute_keypoints(inputs, model, frame_idx, masks, width = 1024, height = 570):

    # Assume GPU 0
    loc = 'cuda:{}'.format(0)
    inputs = inputs.to(loc)

    output = model(inputs, frame_idx=frame_idx, masks=masks)

    xy = torch.stack((output[0][0], output[0][1]), dim=2).detach().cpu().numpy()[0]+1

    scale_x = (width / 2.0)

    scale_y = (height / 2.0)

    to_plot = []

    confidence = output[5].detach().cpu().numpy()[0]

    covs = torch.stack((output[6][0], output[6][1], output[6][2]), dim=2).detach().cpu().numpy()[0]+1


    for i in range(0,xy.shape[0]):

        st_y = int(xy[i,1]*scale_y); st_x = int(xy[i,0]*scale_x)

        to_plot.append([st_y, st_x])

    return xy, to_plot, confidence, covs


# Parse input arguments.
ap = argparse.ArgumentParser()
ap.add_argument("--train_dir", help="Path to train directory with directory of images", type = str)
ap.add_argument("--test_dir", help="Path to test directory with directory of images", type = str)
ap.add_argument("--resume", help="Path to checkpoint to resume", type = str)
ap.add_argument("--output_dir", help="Output directory to store the keypoints", type = str)
ap.add_argument("--imsize", default=256, help="Training image size", type=int)
ap.add_argument("--nkpts", default=7, help="Number of discovered keypoints", type=int)
ap.add_argument("--conf_threshold", default=0.1, help="Confidence threshold for plotting less confident points", type=float)

args = vars(ap.parse_args())

num_agents = 2
frame_gap = 1
prompt = 'rat.mouse'

model, save_dir = load_model(args['resume'], args['output_dir'], num_agents, frame_gap, args['imsize'], args['nkpts'])

# input train & test directory
train_dir = args['train_dir']
test_dir = args['test_dir']

counter = 0


# Extract for train dir
for vid in sorted(os.listdir(train_dir)):
    print(counter, vid)

    counter = counter + 1

    keypoint_array =[]
    conf_array = []
    covs_array = []

    vid_name = vid

    current_directory = os.path.join(train_dir, vid)

    save_img_dir = os.path.join(save_dir, 'train_samples')
    num_imgs = len(os.listdir(current_directory))
    sample_ids = np.random.permutation(num_imgs)
    sample_ids = sample_ids[:min(100, num_imgs)]

    if not os.path.isdir(save_img_dir):
        os.makedirs(save_img_dir)

    train_masks = run_demo(4, current_directory, True, 'semionline', 144, './Tracking_Anything_with_DEVA/example/output',
                     prompt, 0.47)

    for ix, images in enumerate(sorted(os.listdir(current_directory))):

        draw_frame = cv2.cvtColor(cv2.imread(os.path.join(current_directory, images), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        height, width, _ = draw_frame.shape
        input_tensor = get_image_tensor(draw_frame)

        _, plot_keypoints, confidence, covs = compute_keypoints(input_tensor, model, ix, train_masks, width=width, height=height)

        if ix == 0:
            values = sns.color_palette("husl", len(plot_keypoints))
            colors = []
            for i in range(len(values)):
                color = [int(values[i][0]*255), int(values[i][1]*255), int(values[i][2]*255)]
                colors.append(color)

        image = draw_frame

        # For visualization (randomly sample 100 images)
        if ix in sample_ids:
            for c, j in enumerate(range(len(plot_keypoints))):
                item = plot_keypoints[j]
                if confidence[j] > args['conf_threshold']:
                    image = cv2.circle(image, (item[1], item[0]),
                        radius=3, color=colors[c], thickness = 3)
                else:
                    image = cv2.circle(image, (item[1], item[0]),
                        radius=2, color=colors[c], thickness = 1)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(save_img_dir, 'image_'+str(ix)+'.png'), image)

        conf_array.append(confidence)
        keypoint_array.append(plot_keypoints)
        covs_array.append(covs)

    print(np.array(keypoint_array).shape, np.array(conf_array).shape, np.array(covs_array).shape)


    np.savez(os.path.join(save_dir, 'train', vid_name), keypoints = keypoint_array, confidence = conf_array,
            covs = covs_array)

torch.cuda.empty_cache()

counter = 0
# Extract for test dir
for vid in sorted(os.listdir(test_dir)):
    print(counter, vid)

    counter = counter + 1

    keypoint_array =[]
    conf_array = []
    covs_array = []

    vid_name = vid

    current_directory = os.path.join(test_dir, vid)
    
    save_img_dir = os.path.join(save_dir, 'test_samples')
    num_imgs = len(os.listdir(current_directory))
    sample_ids = np.random.permutation(num_imgs)
    sample_ids = sample_ids[:min(100, num_imgs)]
    
    if not os.path.isdir(save_img_dir):
        os.makedirs(save_img_dir)

    test_masks = run_demo(4, current_directory, True, 'semionline', 144, './Tracking_Anything_with_DEVA/example/output',
                     prompt, 0.47)
    
    for ix, images in enumerate(sorted(os.listdir(current_directory))):

        draw_frame = cv2.cvtColor(cv2.imread(os.path.join(current_directory, images), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)        
        height, width, _ = draw_frame.shape
        input_tensor = get_image_tensor(draw_frame)

        _, plot_keypoints, confidence, covs = compute_keypoints(input_tensor, model, ix, test_masks, width=width, height=height)
        
        if ix == 0:
            values = sns.color_palette("husl", len(plot_keypoints))
            colors = []
            for i in range(len(values)):
                color = [int(values[i][0]*255), int(values[i][1]*255), int(values[i][2]*255)]
                colors.append(color)

        image = draw_frame
        
        # For visualization (randomly sample 100 images)
        if ix in sample_ids:
            for c, j in enumerate(range(len(plot_keypoints))):
                item = plot_keypoints[j]
                if confidence[j] > args['conf_threshold']:
                    image = cv2.circle(image, (item[1], item[0]),
                        radius=3, color=colors[c], thickness = 3)
                else:                
                    image = cv2.circle(image, (item[1], item[0]),
                        radius=2, color=colors[c], thickness = 1)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(save_img_dir, 'image_'+str(ix)+'.png'), image)

        conf_array.append(confidence)
        keypoint_array.append(plot_keypoints)
        covs_array.append(covs)

    print(np.array(keypoint_array).shape, np.array(conf_array).shape, np.array(covs_array).shape)

    np.savez(os.path.join(save_dir, 'test', vid_name), keypoints = keypoint_array, confidence = conf_array,
            covs = covs_array)
