import time
import warnings

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

from SAMT.SegTracker import SegTracker
from config.config_reader import parse_args, create_parser

from dataloader.load_dataloader import load_dataloader

from model.unsupervised_model_multi_agents import Model
# from model.tusk import Model

from loss.compute_loss import *

from utils import Logger, mkdir_p, save_images
from utils.model_utils import *

import matplotlib.pyplot as plt
plt.ioff()

import os
import cv2
from SAMT.SegTracker import SegTracker
from SAMT.model_args import aot_args, sam_args, segtracker_args
from PIL import Image
from SAMT.aot_tracker import _palette
import numpy as np
import torch
import imageio
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
import gc



best_loss = 10000

def main():

    args = parse_args(create_parser())

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    global best_loss
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # init segtracker
    #################################################################################
    # segment frame
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
    #################################################################################

    # create model
    output_shape = (int(args.image_size/4), int(args.image_size/4))
    model = Model(args.nkpts, output_shape=output_shape, num_agents=args.num_agents, segtracker=segtracker, grounding_caption=grounding_caption, box_threshold=box_threshold, text_threshold=text_threshold, box_size_threshold=box_size_threshold, reset_image=reset_image)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    loss_module = computeLoss(args)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    title = 'Landmark-discovery'
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']

            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss',])

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data)
    valdir = os.path.join(args.data)

    train_dataset, val_dataset = load_dataloader(args)
    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, loss_module, 0, args)
        return

    is_best = True

    # #init segtracker
    # #################################################################################
    # # segment frame
    # sam_args['generator_args'] = {
    #     'points_per_side': 30,
    #     'pred_iou_thresh': 0.8,
    #     'stability_score_thresh': 0.9,
    #     'crop_n_layers': 1,
    #     'crop_n_points_downscale_factor': 2,
    #     'min_mask_region_area': 200,
    # }
    # # Set Text args
    # '''
    # parameter:
    #     grounding_caption: Text prompt to detect objects in key-frames
    #     box_threshold: threshold for box
    #     text_threshold: threshold for label(text)
    #     box_size_threshold: If the size ratio between the box and the frame is larger than the box_size_threshold, the box will be ignored. This is used to filter out large boxes.
    #     reset_image: reset the image embeddings for SAM
    # '''
    # grounding_caption = "rat"
    # box_threshold, text_threshold, box_size_threshold, reset_image = 0.35, 0.5, 0.5, True
    # frame_idx = 0
    # segtracker = SegTracker(segtracker_args, sam_args, aot_args)
    # segtracker.restart_tracker()
    # #################################################################################
    
    rloss = []

    for epoch in range(args.start_epoch, args.epochs):

        running_loss = 0.0

        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train_loss, running_loss, rloss = train(train_loader, model, loss_module, optimizer, epoch, args, running_loss, rloss)

        # evaluate on validation set every val_schedule epochs
        if epoch > 0 and epoch%args.val_schedule == 0:
            test_loss = validate(val_loader, model, loss_module, epoch, args)
        else:
            test_loss = 10000  # set test_loss = 100000 when not using validate

        logger.append([args.lr * (0.1 ** (epoch // args.schedule)), train_loss, test_loss])

        # remember best acc@1 and save checkpoint
        is_best = test_loss < best_loss
        best_loss = min(test_loss, best_loss)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer' : optimizer.state_dict(),
        }, is_best, checkpoint=args.checkpoint)

        saveLoss(args, rloss, epoch)

        
def train(train_loader, model, loss_module, optimizer, epoch, args, running_loss, rloss):

    epoch=epoch+1

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()

    for i, images in enumerate(train_loader):
        # if(i>=200):
        #     break
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            inputs, tr_inputs, loss_mask, in_mask = images[0].cuda(args.gpu, non_blocking=True), \
                                                    images[1].cuda(args.gpu, non_blocking=True), \
                                                    images[2].cuda(args.gpu, non_blocking=True), \
                                                    images[3].cuda(args.gpu, non_blocking=True)

            rot_im1, rot_im2,rot_im3  = images[4].cuda(args.gpu, non_blocking=True), \
                            images[5].cuda(args.gpu, non_blocking=True), \
                            images[6].cuda(args.gpu, non_blocking=True)

        if epoch < args.curriculum:
            output = model(inputs, tr_inputs)
        else:
            output = model(inputs, tr_inputs, gmtr_x1 = rot_im1, gmtr_x2 = rot_im2, gmtr_x3 = rot_im3)

        loss = loss_module.update_loss(inputs, tr_inputs, loss_mask, output, epoch)

        running_loss += loss.item()
        
        # measure accuracy and record loss
        losses.update(loss.item(), images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            
            if args.visualize and epoch%2==0:
                if epoch < args.curriculum:
                    save_images(tr_inputs, output, epoch, args, epoch, i)
                else:
                    save_images(tr_inputs, output, epoch, args, epoch, i)

    epoch_loss = running_loss / len(train_loader)
    rloss.append(epoch_loss)
    return losses.avg, running_loss, rloss 

def validate(val_loader, model, loss_module, epoch, args):

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, images in enumerate(val_loader):

            if args.gpu is not None:
                inputs, tr_inputs, loss_mask, in_mask = images[0].cuda(args.gpu, non_blocking=True), \
                                                    images[1].cuda(args.gpu, non_blocking=True), \
                                                    images[2].cuda(args.gpu, non_blocking=True), \
                                                    images[3].cuda(args.gpu, non_blocking=True)

            rot_im1, rot_im2,rot_im3  = images[4].cuda(args.gpu, non_blocking=True), \
                            images[5].cuda(args.gpu, non_blocking=True), \
                            images[6].cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(inputs, tr_inputs, gmtr_x1 = rot_im1, gmtr_x2 = rot_im2, gmtr_x3 = rot_im3)
            # Note that validate function only computes MSE loss
            loss = loss_module.criterion[1](output[0] * loss_mask, tr_inputs * loss_mask)

            losses.update(loss.item(), images[0].size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

    return losses.avg

def saveLoss(args, loss, curr_epoch): 
    im_dir = os.path.join(args.checkpoint, 'samples/loss')
    
    if not os.path.isdir(im_dir):
        os.makedirs(im_dir)

    plt.plot(loss)

    plt.savefig(os.path.join(im_dir, 'loss'+'.png'))





if __name__ == '__main__':
    main()
