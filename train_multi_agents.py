import time
import warnings

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

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

from PIL import Image

import numpy as np
import torch
import imageio
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
import gc

from Tracking_Anything_with_DEVA.demo.run_with_text import run_demo

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

    #SEGMENTATION HERE
    #################################################################################
    #might have to do for each video in dir in the future
    traindir = os.path.join(args.data, 'train/video1')
    valdir = os.path.join(args.data, 'val/video2')
    train_masks = run_demo(4, traindir, True, 'semionline', 800, './example/output', 'rat.mouse', 0.5)
    val_masks = run_demo(4, valdir, True, 'semionline', 800, './example/output', 'rat.mouse', 0.5)
    #################################################################################

    # create model
    output_shape = (int(args.image_size/4), int(args.image_size/4))
    model = Model(args.nkpts, output_shape=output_shape, num_agents=args.num_agents, frame_gap=args.frame_gap)

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
        validate(val_loader, model, loss_module, 0, args, val_masks)
        return

    is_best = True

    rloss = []

    for epoch in range(args.start_epoch, args.epochs):

        running_loss = 0.0

        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train_loss, running_loss, rloss = train(train_loader, model, loss_module, optimizer, epoch, args, running_loss, rloss, train_masks)

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

        
def train(train_loader, model, loss_module, optimizer, epoch, args, running_loss, rloss, train_masks):

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

        frame_idx = i*args.frame_gap

        if args.gpu is not None:
            inputs, tr_inputs, loss_mask, in_mask = images[0].cuda(args.gpu, non_blocking=True), \
                                                    images[1].cuda(args.gpu, non_blocking=True), \
                                                    images[2].cuda(args.gpu, non_blocking=True), \
                                                    images[3].cuda(args.gpu, non_blocking=True)

            rot_im1, rot_im2,rot_im3  = images[4].cuda(args.gpu, non_blocking=True), \
                            images[5].cuda(args.gpu, non_blocking=True), \
                            images[6].cuda(args.gpu, non_blocking=True)

        if epoch < args.curriculum:
            output = model(inputs, tr_inputs, frame_idx=frame_idx, masks=train_masks)
        else:
            output = model(inputs, tr_inputs, gmtr_x1 = rot_im1, gmtr_x2 = rot_im2, gmtr_x3 = rot_im3, frame_idx=frame_idx, masks=train_masks)

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

def validate(val_loader, model, loss_module, epoch, args, val_masks):

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
            frame_idx = i * args.frame_gap

            if args.gpu is not None:
                inputs, tr_inputs, loss_mask, in_mask = images[0].cuda(args.gpu, non_blocking=True), \
                                                    images[1].cuda(args.gpu, non_blocking=True), \
                                                    images[2].cuda(args.gpu, non_blocking=True), \
                                                    images[3].cuda(args.gpu, non_blocking=True)

            rot_im1, rot_im2,rot_im3  = images[4].cuda(args.gpu, non_blocking=True), \
                            images[5].cuda(args.gpu, non_blocking=True), \
                            images[6].cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(inputs, tr_inputs, gmtr_x1 = rot_im1, gmtr_x2 = rot_im2, gmtr_x3 = rot_im3, frame_idx=frame_idx, masks=val_masks)
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
