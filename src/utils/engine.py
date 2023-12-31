import math
import sys
import time
import torch
from torch import inf
import torchvision.models.detection.mask_rcnn
import cv2
import numpy as np

from utils.coco_utils import get_coco_api_from_dataset
from utils.coco_eval import CocoEvaluator
import utils.utils as utils
import config as config


def train_one_epoch(model, optimizer, data_loader, epoch, print_freq, device=config.device):
    """
    Train the model for one epoch
    """
    # Training with CUDA capable GPU
    if device == 'cuda':
        model.train()
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)

        lr_scheduler = None

        if epoch == 0:
            warmup_factor = 1. / 1000
            warmup_iters = min(1000, len(data_loader) - 1)

            lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

        for images, targets in metric_logger.log_every(data_loader, print_freq, header):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs (if available) for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            loss_value = losses_reduced.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

            metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        return metric_logger

    # Training with CPU
    else:
        model.train()
        print('Epoch: [{}]'.format(epoch))

        lr_scheduler = None

        for images, targets in enumerate(data_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            loss_value = losses.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict)
                sys.exit(1)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

            print(f'Losses: {losses}, {loss_dict}')
            print(f'Learning rate: {optimizer.param_groups[0]["lr"]}')


def _get_iou_types(model):
    """
    Get the evaluation metrics for the model
    """
    model_without_ddp = model

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]

    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")

    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, device):
    """
    Evaluate the model on the validation set
    """
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    # Change print_freq to smaller number to print more often
    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}

        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # Accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)

    return coco_evaluator
