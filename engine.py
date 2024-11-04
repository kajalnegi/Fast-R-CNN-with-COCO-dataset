import math
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
import utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
from forward_loss import eval_forward


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    batch_loss_classifier = []
    batch_box_reg = []
    batch_loss_rpn_box_reg = []
    batch_objectness = []
    total_loss = 0
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        loss_classifier = loss_dict['loss_classifier'].item()
        #print(loss_classifier)
        loss_box_reg = loss_dict['loss_box_reg'].item()
        loss_objectness = loss_dict['loss_objectness'].item()
        loss_rpn_box_reg = loss_dict['loss_rpn_box_reg'].item()
        batch_loss_classifier.append(loss_classifier)
        batch_box_reg.append(loss_box_reg)
        batch_objectness.append(loss_objectness)
        batch_loss_rpn_box_reg.append(loss_rpn_box_reg)
        total_loss = total_loss + losses.item()
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            #print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()
        
        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        #break
    loss_classifier = sum(batch_loss_classifier)/len(batch_loss_classifier)
    loss_box_reg = sum(batch_box_reg)/len(batch_box_reg)
    loss_objectness = sum(batch_objectness)/len(batch_objectness)
    loss_rpn_box_reg = sum(batch_loss_rpn_box_reg)/len(batch_loss_rpn_box_reg)
    avg_total_loss = total_loss/len(data_loader)
    return loss_classifier, loss_box_reg, loss_objectness,loss_rpn_box_reg, avg_total_loss#, metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)
    val_batch_loss_classifier = []
    val_batch_box_reg = []
    val_batch_objectness = []
    val_batch_loss_rpn_box_reg = []
    total_loss = 0
    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"]: output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
        loss_dict, detections = eval_forward(model, images, targets, device)
        val_loss_classifier = loss_dict['loss_classifier'].item()
        val_loss_box_reg = loss_dict['loss_box_reg'].cpu().item()
        val_loss_objectness = loss_dict['loss_objectness'].item()
        val_loss_rpn_box_reg = loss_dict['loss_rpn_box_reg'].item()
        val_batch_loss_classifier.append(val_loss_classifier)
        val_batch_box_reg.append(val_loss_box_reg)
        val_batch_objectness.append(val_loss_objectness)
        val_batch_loss_rpn_box_reg.append(val_loss_rpn_box_reg)
        total_loss = total_loss + sum(loss for loss in loss_dict.values()).item()
        #break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    pr_metr = coco_evaluator.summarize()
    torch.set_num_threads(n_threads)

    #loss
    loss_classifier = sum(val_batch_loss_classifier)/len(val_batch_loss_classifier)
    loss_box_reg = sum(val_batch_box_reg)/len(val_batch_box_reg)
    loss_objectness = sum(val_batch_objectness)/len(val_batch_objectness)
    loss_rpn_box_reg = sum(val_batch_loss_rpn_box_reg)/len(val_batch_loss_rpn_box_reg)
    avg_total_loss = total_loss/len(data_loader)
    return coco_evaluator, loss_classifier, loss_box_reg, loss_objectness,loss_rpn_box_reg, avg_total_loss, pr_metr