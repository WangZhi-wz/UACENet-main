import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
from opt import opt



class structure_loss(nn.Module):
    def __init__(self):
        super(structure_loss, self).__init__()

    def forward(self, pred, mask):
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        return (wbce + wiou).mean()



"""BCE loss"""


class BCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss(weight=weight, size_average=size_average)

    def forward(self, pred, target):
        size = pred.size(0)
        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)

        loss = self.bceloss(pred_flat, target_flat)

        return loss


"""wBCE loss"""

class BCELossw(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELossw, self).__init__()
        self.bceloss = nn.BCEWithLogitsLoss(weight=weight, size_average=size_average)

    def forward(self, pred, target):
        size = pred.size(0)
        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)

        weit = 1 + 5 * torch.abs(F.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target)

        wbce = self.bceloss(pred_flat, target_flat)
        loss = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))


        return loss.mean()


"""Dice loss"""

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1

        size = pred.size(0)

        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)

        intersection = pred_flat * target_flat
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_flat.sum(1) + target_flat.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size

        return dice_loss


"""wDice loss"""

class DiceLossw(nn.Module):
    def __init__(self):
        super(DiceLossw, self).__init__()

    def forward(self, pred, target):
        smooth = 1

        size = pred.size(0)

        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)

        weit = 1 + 5 * torch.abs(F.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target)

        intersection = pred_flat * target_flat
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_flat.sum(1) + target_flat.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size
        dice_loss = (weit * dice_loss).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))


        return dice_loss.mean()




"""BCE + DICE Loss + IoULoss"""
class BceDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BceDiceLoss, self).__init__()
        # self.bce = BCELossw(weight, size_average)
        self.fl = FocalLoss()
        self.dice = DiceLoss()
        self.iou = IoULossW()
        # self.structure = structure_loss()

    # def tversky_index(self, y_true, y_pred):
    #     smooth = 1
    #     y_true_pos = torch.flatten(y_true)
    #     y_pred_pos = torch.flatten(y_pred)
    #     true_pos = torch.sum(y_true_pos * y_pred_pos)
    #     false_neg = torch.sum(y_true_pos * (1 - y_pred_pos))
    #     false_pos = torch.sum((1 - y_true_pos) * y_pred_pos)
    #     alpha = 0.7
    #     return (true_pos + smooth) / (true_pos + alpha * false_neg + (
    #             1 - alpha) * false_pos + smooth)
    #
    # def tversky_loss(self, y_true, y_pred):
    #     return 1 - self.tversky_index(y_true, y_pred)
    #
    # def focal_tversky(self, y_true, y_pred):
    #     pt_1 = self.tversky_index(y_true, y_pred)
    #     gamma = 0.75
    #     return torch.pow((1 - pt_1), gamma)

    def forward(self, pred, target):
        # bceloss = self.bce(pred, target)
        fcloss = self.fl(pred, target)
        diceloss = self.dice(pred, target)
        iouloss = self.iou(pred, target)
        # tverskyloss = self.tversky_loss(target, pred)
        # structure_loss = self.structure(pred, target)  # wbce + wiou
        loss = fcloss + diceloss + iouloss    #Use the obmination of loss

        return loss




""" Deep Supervision Loss"""


def DeepSupervisionLoss(pred, gt):
    d0, d1, d2, d3, d4 = pred[0:]

    criterion = BceDiceLoss()

    loss0 = criterion(d0, gt)
    gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True, recompute_scale_factor=True)
    loss1 = criterion(d1, gt)
    gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True, recompute_scale_factor=True)
    loss2 = criterion(d2, gt)
    gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True, recompute_scale_factor=True)
    loss3 = criterion(d3, gt)
    gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True, recompute_scale_factor=True)
    loss4 = criterion(d4, gt)

    return loss0 + loss1 + loss2 + loss3 + loss4


"""Focal loss"""
class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=4,reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):

        ce_loss = F.binary_cross_entropy(input, target,reduction=self.reduction,weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


"""wFocal loss"""
class wFocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=4,reduction='mean'):
        super(wFocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):
        weit = 1 + 5 * torch.abs(F.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target)

        ce_loss = F.binary_cross_entropy(input, target,reduction=self.reduction,weight=self.weight)
        ce_loss = (weit * ce_loss).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()

        return focal_loss


"""IoU loss"""
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth)/(union + smooth)


        return 1 - IoU



"""wIoU loss"""
class IoULossW(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULossW, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        pred = torch.sigmoid(inputs)
        weit = 1 + 5*torch.abs(F.avg_pool2d(targets, kernel_size=31, stride=1, padding=15) - targets)
        inter = ((pred * targets)* weit).sum(dim=(2, 3))
        #print(inter.shape)
        union = ((pred + targets)* weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1)/(union - inter+1)
        wiou = wiou.mean()*opt.weight_const

        return wiou
