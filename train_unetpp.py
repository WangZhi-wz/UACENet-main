import datetime
import os
import random
from random import seed

import cv2
import numpy as np
from PIL import Image
from skimage import io

import models

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import datasets
from utils.metrics import evaluate
from opt import opt
from utils.comm import generate_model
from utils.lossforunet import DeepSupervisionLoss, BceDiceLoss
# from utils.loss_mod import DeepSupervisionLoss, BceDiceLoss
# from utils.lossforParNet import DeepSupervisionLoss
from utils.metrics import Metrics
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn



def seed_everything(seed=1029):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

# seed_everything()

def test(model, test_dataloader, total_batch):
    model.eval()
    # metrics_logger initialization
    metrics = Metrics(['recall', 'specificity', 'precision', 'F1', 'F2',
                       'ACC_overall', 'IoU_poly', 'IoU_bg', 'IoU_mean', 'Dice'])

    with torch.no_grad():
        test_loss = 0
        bar = tqdm(enumerate(test_dataloader), total=total_batch)
        for i, data in bar:

            img, gt = data['image'], data['label']

            # img_out = torch.squeeze(img[0]).cpu().numpy()  # 320,320
            # img_out = np.transpose(img_out, [1,2,0])
            #
            # gt_out = torch.squeeze(gt[0]).cpu().numpy()  # 320,320
            # gt_out[gt_out > 0.5] = 255
            # gt_out[gt_out <= 0.5] = 0
            # # 添加一个维度
            # gt_out = np.expand_dims(gt_out, axis=2)
            # # 一维转三维
            # gt_out = np.squeeze(gt_out)
            # gt_out = [gt_out, gt_out, gt_out]
            # gt_out = np.transpose(gt_out, (1, 2, 0))

            # io.imsave(os.path.join("./result/" + opt.model , str(i) + "_old.jpg"), img_out)
            # io.imsave(os.path.join("./result/" + opt.model , str(i) + "_mask.jpg"), gt_out)

            if opt.use_gpu:
                img = img.cuda()    # 1,3,320,320
                gt = gt.cuda()      # 1,1,320,320

            # output = model(img, flag = "test")
            output = model(img)

            # predict = torch.squeeze(output[0]).cpu().numpy()    # 320,320
            #
            # predict[predict > 0.5] = 255
            # predict[predict <= 0.5] = 0
            # # 添加一个维度
            # predict = np.expand_dims(predict, axis=2)
            # # 一维转三维
            # predict = np.squeeze(predict)
            # predict = [predict, predict, predict]
            # predict = np.transpose(predict, (1, 2, 0))
            # # io.imsave(os.path.join("./result/" + opt.model , str(i) + "_pre.jpg"), predict)
            #
            # sep_line = np.ones((320, 10, 3)) * 255
            # all_images = [
            #     img_out*255,
            #     sep_line, gt_out,
            #     sep_line, predict,
            # ]
            # io.imsave(f"./result/{opt.model}/{i+1}.jpg", np.concatenate(all_images, axis=1))

            _recall, _specificity, _precision, _F1, _F2, \
            _ACC_overall, _IoU_poly, _IoU_bg, _IoU_mean, _Dice = evaluate(output, gt, 0.5)

            metrics.update(recall=_recall, specificity=_specificity, precision=_precision,
                           F1=_F1, F2=_F2, ACC_overall=_ACC_overall, IoU_poly=_IoU_poly,
                           IoU_bg=_IoU_bg, IoU_mean=_IoU_mean, Dice=_Dice
                           )
            loss = DeepSupervisionLoss(output, gt)
            bar.set_postfix_str('loss: %.5s' % loss.item())
            test_loss = test_loss + loss
        test_loss_all = test_loss/200
        print("test_loss_all:%.4f" % test_loss_all)

    test_result = metrics.mean(total_batch)

    return test_result, test_loss_all


def valid(model, valid_dataloader, total_batch):
    model.eval()

    # Metrics_logger initialization
    metrics = Metrics(['recall', 'specificity', 'precision', 'F1', 'F2',
                       'ACC_overall', 'IoU_poly', 'IoU_bg', 'IoU_mean', 'Dice'])

    with torch.no_grad():
        bar = tqdm(enumerate(valid_dataloader), total=total_batch)
        val_loss = 0
        for i, data in bar:
            img, gt = data['image'], data['label']

            if opt.use_gpu:
                img = img.cuda()
                gt = gt.cuda()

            # output = model(img, flag='train')
            output = model(img)

            _recall, _specificity, _precision, _F1, _F2, \
            _ACC_overall, _IoU_poly, _IoU_bg, _IoU_mean, _Dice = evaluate(output, gt, 0.5)

            metrics.update(recall=_recall, specificity=_specificity, precision=_precision,
                           F1=_F1, F2=_F2, ACC_overall=_ACC_overall, IoU_poly=_IoU_poly,
                           IoU_bg=_IoU_bg, IoU_mean=_IoU_mean, Dice=_Dice,
                           )
            loss = DeepSupervisionLoss(output, gt)
            bar.set_postfix_str('loss: %.5s' % loss.item())
            val_loss = val_loss + loss
        val_loss_all = val_loss/200
        print("val_loss_all:%.4f" % val_loss_all)

    metrics_result = metrics.mean(total_batch)

    return metrics_result, val_loss_all


def train():
    # time
    x = datetime.datetime.now()

    # tensorboard
    print(
        '运行tensorboard： tensorboard --logdir "D:\\paperin\\Enhanced-U-Net-main\\logs" --bind_all, view at http://wangzhiyyds:6006/')

    # 实例化SummaryWriter对象
    if os.path.exists(f"./weights/{opt.dataset}") is False:
        os.makedirs(f"./weights/{opt.dataset}")
    tb_writer = SummaryWriter(
        log_dir=f"./weights/{opt.dataset}/logs_" + opt.model + "_" + opt.expName + "_" + str(x.strftime('%Y_%m_%d_%H_%M_%S')))

    # 创建存储模型文件
    if os.path.exists(f"F:/checkpoints/{opt.dataset}/" + opt.expName) is False:
        os.makedirs(f"F:/checkpoints/{opt.dataset}/" + opt.expName)
    if os.path.exists(f"./result/{opt.dataset}/" + opt.model) is False:
        os.makedirs(f"./result/{opt.dataset}/" + opt.model)
    # 创建存储结果文件
    if os.path.exists(f"data_record/{opt.dataset}") is False:
        os.makedirs(f"data_record/{opt.dataset}")
    if os.path.exists(f"data_record/{opt.dataset}") is False:
        os.makedirs(f"data_record/{opt.dataset}")

    # model = generate_model(opt)
    model = getattr(models, opt.model)(opt.nclasses)
    if opt.use_gpu:
        model.cuda()
        torch.backends.cudnn.benchmark = True
    # print(model)
    # summary(model, (3, 256, 256), 4)
    # model = nn.DataParallel(model)

    # 将模型写入tensorboard
    # init_img = torch.zeros((4, 3, 256, 256)).cuda()
    # tb_writer.add_graph(model, init_img)

    # load data
    train_data = getattr(datasets, opt.dataset)(opt.root, opt.train_data_dir, mode='train')
    train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    valid_data = getattr(datasets, opt.dataset)(opt.root, opt.valid_data_dir, mode='test')
    valid_dataloader = DataLoader(valid_data, batch_size=1, shuffle=False, num_workers=opt.num_workers)
    val_total_batch = int(len(valid_data) / 1)

    test_data = getattr(datasets, opt.dataset)(opt.root, opt.test_data_dir, mode="test")
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=opt.num_workers)
    total_batch_test = int(len(test_data) / 1)

    # load optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.mt, weight_decay=opt.weight_decay)

    lr_lambda = lambda epoch: 1.0 - pow((epoch / opt.nEpoch), opt.power)
    scheduler = LambdaLR(optimizer, lr_lambda)

    # train
    print('Start training')
    print('---------------------------------\n')

    # 将模型名写入模型结果文件
    f = f"data_record/{opt.dataset}/{opt.test_text}"
    with open(f, "a") as filenametest:  # ”w"代表着每次运行都覆盖内容
        filenametest.write(
            "\n\n\n" + opt.model + "\n ----------  model:" + opt.model + ",  model save:" + opt.expName + ",  epochs:" + str(
                opt.nEpoch) +
            "  batch_size:" + str(opt.batch_size) + ",  time:" + x.strftime('%Y-%m-%d %H:%M:%S') + "  ----------\n")
    f = f"data_record/{opt.dataset}/{opt.val_text}"
    with open(f, "a") as filenameval:  # ”w"代表着每次运行都覆盖内容
        filenameval.write(
            "\n\n\n" + opt.model + "\n ----------  model:" + opt.model + ",  model save:" + opt.expName + ",  epochs:" + str(
                opt.nEpoch) +
            ",  batch_size:" + str(opt.batch_size) + ",  time:" + x.strftime('%Y-%m-%d %H:%M:%S') + "  ----------\n")

    for epoch in range(opt.nEpoch):
        print('------ Epoch ------', epoch + 1)
        model.train()
        total_batch = int(len(train_data) / opt.batch_size)
        bar = tqdm(enumerate(train_dataloader), total=total_batch)

        for i, data in bar:
            img = data['image']
            gt = data['label']

            if opt.use_gpu:
                img = img.cuda()
                gt = gt.cuda()

            optimizer.zero_grad()
            # output = model(img, flag='train')
            output = model(img)

            loss = DeepSupervisionLoss(output, gt)
            loss.backward()

            optimizer.step()
            bar.set_postfix_str('loss: %.5s' % loss.item())

        scheduler.step()

        # ---------------------------------------- Valid -------------------------------------------------
        metrics_result, val_loss = valid(model, valid_dataloader, val_total_batch)

        print("Valid Result:")
        print('Dice: %.4f, IoU_mean: %.4f, precision: %.4f, ACC_overall: %.4f, IoU_poly: %.4f,'
              'IoU_bg: %.4f, recall: %.4f, specificity: %.4f, F1: %.4f, F2: %.4f, val_loss: %.4f'
              % (metrics_result['Dice'], metrics_result['IoU_mean'],
                 metrics_result['precision'], metrics_result['ACC_overall'],
                 metrics_result['IoU_poly'], metrics_result['IoU_bg'],
                 metrics_result['recall'], metrics_result['specificity'],
                 metrics_result['F1'], metrics_result['F2'], val_loss))

        # add into tensorboard
        tags = ["val_precision", "val_IoU", "val_Dice", "val_loss", "test_precision", "test_IoU", "test_Dice", "test_loss"]
        tb_writer.add_scalar(tags[0], metrics_result['precision'], epoch + 1)
        tb_writer.add_scalar(tags[1], metrics_result['IoU_mean'], epoch + 1)
        tb_writer.add_scalar(tags[2], metrics_result['Dice'], epoch + 1)
        tb_writer.add_scalar(tags[3], val_loss, epoch + 1)

        # add result into txt
        f = f"data_record/{opt.dataset}/{opt.val_text}"
        with open(f, "a") as file:  # ”w"代表着每次运行都覆盖内容
            file.write('epoch: %.f,\tDice: %.4f,\tIoU_mean: %.4f,\tprecision: %.4f,\tACC_overall: %.4f,\t'
                       'IoU_poly: %.4f,\tIoU_bg: %.4f,\trecall: %.4f,\tspecificity: %.4f,\tF1: %.4f,\tF2: %.4f,val_loss:%.4f\n'
                       % (epoch + 1,
                          metrics_result['Dice'], metrics_result['IoU_mean'],
                          metrics_result['precision'], metrics_result['ACC_overall'],
                          metrics_result['IoU_poly'], metrics_result['IoU_bg'],
                          metrics_result['recall'], metrics_result['specificity'],
                          metrics_result['F1'], metrics_result['F2'], val_loss))

        if ((epoch + 1) % opt.ckpt_period == 0):
            torch.save(model.state_dict(),f"F:/checkpoints/{opt.dataset}/" + opt.expName + "/ck_{}.pth".format(epoch + 1))
        # ---------------------------------------- test -------------------------------------------------
        test_result, test_loss = test(model, test_dataloader, total_batch_test)
        print("Test Result:")
        print('Dice: %.4f, IoU_mean: %.4f, precision: %.4f, ACC_overall: %.4f, IoU_poly: %.4f,'
              'IoU_bg: %.4f, recall: %.4f, specificity: %.4f, F1: %.4f, F2: %.4f, test_loss: %.4f'
              % (test_result['Dice'], test_result['IoU_mean'],
                 test_result['precision'], test_result['ACC_overall'],
                 test_result['IoU_poly'], test_result['IoU_bg'],
                 test_result['recall'], test_result['specificity'],
                 test_result['F1'], test_result['F2'], test_loss))

        f = f"data_record/{opt.dataset}/{opt.test_text}"
        with open(f, "a") as file1:  # ”w"代表着每次运行都覆盖内容
            file1.write('epoch: %.f,\tDice: %.4f,\tIoU_mean: %.4f,\tprecision: %.4f,\tACC_overall: %.4f,\t'
                        'IoU_poly: %.4f,\tIoU_bg: %.4f,\trecall: %.4f,\tspecificity: %.4f,\tF1: %.4f,\tF2: %.4f ,test_loss:%.4f\n'
                        % (epoch + 1,
                           test_result['Dice'], test_result['IoU_mean'],
                           test_result['precision'], test_result['ACC_overall'],
                           test_result['IoU_poly'], test_result['IoU_bg'],
                           test_result['recall'], test_result['specificity'],
                           test_result['F1'], test_result['F2'], test_loss))

        # add into tensorboard
        tb_writer.add_scalar(tags[4], test_result['precision'], epoch + 1)
        tb_writer.add_scalar(tags[5], test_result['IoU_mean'], epoch + 1)
        tb_writer.add_scalar(tags[6], test_result['Dice'], epoch + 1)
        tb_writer.add_scalar(tags[7], test_loss, epoch + 1)


if __name__ == '__main__':

    seed_everything()
    if opt.mode == 'train':
        print('---Kavsir Train---')
        train()

    print('Done')

# tensorboard --logdir "D:\paperin\ResUNetPlusPlus-master\logs\20220326-165433\train" --bind_all
# localhost:6006