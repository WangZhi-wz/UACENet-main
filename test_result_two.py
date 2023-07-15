import io
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# from utils.comm import generate_model
import numpy as np

from models import DDANet

import torch
from tqdm import tqdm
from opt import opt
from utils.metrics import evaluate
import datasets
from torch.utils.data import DataLoader
# from utils.comm import generate_model
from utils.metrics import Metrics
import skimage.io as io
import models



def generate_model(opt):
    model = getattr(models, opt.model)(opt.nclasses)
    # model = DDANet()

    if opt.use_gpu:
        model.cuda()
        torch.backends.cudnn.benchmark = True

    # if opt.load_ckpt is not None:
        model_dict = model.state_dict()
        # load_ckpt_path = os.path.join('E:/checkpoints/'+opt.expName+'/', 'ck_' + str(opt.load_ckpt) + '.pth')

        load_ckpt_path = os.path.join(r'E:\checkpoints\ACSNet_DP\ck_3.pth')

        # load_ckpt_path = os.path.join('C:\\Users\\15059\\Desktop\\best_model\\Kvasir-SEG\\9.29 ACSNet_caRABAsaBD_modDCR DCR+gcmnonsa RA+LCA+CA 0.9237 0.9196'+'/', 'ck_164' + '.pth')
        # load_ckpt_path = os.path.join('C:\\Users\\15059\\Desktop\\best_model\\Kvasir-SEG-val--test-best\\9.30 ACSNet'+'/', 'ck_172' + '.pth')
        # load_ckpt_path = os.path.join('C:\\Users\\15059\\Desktop\\修改2023.01.03\\model_data\\FACENet\\Kvasir-SEG\\ck_149.pth')
        # load_ckpt_path = os.path.join('C:\\Users\\15059\\Desktop\\best_model\\Kvasir-SEG-val--test-best\\10.1 CCBANet'+'/', 'ck_165' + '.pth')
        # load_ckpt_path = os.path.join('C:\\Users\\15059\\Desktop\\best_model\\Kvasir-SEG-val--test-best\\9.30 UACANet'+'/', 'ck_174' + '.pth')
        # load_ckpt_path = os.path.join('C:\\Users\\15059\\Desktop\\best_model\\Kvasir-SEG-val--test-best\\10.1 EUNet'+'/', 'ck_94' + '.pth')
        # load_ckpt_path = os.path.join('C:\\Users\\15059\\Desktop\\best_model\\Kvasir-SEG-val--test-best\\10.2 PraNet'+'/', 'ck_166' + '.pth')
        # load_ckpt_path = os.path.join('C:\\Users\\15059\\Desktop\\best_model\\Kvasir-SEG-val--test-best\\10.2 DDANet'+'/', 'ck_198' + '.pth')
        # load_ckpt_path = os.path.join('C:\\Users\\15059\\Desktop\\best_model\\Kvasir-SEG-val--test-best\\10.2 Resunetpp'+'/', 'ck_192' + '.pth')
        # load_ckpt_path = os.path.join('C:\\Users\\15059\\Desktop\\best_model\\Kvasir-SEG-val--test-best\\10.19 UNetpp'+'/', 'ck_196' + '.pth')
        # load_ckpt_path = os.path.join('C:\\Users\\15059\\Desktop\\best_model\\Kvasir-SEG-val--test-best\\10.19 UNet'+'/', 'ck_199' + '.pth')
        # load_ckpt_path = os.path.join('C:\\Users\\15059\\Desktop\\model_data\\FACENet\\Kvasir-SEG\\ck_153.pth')

        # load_ckpt_path = os.path.join('C:\\Users\\15059\\Desktop\\best_model\\CVC_ClinicDB\\10.8 ACSNet_caRABAsaBD_modDCR DCR+gcmnonsa RA+LCA+CA'+'/', 'ck_192' + '.pth')
        # load_ckpt_path = os.path.join('C:\\Users\\15059\\Desktop\\best_model\\CVC_ClinicDB\\10.27 ACSNet_caRABAsaBD_modDCR ASMCA'+'/', 'ck_149' + '.pth')
        # load_ckpt_path = os.path.join('C:\\Users\\15059\\Desktop\\best_model\\CVC_ClinicDB\\10.7 UACANet 0.9208 0.9232'+'/', 'ck_47' + '.pth')
        # load_ckpt_path = os.path.join('C:\\Users\\15059\\Desktop\\best_model\\CVC_ClinicDB\\10.7 CCBANet 0.9328 0.9357'+'/', 'ck_162' + '.pth')
        # load_ckpt_path = os.path.join('C:\\Users\\15059\\Desktop\\best_model\\CVC_ClinicDB\\10.6 ACSNet 0.9326 0.9356'+'/', 'ck_144' + '.pth')
        # load_ckpt_path = os.path.join('C:\\Users\\15059\\Desktop\\best_model\\CVC_ClinicDB\\10.17 EUNet 0.9284 0.9333'+'/', 'ck_135' + '.pth')
        # load_ckpt_path = os.path.join('C:\\Users\\15059\\Desktop\\best_model\\CVC_ClinicDB\\10.18 PraNet'+'/', 'ck_196' + '.pth')
        # load_ckpt_path = os.path.join('C:\\Users\\15059\\Desktop\\best_model\\CVC_ClinicDB\\10.18 ResUnetpp'+'/', 'ck_193' + '.pth')
        # load_ckpt_path = os.path.join('C:\\Users\\15059\\Desktop\\best_model\\CVC_ClinicDB\\10.19 UNet'+'/', 'ck_199' + '.pth')
        # load_ckpt_path = os.path.join('C:\\Users\\15059\\Desktop\\best_model\\CVC_ClinicDB\\10.19 UNetpp'+'/','ck_200' + '.pth')
        # load_ckpt_path = os.path.join('C:\\Users\\15059\\Desktop\\best_model\\CVC_ClinicDB\\10.28 DDANet' + '/','ck_198' + '.pth')
        # load_ckpt_path = os.path.join('C:\\Users\\15059\\Desktop\\model_data\\FACENet\\CVC_ClinicDB\\ck_159.pth')

        # load_ckpt_path = os.path.join('C:\\Users\\15059\\Desktop\\best_model\\ETIS-LaribPolypDB\\FACENet\\ck_187.pth')
        # load_ckpt_path = os.path.join('C:\\Users\\15059\\Desktop\\best_model\\ETIS-LaribPolypDB\\UACANet\\ck_154.pth')
        # load_ckpt_path = os.path.join('C:\\Users\\15059\\Desktop\\best_model\\ETIS-LaribPolypDB\\CCBANet\\ck_132.pth')
        # load_ckpt_path = os.path.join('C:\\Users\\15059\\Desktop\\best_model\\ETIS-LaribPolypDB\\ACSNet\\ck_139.pth')
        # load_ckpt_path = os.path.join('C:\\Users\\15059\\Desktop\\best_model\\ETIS-LaribPolypDB\\PraNet\\ck_100.pth')
        # load_ckpt_path = os.path.join('C:\\Users\\15059\\Desktop\\best_model\\ETIS-LaribPolypDB\\EUNet\\ck_142.pth')

        # load_ckpt_path = os.path.join('C:\\Users\\15059\\Desktop\\best_model\\ETIS-LaribPolypDB\\DDANet\\ck_64.pth')
        # load_ckpt_path = os.path.join('C:\\Users\\15059\\Desktop\\best_model\\ETIS-LaribPolypDB\\UNetpp\\ck_152.pth')
        # load_ckpt_path = os.path.join('C:\\Users\\15059\\Desktop\\best_model\\ETIS-LaribPolypDB\\UNet\\ck_142.pth')

        # load_ckpt_path = os.path.join('C:\\Users\\15059\\Desktop\\best_model\ETIS-LaribPolypDB\\ResUnetpp\\ck_197.pth')



        # load_ckpt_path = os.path.join('C:\\Users\\15059\\Desktop\\best_model\\piccolo\\9.7 ACSNet_caRABAsaBD_modDCR - piccolo 0.8460 0.8757'+'/', 'ck_161' + '.pth')
        # load_ckpt_path = os.path.join('C:\\Users\\15059\\Desktop\\best_model\\piccolo\\9.8 ACSNet - piccolo 0.8261 0.8670'+'/', 'ck_158' + '.pth')
        # load_ckpt_path = os.path.join('C:\\Users\\15059\\Desktop\\best_model\\piccolo\\9.11 UACANet - piccolo 0.8245 0.8594'+'/', 'ck_57' + '.pth')
        # load_ckpt_path = os.path.join('C:\\Users\\15059\\Desktop\\best_model\\piccolo\\9.12 CCBANet - piccolo 0.8438 0.8747'+'/', 'ck_86' + '.pth')
        # load_ckpt_path = os.path.join('C:\\Users\\15059\\Desktop\\best_model\\piccolo\\UNet\\ck_1.pth')
        # load_ckpt_path = os.path.join('C:\\Users\\15059\\Desktop\\best_model\\piccolo\\UNetpp\\ck_2.pth')
        # load_ckpt_path = os.path.join('C:\\Users\\15059\\Desktop\\best_model\\piccolo\\Resunetpp\\ck_9.pth')
        # load_ckpt_path = os.path.join('C:\\Users\\15059\\Desktop\\best_model\\piccolo\\9.9 PraNet - piccolo 0.8108 0.8509\\ck_168.pth')
        # load_ckpt_path = os.path.join('C:\\Users\\15059\\Desktop\\best_model\\piccolo\\9.8 ACSNet - piccolo 0.8261 0.8670\\ck_158.pth')
        # load_ckpt_path = os.path.join('C:\\Users\\15059\\Desktop\\best_model\\piccolo\\DDANet\\ck_16.pth')
        # load_ckpt_path = os.path.join('C:\\Users\\15059\\Desktop\\best_model\\piccolo\\EUNet\\ck_28.pth')
        # load_ckpt_path = os.path.join('C:\\Users\\15059\\Desktop\\best_model\\piccolo\\9.12 CCBANet - piccolo 0.8438 0.8747\\ck_86.pth')
        # load_ckpt_path = os.path.join('C:\\Users\\15059\\Desktop\\best_model\\piccolo\\9.11 UACANet - piccolo 0.8245 0.8594\\ck_57.pth')
        # load_ckpt_path = os.path.join('C:\\Users\\15059\\Desktop\\best_model\\piccolo\\MSRFNet\\ck_43.pth')
        # load_ckpt_path = os.path.join('C:\\Users\\15059\\Desktop\\best_model\\piccolo\\FACENet\\ck_45.pth')

        # load_ckpt_path = os.path.join('F:\\checkpoints\\CVC_ClinicDB\\ACSNet_caRABAsaBD_modDCR'+'/', 'ck_6' + '.pth')
        # load_ckpt_path = os.path.join('F:\\checkpoints\\kvasir_SEG\\ACSNet_caRABAsaBD_modDCR ASMCA'+'/', 'ck_175' + '.pth')
        # load_ckpt_path = os.path.join('F:\\checkpoints\\esophagus\\ACSNet'+'/', 'ck_145' + '.pth')





        # load_ckpt_path = os.path.join('./checkpoints/exp-cvcframe/', 'ck_'+ str(opt.load_ckpt) + '.pth')
        print(load_ckpt_path)
        assert os.path.isfile(load_ckpt_path), 'No checkpoint found.'
        print('Loading checkpoint......')
        checkpoint = torch.load(load_ckpt_path)
        new_dict = {k: v for k, v in checkpoint.items() if k in model_dict.keys()}
        model_dict.update(new_dict)
        model.load_state_dict(model_dict)

        print('Done')

    return model



def test(model, test_dataloader, total_batch):


    model.eval()

    # Metrics_logger initialization
    metrics = Metrics(['recall', 'specificity', 'precision', 'F1', 'F2',
                       'ACC_overall', 'IoU_poly', 'IoU_bg', 'IoU_mean', 'Dice'])

    folder_path1 = "result1" + "/" + opt.model + "/" + opt.dataset + "/" + "result"
    folder_path2 = "result1" + "/" + opt.model + "/" + opt.dataset + "/" + "only_predict"
    folder_path3 = "result1" + "/" + opt.model + "/" + opt.dataset + "/" + "only_mask"
    folder_path4 = "result1" + "/" + opt.model + "/" + opt.dataset + "/" + "only_image"
    if not os.path.exists(folder_path1):
        os.makedirs(folder_path1)
    if not os.path.exists(folder_path2):
        os.makedirs(folder_path2)
    if not os.path.exists(folder_path3):
        os.makedirs(folder_path3)
    if not os.path.exists(folder_path4):
        os.makedirs(folder_path4)

    with torch.no_grad():
        bar = tqdm(enumerate(test_dataloader), total=total_batch)
        # val_loss = 0
        for i, data in bar:
            img, gt = data['image'], data['label']

            # 输出结果使用代码
            img_out = torch.squeeze(img[0]).cpu().numpy()  # 3,320,320
            img_out = np.transpose(img_out, [1,2,0])  # 320,320,3

            gt_out = torch.squeeze(gt[0]).cpu().numpy()  # 320,320
            gt_out[gt_out > 0.5] = 255
            gt_out[gt_out <= 0.5] = 0
            # 添加一个维度
            gt_out = np.expand_dims(gt_out, axis=2)
            # 一维转三维
            gt_out = np.squeeze(gt_out)
            gt_out = [gt_out, gt_out, gt_out]
            gt_out = np.transpose(gt_out, (1, 2, 0))

            if opt.use_gpu:
                img = img.cuda()
                gt = gt.cuda()

            if opt.model == 'ACSNet_caRABAsaBD_modDCR':
                output = model(img, flag = 'test')
            else:
                output = model(img)
            print(output)

            # 输出图片代码 原图+标签+预测
            predict = torch.squeeze(output[0]).cpu().numpy()    # 320,320
            predict[predict > 0.5] = 255
            predict[predict <= 0.5] = 0
            # 添加一个维度
            predict = np.expand_dims(predict, axis=2)
            # 一维转三维
            predict = np.squeeze(predict)
            predict = [predict, predict, predict]
            predict = np.transpose(predict, (1, 2, 0))
            # io.imsave(os.path.join("./result/" + opt.model , str(i) + "_pre.jpg"), predict)

            sep_line = np.ones((320, 10, 3)) * 255
            all_images = [
                img_out*255,
                sep_line, gt_out,
                sep_line, predict,
            ]

            io.imsave(os.path.join(folder_path1, str(i) + ".jpg"), np.concatenate(all_images, axis=1))
            # 只输出结果图
            # predict1 = torch.squeeze(output[0]).cpu().numpy()
            # predict1[predict1 > 0.5] = 255
            # predict1[predict1 <= 0.5] = 0
            # print("___________________________________________")
            # print(predict)
            io.imsave(os.path.join(folder_path2, str(i) + ".jpg"), predict)
            io.imsave(os.path.join(folder_path3, str(i) + ".jpg"), gt_out)
            io.imsave(os.path.join(folder_path4, str(i) + ".jpg"), img_out*255)


            _recall, _specificity, _precision, _F1, _F2, \
            _ACC_overall, _IoU_poly, _IoU_bg, _IoU_mean, _Dice = evaluate(output, gt, 0.5)

            metrics.update(recall= _recall, specificity= _specificity, precision= _precision,
                            F1= _F1, F2= _F2, ACC_overall= _ACC_overall, IoU_poly= _IoU_poly,
                            IoU_bg= _IoU_bg, IoU_mean= _IoU_mean, Dice= _Dice,
                        )

    test_result = metrics.mean(total_batch)

    return test_result



def test_run():



    test_data = getattr(datasets, opt.dataset)(opt.root, opt.test_data_dir, mode="test")
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=opt.num_workers)
    total_batch_test = int(len(test_data) / 1)

    model = generate_model(opt)

    model.eval()

    # ---------------------------------------- test -------------------------------------------------
    test_result = test(model, test_dataloader, total_batch_test)
    print("Test Result:")
    print('Dice: %.4f, IoU_mean: %.4f, precision: %.4f, ACC_overall: %.4f, IoU_poly: %.4f,'
          'IoU_bg: %.4f, recall: %.4f, specificity: %.4f, F1: %.4f, F2: %.4f'
          % (test_result['Dice'], test_result['IoU_mean'],
             test_result['precision'], test_result['ACC_overall'],
             test_result['IoU_poly'], test_result['IoU_bg'],
             test_result['recall'], test_result['specificity'],
             test_result['F1'], test_result['F2']))

    f = f"checkpoints/{opt.test_text}"
    with open(f, "a") as file1:  # ”w"代表着每次运行都覆盖内容
        file1.write(opt.dataset + " - " + opt.model + "\n" + 'Dice: %.4f,\tIoU_mean: %.4f,\tprecision: %.4f,\tACC_overall: %.4f,\t'
                    'IoU_poly: %.4f,\tIoU_bg: %.4f,\trecall: %.4f,\tspecificity: %.4f,\tF1: %.4f,\tF2: %.4f\n'
                    % (
                       test_result['Dice'], test_result['IoU_mean'],
                       test_result['precision'], test_result['ACC_overall'],
                       test_result['IoU_poly'], test_result['IoU_bg'],
                       test_result['recall'], test_result['specificity'],
                       test_result['F1'], test_result['F2']))

if __name__ == '__main__':

    # if opt.mode == 'test':
    print('---Kavsir Train---')
    test_run()

    print('Done')