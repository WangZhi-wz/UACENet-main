import io
import os

from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

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

        # load_ckpt_path = os.path.join(r'E:\checkpoints\ACSNet_DP\ck_1.pth')

        # load_ckpt_path = os.path.join('E:/checkpoints/'+opt.expName+'/', 'ck_' + str(opt.load_ckpt) + '.pth')
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
        load_ckpt_path = os.path.join('C:\\Users\\15059\\Desktop\\20220103\\model_data\\FACENet\\Kvasir-SEG\\ck_150.pth')
        # load_ckpt_path = os.path.join(r'C:\Users\15059\Desktop\best_model\Kvasir-SEG\8.21 ACSNet_caRABAsa_modDCR 0.9206 0.9171\ck_148.pth')


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

    layername = 'outconv'
    # 自己要放CAM的位置-----------------------------------------------------------------------------------------------
    target_layers = [model.outconv]

    folder_path2 = "result_featuremap" + "/" + layername + "/" + "predict"
    folder_path5 = "result_featuremap" + "/" + layername + "/" + "result_map_all"
    folder_path6 = "result_featuremap" + "/" + layername + "/" + "result_map"
    if not os.path.exists(folder_path2):
        os.makedirs(folder_path2)
    if not os.path.exists(folder_path5):
        os.makedirs(folder_path5)
    if not os.path.exists(folder_path6):
        os.makedirs(folder_path6)

    # with torch.no_grad():
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
        predict = torch.squeeze(output[0]).detach().cpu().numpy()    # 320,320
        predict[predict > 0.5] = 255
        predict[predict <= 0.5] = 0
        # 添加一个维度
        predict = np.expand_dims(predict, axis=2)
        # 一维转三维
        predict = np.squeeze(predict)
        predict = [predict, predict, predict]
        predict = np.transpose(predict, (1, 2, 0))

        # feature map
        normalized_masks = torch.sigmoid(output[0]).cpu()  # 1,1,320,320
        sem_classes = [
            '__background__', 'ployp'
        ]

        sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}
        round_category = sem_class_to_idx['__background__']
        round_mask = torch.argmax(normalized_masks[0], dim=0).detach().cpu().numpy()  # [320,320]
        # round_mask_uint8 = 255 * np.uint8(round_mask == round_category)
        round_mask_float = np.float32(round_mask == round_category)


        class SemanticSegmentationTarget:
            def __init__(self, category, mask):
                self.category = category
                self.mask = torch.from_numpy(mask)
                if torch.cuda.is_available():
                    self.mask = self.mask.cuda()

            def __call__(self, model_output):
                return (model_output[self.category, :, :] * self.mask).sum()



        targets = [SemanticSegmentationTarget(round_category, round_mask_float)]

        img1 = torch.squeeze(img).cpu().numpy()  # 3,320,320
        img1 = np.transpose(img1, [1, 2, 0])  # 320,320,3
        input_tensor = preprocess_image(img1,
                                        mean=[0.1, 0.1, 0.1],
                                        std=[0.9, 0.9, 0.9])
        # input_tensor = preprocess_image(img1,
        #                                 mean=[0.485, 0.456, 0.406],
        #                                 std=[0.229, 0.224, 0.225])

        with GradCAM(model=model, target_layers=target_layers,
                     use_cuda=torch.cuda.is_available()) as cam:
            grayscale_cam = cam(input_tensor=input_tensor,
                                targets=targets)[0, :]
            cam_image = show_cam_on_image(img1, grayscale_cam, use_rgb=True)


        # 保存CAM的结果
        img = Image.fromarray(cam_image)
        # img.show()
        img.save(folder_path6 + '/' + str(i) + '.png')

        # save all
        # img = np.float32(img) / 255
        sep_line = np.ones((320, 10, 3)) * 255
        all_imagess = [
            img_out * 255,
            sep_line, gt_out,
            sep_line, predict,
            sep_line, img,
        ]
        io.imsave(os.path.join(folder_path2, str(i) + ".jpg"), predict)
        io.imsave(os.path.join(folder_path5, str(i) + ".jpg"), np.concatenate(all_imagess, axis=1))


def test_run():



    test_data = getattr(datasets, opt.dataset)(opt.root, opt.test_data_dir, mode="test")
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=opt.num_workers)
    total_batch_test = int(len(test_data) / 1)

    model = generate_model(opt)

    model.eval()

    # ---------------------------------------- test -------------------------------------------------
    test(model, test_dataloader, total_batch_test)





if __name__ == '__main__':

    # if opt.mode == 'test':
    print('---Kavsir Train---')
    test_run()

    print('Done')