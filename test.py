import datetime
import io
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
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


def test():
    model = getattr(models, opt.model)(opt.nclasses)
    model.cuda()
    torch.backends.cudnn.benchmark = True
    epochs = 200
    x = 0
    for i in range(epochs):
        print('loading data......')
        test_data = getattr(datasets, opt.dataset)(opt.root, opt.test_data_dir, mode='test')
        test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=opt.num_workers)
        total_batch = int(len(test_data) / 1)
        # model = generate_model(opt)

        # model = getattr(models, opt.model)(opt.nclasses)
        model_dict = model.state_dict()
        load_ckpt_path = os.path.join('F:/checkpoints/' + opt.dataset + '/' + opt.expName + '/', 'ck_' + str(i+1) + '.pth')
        print(load_ckpt_path)       
        checkpoint = torch.load(load_ckpt_path)
        new_dict = {k: v for k, v in checkpoint.items() if k in model_dict.keys()}
        model_dict.update(new_dict)
        model.load_state_dict(model_dict)



        model.eval()

        # metrics_logger initialization
        metrics = Metrics(['recall', 'specificity', 'precision', 'F1', 'F2',
                           'ACC_overall', 'IoU_poly', 'IoU_bg', 'IoU_mean', 'Dice'])

        with torch.no_grad():
            bar = tqdm(enumerate(test_dataloader), total=total_batch)
            for i, data in bar:
                img, gt = data['image'], data['label']

                if opt.use_gpu:
                    img = img.cuda()
                    gt = gt.cuda()

                if opt.model == 'ACSNet_caRABAsaBD_modDCR':
                    output = model(img, flag="test")
                else:
                    output = model(img)

                # predict = torch.squeeze(output[0]).cpu().numpy()
                #
                # predict[predict > 0.5] = 255
                # predict[predict <= 0.5] = 0
                # # print("___________________________________________")
                # # print(predict)
                #
                # io.imsave(os.path.join("result", str(i) + ".jpg"), predict)


                _recall, _specificity, _precision, _F1, _F2, \
                _ACC_overall, _IoU_poly, _IoU_bg, _IoU_mean, _Dice = evaluate(output, gt, 0.5)

                metrics.update(recall= _recall, specificity= _specificity, precision= _precision,
                                F1= _F1, F2= _F2, ACC_overall= _ACC_overall, IoU_poly= _IoU_poly,
                                IoU_bg= _IoU_bg, IoU_mean= _IoU_mean, Dice = _Dice
                            )

        metrics_result = metrics.mean(total_batch)

        print("Test Result:")
        print('Dice: %.4f, IoU_mean: %.4f, precision: %.4f, ACC_overall: %.4f, IoU_poly: %.4f,'
              'IoU_bg: %.4f, recall: %.4f, specificity: %.4f, F1: %.4f, F2: %.4f'
              % (metrics_result['Dice'], metrics_result['IoU_mean'],
                 metrics_result['precision'], metrics_result['ACC_overall'],
                 metrics_result['IoU_poly'], metrics_result['IoU_bg'],
                 metrics_result['recall'], metrics_result['specificity'],
                 metrics_result['F1'], metrics_result['F2']))

        f = f"checkpoints/test_test.txt"
        with open(f, "a") as file1:  # ”w"代表着每次运行都覆盖内容
            file1.write('epoch: %.f,\tDice: %.4f,\tIoU_mean: %.4f,\tprecision: %.4f,\tACC_overall: %.4f,\t'
                        'IoU_poly: %.4f,\tIoU_bg: %.4f,\trecall: %.4f,\tspecificity: %.4f,\tF1: %.4f,\tF2: %.4f\n'
                        % (x + 1,
                           metrics_result['Dice'], metrics_result['IoU_mean'],
                           metrics_result['precision'], metrics_result['ACC_overall'],
                           metrics_result['IoU_poly'], metrics_result['IoU_bg'],
                           metrics_result['recall'], metrics_result['specificity'],
                           metrics_result['F1'], metrics_result['F2']))
        x = x + 1


if __name__ == '__main__':


    # time
    x = datetime.datetime.now()

    print('--- kvasir-SEG Test---')
    f = f"checkpoints/test_test.txt"
    with open(f, "a") as filenametest:  # ”w"代表着每次运行都覆盖内容
        filenametest.write(
            "\n\n\n" + opt.model + "\n ----------  model:" + opt.model + ",  model save:" + opt.expName + ",  epochs:" + str(
                opt.nEpoch) +
            "  batch_size:" + str(opt.batch_size) + ",  time:" + x.strftime('%Y-%m-%d %H:%M:%S') + "  ----------\n")
    test()

print('Done')
