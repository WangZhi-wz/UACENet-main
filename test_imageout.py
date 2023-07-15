import io
import os

from utils.loss_mod import DeepSupervisionLoss

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from tqdm import tqdm
from opt import opt
from utils.metrics import evaluate
import datasets
from torch.utils.data import DataLoader
from utils.comm import generate_model
from utils.metrics import Metrics
import skimage.io as io


def test():
    print('loading data......')
    test_data = getattr(datasets, opt.dataset)(opt.root, opt.test_data_dir, mode='test')
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=opt.num_workers)
    total_batch = int(len(test_data) / 1)
    model = generate_model(opt)

    model.eval()

    # metrics_logger initialization
    metrics = Metrics(['recall', 'specificity', 'precision', 'F1', 'F2',
                       'ACC_overall', 'IoU_poly', 'IoU_bg', 'IoU_mean', 'Dice'])

    with torch.no_grad():
        test_loss = 0
        bar = tqdm(enumerate(test_dataloader), total=total_batch)
        for i, data in bar:
            img, gt = data['image'], data['label']

            if opt.use_gpu:
                img = img.cuda()
                gt = gt.cuda()

            if opt.model == 'ACSNet_caRABAsaBD_modDCR':
                output = model(img, flag = 'test')
            else:
                output = model(img)

            predict = torch.squeeze(output[0]).cpu().numpy()

            predict[predict > 0.5] = 255
            predict[predict <= 0.5] = 0
            # print("___________________________________________")
            # print(predict)

            io.imsave(os.path.join("result", str(i) + ".jpg"), predict)

            _recall, _specificity, _precision, _F1, _F2, \
            _ACC_overall, _IoU_poly, _IoU_bg, _IoU_mean, _Dice = evaluate(output, gt, 0.5)

            metrics.update(recall=_recall, specificity=_specificity, precision=_precision,
                           F1=_F1, F2=_F2, ACC_overall=_ACC_overall, IoU_poly=_IoU_poly,
                           IoU_bg=_IoU_bg, IoU_mean=_IoU_mean, Dice=_Dice
                           )
            test_loss_all = test_loss + DeepSupervisionLoss(output, gt)
        test_loss_all_a = test_loss_all
        print("test_loss_all:%.4f" % test_loss_all_a)

    metrics_result = metrics.mean(total_batch)

    print("Test Result:")
    print('recall: %.4f, specificity: %.4f, precision: %.4f, F1: %.4f, F2: %.4f, '
          'ACC_overall: %.4f, IoU_poly: %.4f, IoU_bg: %.4f, IoU_mean: %.4f, IOU_Dice:%.4f'
          % (metrics_result['recall'], metrics_result['specificity'], metrics_result['precision'],
             metrics_result['F1'], metrics_result['F2'], metrics_result['ACC_overall'],
             metrics_result['IoU_poly'], metrics_result['IoU_bg'], metrics_result['IoU_mean'], metrics_result['Dice']))


if __name__ == '__main__':

    if opt.mode == 'test':
        print('--- kvasir-SEG Test---')
        test()

    print('Done')
