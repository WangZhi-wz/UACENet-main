import os
import os

from grad_cam.opt import opt

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import models
from torchvision import transforms
from utils import GradCAM, show_cam_on_image



def main(opt):
    # model = models.mobilenet_v3_large(pretrained=True)
    # target_layers = [model.features[-1]]

    model = getattr(models, opt.model)(opt.nclasses)
    # model = DDANet()

    if opt.use_gpu:
        model.cuda()
        torch.backends.cudnn.benchmark = True

    # if opt.load_ckpt is not None:
        model_dict = model.state_dict()

        load_ckpt_path = os.path.join('C:\\Users\\15059\\Desktop\\best_model\\ETIS-LaribPolypDB\\EUNet\\ck_142.pth')

        print(load_ckpt_path)
        assert os.path.isfile(load_ckpt_path), 'No checkpoint found.'
        print('Loading checkpoint......')
        checkpoint = torch.load(load_ckpt_path)
        new_dict = {k: v for k, v in checkpoint.items() if k in model_dict.keys()}
        model_dict.update(new_dict)
        model.load_state_dict(model_dict)

        target_layers = [model.outconv]

    # model = models.vgg16(pretrained=True)
    # target_layers = [model.features]

    # model = models.resnet34(pretrained=True)
    # target_layers = [model.layer4]

    # model = models.regnet_y_800mf(pretrained=True)
    # target_layers = [model.trunk_output]

    # model = models.efficientnet_b0(pretrained=True)
    # target_layers = [model.features]

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # load image
    img_path = "both.png"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)

    # [N, C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    target_category = 2  # tabby, tabby cat
    # target_category = 254  # pug, pug-dog

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
    plt.imshow(visualization)
    plt.show()


if __name__ == '__main__':
    main(opt)
