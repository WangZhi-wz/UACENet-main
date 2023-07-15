import argparse
import os
import random
import shutil
from PIL import Image

def main(opt):
    p = opt.percentage
    img_dir = opt.image_dir
    lab_dir = opt.label_dir
    res_img_dir = opt.res_img_dir
    res_lab_dir = opt.res_lab_dir
    res_img_remain = opt.res_img_remain
    res_lab_remain = opt.res_lab_remain

    # shutil.rmtree(res_img_dir) # 清空文件中内容
    # shutil.rmtree(res_lab_dir)
    # shutil.rmtree(res_img_remain)
    # shutil.rmtree(res_lab_remain)

    if not os.path.exists(res_img_dir): # 如果文件夹不存在，则创建文件夹
        os.makedirs(res_img_dir)

    if not os.path.exists(res_lab_dir):
        os.makedirs(res_lab_dir)

    if not os.path.exists(res_img_remain): # 如果文件夹不存在，则创建文件夹
        os.makedirs(res_img_remain)

    if not os.path.exists(res_lab_remain):
        os.makedirs(res_lab_remain)

    img_name_list = os.listdir(img_dir) # 获取图片名组成的列表
    random.shuffle(img_name_list) # 打乱列表
    idx = int((p / 100) * len(img_name_list)) # 按百分比获取需要提取的图片数量
    print(idx)
    ext_list = img_name_list[:idx]
    ext_list1 = img_name_list[idx:]

    for i, img in enumerate(ext_list):
        im = Image.open(img_dir + os.sep + img)
        im.save(res_img_dir + os.sep + img)
        name = img.split('.')[0] + '.png'
        if name in os.listdir(lab_dir):
            shutil.copy(lab_dir + os.sep + name + '', res_lab_dir)

    for i, img in enumerate(ext_list1):
        im = Image.open(img_dir + os.sep + img)
        im.save(res_img_remain + os.sep + img)
        name = img.split('.')[0] + '.png'
        if name in os.listdir(lab_dir):
            shutil.copy(lab_dir + os.sep + name, res_lab_remain)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--percentage', type=int, default=50, help='percentage of choose')
    parser.add_argument('--image_dir', type=str, default=r'D:\datasets\paper2_dataset\CVC-ColonDB\res\images', help='source of images')
    parser.add_argument('--label_dir', type=str, default=r'D:\datasets\paper2_dataset\CVC-ColonDB\res\masks', help='source of labels')
    parser.add_argument('--res_img_dir', type=str, default=r'D:\datasets\paper2_dataset\CVC-ColonDB\valid\images', help='output of images')
    parser.add_argument('--res_lab_dir', type=str, default=r'D:\datasets\paper2_dataset\CVC-ColonDB\valid\masks', help='output of labels')
    parser.add_argument('--res_img_remain', type=str, default=r'D:\datasets\paper2_dataset\CVC-ColonDB\test\images',help='output of labels')
    parser.add_argument('--res_lab_remain', type=str, default=r'D:\datasets\paper2_dataset\CVC-ColonDB\test\masks',help='output of labels')
    opt = parser.parse_args()
    main(opt)
