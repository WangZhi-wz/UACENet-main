import cv2
from PIL import Image
import os


def image_resize(image_path, new_path):  # 统一图片尺寸
    print('============>>bmp彩色图片变成jpb二值图片')
    for img_name in os.listdir(image_path):
        img_path = image_path + "/" + img_name  # 获取该图片全称

        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        res, thresh = cv2.threshold(gray, 0, 255, 0)
        cv2.imwrite(new_path + "//" + img_name, thresh)
    print("end the processing!")


if __name__ == '__main__':
    print("ready for ::::::::  ")
    ori_path = r"C:\Users\15059\Desktop\GLas\masks"  # 输入图片的文件夹路径
    new_path = r"C:\Users\15059\Desktop\GLas\masks1"  # resize之后的文件夹路径
    image_resize(ori_path, new_path)