from PIL import Image
import os


def image_resize(image_path, new_path):  # 统一图片尺寸
    print('============>>修改图片尺寸')
    for img_name in os.listdir(image_path):
        img_path = image_path + "/" + img_name  # 获取该图片全称
        print(img_path)
        image = Image.open(img_path)  # 打开特定一张图片
        image = image.resize((352, 352))  # 设置需要转换的图片大小
        image.save(new_path + '/' + img_name)
    print("end the processing!")


if __name__ == '__main__':
    print("ready for ::::::::  ")
    ori_path = r'C:\Users\15059\Desktop\Paper2model\CVC_ClinicDB-image-result\TransUNet'  # 输入图片的文件夹路径
    new_path = r'C:\Users\15059\Desktop\Paper2model\CVC_ClinicDB-image-result\TransUNet'  # resize之后的文件夹路径
    image_resize(ori_path, new_path)