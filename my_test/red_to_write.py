import numpy as np
import os
from PIL import Image

np.set_printoptions(threshold=np.inf)


# 这个函数用于将红色标签图转为白的的标签（其实红色的标签表示灰度值为1(也是只有一个通道）），但不知道为何会显示出红色
def RedToWhite(img_dir, new_img_dir):
    folders = os.listdir(img_dir)  # 得img_dir中所有文件的名字

    for floder in folders:
        image_path = os.path.join(img_dir, floder)
        img = Image.open(image_path)  # 打开图片
        img = img.convert('RGB')
        img = img.convert('P')
        print(img)
        newImg = np.array(img) * 255  # 红色的标签表示灰度值为1,乘以255后都变为255
        newImg = newImg.astype(np.uint8)
        newImg = Image.fromarray(newImg)
        newImg_path = os.path.join(new_img_dir, floder)
        newImg.save(newImg_path)


if __name__ == '__main__':
    img_path = r'D:\datasets\poly_dataset\esophagus\esophagus_h\mask'
    newImg_path = r'D:\datasets\poly_dataset\esophagus\esophagus_h\masks'
    RedToWhite(img_path, newImg_path)
# ![labelme生成的红黑图片(https://img-blog.csdnimg.cn/11732e32d6084a3996bc9bc344768dd0.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVyaXJp,size_13,color_FFFFFF,t_70,g_se,x_16#pic_center)
