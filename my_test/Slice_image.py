from PIL import Image
import sys
import os


def cut_image(image):
    width, height = image.size
    item_width = int(width /3)#修改这里的参数
    item_height=int(height / 3)#修改这里的参数
    box_list = []
    count = 0
    for j in range(0,3):#修改这里的参数
        for i in range(0, 3):#修改这里的参数
            count += 1
            box = (i * item_width, j * item_height, (i + 1) * item_width, (j + 1) * item_height)
            box_list.append(box)
    print(count)
    image_list = [image.crop(box) for box in box_list]
    return image_list


def save_images(image_list,image_name,save_path):
    index = 1
    for image in image_list:
        image.save(os.path.join(save_path,image_name.split(".")[0]+".jpg"))
        index += 1


if __name__ == '__main__':
    file_path = r"D:\datasets\poly_dataset\esophagus\esophagus_h\mask"
    save_path=r"D:\datasets\poly_dataset\esophagus\esophagus_h\masks"
    # 打开图像
    for image_name in os.listdir(file_path):
        print(image_name)
        image = Image.open(os.path.join(file_path,image_name))
        image_list = cut_image(image)
        save_images(image_list,image_name,save_path)
