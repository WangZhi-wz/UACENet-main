import os

# 这是你需要修改文件的路径地址
filePath = r"D:\datasets\poly_dataset\esophagus\esophagus_h\masks"


def update(filePath):
    # listdir：返回指定的文件夹包含的文件或文件夹的名字的列表
    files = os.listdir(filePath)
    for file in files:
        fileName = filePath + os.sep + file
        path1 = filePath
        # 运用递归;isdir：判断某一路径是否为目录
        if os.path.isdir(fileName):
            update(fileName)
            continue
        else:
            if file.endswith('.png'):
                test = file.replace(".png", ".jpg")
                print("修改前:" + path1 + os.sep + file)
                print("修改后:" + path1 + os.sep + test)
                os.renames(path1 + os.sep + file, path1 + os.sep + test)


if __name__ == '__main__':
    update(filePath)


