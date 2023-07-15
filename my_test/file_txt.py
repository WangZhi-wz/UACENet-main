import os
# readInfo函数，根据文件夹路径读取文件夹下所有文件名
def readInfo():
    filePath = 'D:\\paperin\\a_new_road\\FANet-main\\Kvasir-SEG\\images\\'
    name = os.listdir(filePath)         # os.listdir方法返回一个列表对象
    return name

# 程序入口
if __name__ == "__main__":
    fileList = readInfo()       # 读取文件夹下所有的文件名，返回一个列表
    print(fileList)
    file = open('train.txt', 'w')   # 创建文件，权限为写入
    for i in fileList:
        imageDir = i.split(".")[0]
        # annotationDir = 'images/' + i
        rowInfo = imageDir + '\n'
        print(rowInfo)
        file.write(rowInfo)
