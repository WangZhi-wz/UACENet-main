from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img_path = "D:\datasets\poly_dataset\esophagus\esophagus_h\mask\I2309382.png"

# 转换前
img = Image.open(img_path)
print("转换前\n shape: {}\n {}".format(img.size, img))
ar = np.array(img)
print(ar.shape)
print(img)
plt.imshow(ar)
plt.show()

# 转换后
img = img.convert('RGB')
print("转换后\n shape: {}\n {}".format(img.size, img))
ar = np.array(img)
print(ar.shape)
print(img)
plt.imshow(ar)
plt.show()