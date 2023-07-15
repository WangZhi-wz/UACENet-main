import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
img = Image.open(r'C:\Users\15059\Desktop\esophagus\wz\images\I2369047.jpg')
# 亮度设置为2
transform_1 = transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2)
img_1 = transform_1(img)
transform_2 = transforms.RandomRotation(90)
img_2 = transform_2(img)

# plt.imshow(img_1)
plt.imshow(img_2)

plt.show()