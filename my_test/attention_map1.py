import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def gray2color(gray_array, color_map):
    rows, cols = gray_array.shape
    color_array = np.zeros((rows, cols, 3), np.uint8)

    for i in range(0, rows):
        for j in range(0, cols):
            color_array[i, j] = color_map[gray_array[i, j]]

    # color_image = Image.fromarray(color_array)

    return color_array


jet_map = np.loadtxt('jet_int.txt', dtype=np.int)

img = Image.open('D:\\paperin\\Enhanced-U-Net-main\\aaa\\ACSNet_caRABAsaBD_modDCR\\old1.jpg').convert('L')
img = np.array(img)

img_color_jet = gray2color(img, jet_map)
# plt.subplot(212)


# attention = Image.open('attention_0.jpg').convert('L')
# attention = np.array(attention)
# for i in range(1, 8):
#     x = Image.open('attention_' + str(i) + '.jpg').convert('L')
#     x = np.array(x)
#     attention += x
# attention_color_jet = gray2color(attention, jet_map)
# plt.subplot(212)
print(img_color_jet.shape)
plt.imshow(img_color_jet)
# plt.imshow(attention_color_jet)
plt.imshow(img_color_jet, alpha=0.1)
# plt.imshow(attention_color_jet, alpha=0.5)
plt.show()

