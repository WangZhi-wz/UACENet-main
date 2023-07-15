import torch



# tensor相加是直接将对应维度元素加到一起
from torch import tensor

# a = torch.ones([8, 4, 5, 6])
# b = torch.ones([1, 1, 5, 6])
# c = a+b
# print(a)
# print(b)
# print(c)
#
# # tensor相乘是直接将对应维度元素相乘
# a1 = torch.ones([8, 4, 5, 6])
# b1 = torch.ones([1, 1, 5, 6])
# c1 = a1 * b1
# print(a1)
# print(b1)
# print(c1)
#
# # cat拼接，按维度进行拼接 dim=0按行拼接 dim=1按列拼接
# a2 = torch.ones([1, 1, 5, 6])
# b2 = 2*torch.ones([1, 1, 5, 6])
# c2 = torch.cat([a2, b2])
# print(a2)
# print(b2)
# print(c2)
#
#
# a4 = torch.rand(2,2)
# a_sigmoid = torch.sigmoid(a4)
# b4 = - a4
# b_sigmoid = torch.sigmoid(b4)
# print(a4)
# print(a_sigmoid)
# print(b4)
# print(b_sigmoid)



a5 = tensor([[0.9, 0.5],[0.3, 0.6]])
b5 = tensor([[0.9, 0.1],[0.3, 0.6]])

ab51 = a5 + b5
ab52 = torch.cat([a5, b5])

score = torch.sigmoid(a5)
dist = torch.abs(score - 0.5)
att = 1 - (dist / 0.5)
att_x = a5 * att

atta = att_x + a5

print(a5)
print(b5)
c5 = a5 + b5
print(c5)
