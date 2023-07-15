import numpy as np
from matplotlib import pyplot as plt

data_list = []
a = [0.9224,0.9112,0.9110,0.9128,0.9120,0.8524,0.9013,0.7362,0.5261]
# b = np.random.randn(10)
# c = np.random.randn(10)
# d = np.random.randn(10)
# e = np.random.randn(10)
# print(a, b, c, d, e)
data_list.append(a)
# data_list.append(b)
# data_list.append(c)
# data_list.append(d)
# data_list.append(e)
print(data_list)
fig = plt.figure()
plt.boxplot(data_list, notch=None, vert=None, meanline=None, showmeans=None, sym=None)
plt.xticks([data+1 for data in range(len(data_list))], ['a', 'b', 'c', 'd', 'e'])
plt.plot()
plt.show()
