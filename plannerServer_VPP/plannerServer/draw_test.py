import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

for v in np.linspace(0-0.26,  0+0.26, 7):
    print("u,v = ",v)

# fig = plt.figure()
# ax = fig.add_subplot(111,projection='3d')
# ax.set_xlim3d(0, 0.8)
# ax.set_ylim3d(0, 0.8)
# ax.set_zlim3d(0, 0.8)
# ax.quiver(0, 0, 0, 1, 1, 1, length = 0.5, normalize = True)
# plt.show()

# import torch
# x = torch.tensor(1.0, requires_grad=True)
# y = torch.tensor(2.0, requires_grad=True)
# z = x**2+y
# z.backward()
# print(z, x.grad, y.grad)
# a = x + x.grad * 0.1
# print(a)


