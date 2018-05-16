import sys, os
sys.path.append(os.pardir)
print(os)
sys.path.append('c:/Users/bueno/Documents/Deeplearning/')
print(sys.path)
import numpy as np
import matplotlib.pylab as plt
from dataset.mnist import load_mnist

(x_train, t_train),(x_test,t_test) = \
load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape)
print(t_train.shape)

t_train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(t_train_size,batch_size)

def function_1(x):
    return 0.01*x**2 + 0.1*x

x = np.arange(0.0,20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.xlabel("f(x)")
plt.plot(x,y)
plt.show()
