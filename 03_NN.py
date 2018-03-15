# activation fucntion implement
import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    return np.array(x > 0, dtype=np.int)

def sigmoid(x): #sigmoid -> S자 모양이란 뜻
    return 1 / (1+np.exp(-x))

def relu(x):
    return np.maximum(0,x)

x = np.arange(-5.0,5.0, 0.1)
# y = step_function(x)
y = sigmoid(x)
plt.plot(x,y)
plt.ylim(-0.1, 1.1) #y축의 범위 지정
# plt.show()

B = np.array([[1,2],[3,4],[5,6]])
print(B)
print(np.ndim(B))
print(B.shape) #행열 정보를 튜플로 반환

A = np.array([[1,2,],[3,4]])
B = np.array([[5,6,],[7,8]]) #행렬은 대문자로 표기하자
print(np.dot(A,B)) #내적을 점곱(dot product)

#신경망 내적
X = np.array([1,2])
W = np.array([[1,3,5],[2,4,6]])
Y = np.dot(X,W)
print(Y)