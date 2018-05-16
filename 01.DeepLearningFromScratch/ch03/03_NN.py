# activation fucntion implement
import numpy as np
import matplotlib.pylab as plt


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):  # sigmoid -> S자 모양이란 뜻
    return 1 / (1+np.exp(-x))


def relu(x):
    return np.maximum(0, x)


x = np.arange(-5.0, 5.0, 0.1)
# y = step_function(x)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)  # y축의 범위 지정
# plt.show()

B = np.array([[1, 2], [3, 4], [5, 6]])
print(B)
print(np.ndim(B))
print(B.shape)  # 행열 정보를 튜플로 반환

A = np.array([[1, 2, ], [3, 4]])
B = np.array([[5, 6, ], [7, 8]])  # 행렬은 대문자로 표기하자
print(np.dot(A, B))  # 내적을 점곱(dot product)

# 신경망 내적
X = np.array([1, 2])
W = np.array([[1, 3, 5], [2, 4, 6]])
Y = np.dot(X, W)
print(Y)

# 3.4.2
X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

A1 = np.dot(X, W1) + B1
Z1 = sigmoid(A1)
print(Z1)

W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)
print(Z2)

# 3.4.3 구현 정리
def identity_function(x):
    return x

def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)

    return y

network = init_network()
x = np.array([1.0,0.5])
y = forward(network, x)
print(y) # [0.3168 0.6962]

#3.5 출력층 설계하기
a = np.array([0.3,2.9,4.0])

exp_a = np.exp(a) #지수함수
print(exp_a)
sum_exp_a = np.sum(exp_a)
print(sum_exp_a)
y = exp_a / sum_exp_a
print(y)

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c) #prevent overflow
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

a = np.array([0.3,2.9,4.0])
y = softmax(a)
print(y)
print(np.sum(y))
