# def AND(x1, x2):
#     w1, w2, theta = 0.5, 0.5, 0.7
#     tmp = x1*w1 + x2*w2
#     if tmp <= theta:
#         return 0
#     elif tmp > theta:
#         return 1

# print(AND(0,0)) #0
# print(AND(1,0)) #0
# print(AND(0,1)) #0
# print(AND(1,1)) #1
            
            
#가중치와 편향 도입
import numpy as np

x = np.array([0,1]) #input
w = np.array([0.5, 0.5]) #weight
b = -0.7    #bias
print(w * x)
print(np.sum(w * x))
print(np.sum(w * x)+b) # 대략 -0.2

def AND(x1, x2):
    x = np.array([x1,x2])
    w = np.array([0.5, 0.5]) #weight
    b = -0.7    #bias
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1

print(AND(0,0)) #0
print(AND(1,0)) #0
print(AND(0,1)) #0
print(AND(1,1)) #1
print("")

def NAND(x1, x2):
    x = np.array([x1,x2])
    w = np.array([-0.5, -0.5]) #weight defferent here!
    b = -0.7    #bias
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1

def OR(x1, x2):
    x = np.array([x1,x2])
    w = np.array([0.5, 0.5]) #weight 
    b = -0.2    #bias defferent here!
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1

def XOR(x1,x2):
    s1 = NAND(x1,x2)
    s2 = AND(x1,x2)
    y = AND(s1,s2)
    return y

print(XOR(0,0)) #0
print(XOR(1,0)) #1
print(XOR(0,1)) #1
print(XOR(1,1)) #0




