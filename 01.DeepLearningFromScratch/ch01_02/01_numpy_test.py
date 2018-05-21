import numpy as np
x = np.array([-1.0,2.0, -3.0])
y = np.array([2.0,4.0, 6.0])
z = x + y # 원소 덧셈
print(z)
z = x - y
print(z)
z = x * y
print(z)
z = x / y
print(z)
z = x / 2
print(z)

# class leejung:
#     def __init__(self):
#         self.mask = (x <= 0)

k = np.array([[0,0,0],[10,11,12]])
print(k + x)

mask = (x<=0)
print("mask {} ".format(mask))


# A =  np.array([[1,2], [3,4]])
# print(A)
# print(A.shape)
# print(A.dtype)

# B =  np.array([[3,0], [0,6]])
# print(A + B)
# print(A * B) #행렬곱
# print(A * 10) #스칼라 산술 -> 브로드캐스트

# # 1차원 배열 : vector, 2차원 배열 : metrix(행렬), 3차원은 다차원 배열
# # 텐서(tensor) :벡터, 행열 일반화

# A =  np.array([[1,2], [3,4]])
# B =  np.array([10,20])
# print(A * B) #브로드캐스트로 연산함.

X = np.array([[51,55],[14,19],[0,4]])
print(X)
print(X[0])
print(X[0][1]) #(0,1) 위치의 원소

for row in X:
    print(row)

X = X.flatten() #X를 1차원 배열로 변환(평탄화)
print(X)
print(X[np.array([0,2,4])]) #인덱스가 0,2,4인 원소 얻기.

print(X>15)  #넘파일 배열에 부등호 연산을하면 배열의 원소 각각에 부등호 연산을 수행한 bool 배열이 생성된다....
print(X[X>15])

# matplotlib 그래프 그리기
import matplotlib.pyplot as plt

#데이터 준비
x = np.arange(0,6,0.1) #0~6까지 0.1간격 생성
y1 = np.sin(x)
y2 = np.cos(x)


# #그래프 그리기
# plt.plot(x,y1)
# plt.show()

# #그래프 그리기 심화
# plt.plot(x,y1, label="sin")
# plt.plot(x,y2, linestyle="--", label="cos")
# plt.xlabel("x axis") #X 축이름
# plt.xlabel("y axis") #Y 축이름
# plt.title('sin & con') # 제목
# plt.legend()
# plt.show()

#이미지 읽어서 표시
from matplotlib.image import imread

img = imread('C:/Users/bueno/Documents/Deeplearning/01.DeepLearningFromScratch/image/roun.png') #이미지 읽어오기

plt.imshow(img)
plt.show()



# 튜플이 매개변수인 함수
def func_print_tuple(*arg):
 print( arg )

# 딕션너리가 매개변수인 함수
def func_print_dict(**arg):
 print( arg )

#위의 함수들은 다음과 같이 호출할 수 있습니다.
func_print_tuple( 1, 2, 3 )
# -> ( 1, 2, 3 )
func_print_dict( a=1, b=2, c=3 )
# -> {'a': 1, 'b': 2, 'c': 3}