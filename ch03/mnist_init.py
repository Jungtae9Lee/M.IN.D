import sys, os
sys.path.append(os.pardir) #부모 디렉터리의 파일을 가져올 수 있도록 설정
os.chdir('/Users/bueno/Documents/Deeplearning/dataset') 
print(os.getcwd())
from dataset.mnist import load_mnist

#처음 한번만 오래걸림.
(x_train, t_train) , (x_test,t_test) = load_mnist(flatten=True,normalize=False)

print(x_train.shape)