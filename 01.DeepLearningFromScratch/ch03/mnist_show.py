# 터미널에서 python3 -m pip install Pillow
# https://blog.naver.com/mozzi_i/221192664548

import sys, os
sys.path.append(os.pardir)
os.chdir('/Users/bueno/Documents/Deeplearning/dataset') 
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train),(x_test,t_test) = load_mnist(flatten = True, normalize=False)

img = x_train[0]
label = t_train[0]
print(label) #5

print(img.shape) #(784,)
img = img.reshape(28,28) #원래 이미지 모양으로 변형
print(img.shape) #(28,28)

img_show(img)