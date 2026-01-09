import numpy as np
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
from common.funcs import *

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0 # x<=0を満たすときは0に置換

        return out
    
    def backward(self, dout):
        dout[self.mask] = 0 # 順伝播時に保持したmaskを再利用(引数xが必要ない)
        dx = dout

        return dx
    

class Sigmoid:
    def __init__(self):
        self.out = None
    
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out
    
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx
        
class Affine:
    def __init__(self, W, b):
        self.W =W
        self.b = b
        
        self.x = None
        self.original_x_shape = None
        # 重み・バイアスパラメータの微分
        self.dW = None
        self.db = None

    def forward(self, x):
        # テンソル対応
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        dx = dx.reshape(*self.original_x_shape)  # 入力データの形状に戻す（テンソル対応）
        return dx
    
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None # 損失
        self.y = None    # softmaxの出力
        self.t = None    # 教師データ(one-hot vector)

    def forward(self, x, t):
        self.t = t
        # use batched softmax implementation (handles 2D input / batches)
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx