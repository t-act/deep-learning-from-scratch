import numpy as np

class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forwad(self, x, y):
        self.x = x
        self.y = y
        out = x * y

        return out

    def backward(self, dout):
        dx = dout * self.y # xとｙをひっくり返す
        dy = dout * self.x

        return dx, dy
    
class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y

        return out
    
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1

        return dx, dy
    
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
    
