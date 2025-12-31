import numpy as np

def numerical_gradient(f, x) -> np.array:
    '''
    偏微分を中心差分法で実装
    
    :param f: 被微分関数
    :param x: 変数
    '''
    h = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]

        x[idx] = tmp_val + h
        fxh1 = f(x) + h # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x) - h  # f(x-h)

        grad[idx] = (fxh1 - fxh2) / (2*h) # 中心差分法

        x[idx] = tmp_val
        it.iternext()

    return grad

