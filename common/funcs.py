import numpy as np

def sigmoid(x):
    """シグモイド関数
    本の実装ではオーバーフローしてしまうため、以下のサイトを参考に修正。
    http://www.kamishima.net/mlmpyja/lr/sigmoid.html

    Args:
        x (numpy.ndarray): 入力
    
    Returns:
        numpy.ndarray: 出力
    """
    # xをオーバーフローしない範囲に補正
    sigmoid_range = 34.538776394910684
    x = np.clip(x, -sigmoid_range, sigmoid_range)

    # シグモイド関数
    return 1 / (1 + np.exp(-x))

def identity_function(x):
    return x

# def softmax(x):
#     c = np.max(x)
#     exp_x = np.exp(x - c) #オーバーフロー対策
#     sum_exp_x = np.sum(exp_x)
#     return exp_x / sum_exp_x

def softmax(x):
    """ソフトマックス関数
    
    Args:
        x (numpy.ndarray): 入力
    
    Returns:
        numpy.ndarray: 出力
    """
    # バッチ処理の場合xは(バッチの数, 10)の2次元配列になる。
    # この場合、ブロードキャストを使ってうまく画像ごとに計算する必要がある。
    # ここでは1次元でも2次元でも共通化できるようnp.max()やnp.sum()はaxis=-1で算出し、
    # そのままブロードキャストできるようkeepdims=Trueで次元を維持する。
    c = np.max(x, axis=-1, keepdims=True)
    exp_a = np.exp(x - c)  # オーバーフロー対策
    sum_exp_a = np.sum(exp_a, axis=-1, keepdims=True)
    y = exp_a / sum_exp_a
    return y

def cross_entropy_error(y, t):
    '''
    交差エントロピー誤差
    4.2.2 P89
    '''
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1) # 正解ラベルのインデックスを取得
    
    batch_size = y.shape[0]
    # 1e-7は発散対策
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

def sigmoid_grad(x):
    """5章で学ぶ関数。誤差逆伝播法を使う際に必要。
    """
    return (1.0 - sigmoid(x)) * sigmoid(x)