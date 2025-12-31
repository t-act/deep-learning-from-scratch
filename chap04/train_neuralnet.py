import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet
import matplotlib.pyplot as plt

# mnistデータセットをロード
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# plot用
train_loss_list = []
iter_list = []

# ハイパーパラメータ
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    # ミニバッチの取得    
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 勾配の計算
    grad = network.gradient(x_batch, t_batch)

    # パラメータの更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate*grad[key]

    # 学習経過の記録
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    iter_list.append(i)

# プロット
plt.plot(iter_list, train_loss_list)
plt.xlim(left=0, right=iters_num)
plt.ylim(bottom=0)
plt.xlabel("iteration")
plt.ylabel('loss')
plt.savefig('figure/loss-iter.png', dpi=300)
plt.show()