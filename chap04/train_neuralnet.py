import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet
import matplotlib.pyplot as plt

# mnistデータセットをロード
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# plot用
iter_list = []
epoch_list = []
train_loss_list = []
train_acc_list = []
test_acc_list = []

# ハイパーパラメータ
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

# 1エポックあたりの繰り返し数
iter_per_epoch = max(train_size/batch_size, 1)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

i_epoch = 0

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

    if i % iter_per_epoch == 0:
        i_epoch += 1
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        epoch_list.append(i_epoch)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

# 損失関数プロット
plt.plot(iter_list, train_loss_list)
plt.xlim(left=0, right=iters_num)
plt.ylim(bottom=0)
plt.xlabel("iteration")
plt.ylabel('loss')
plt.savefig('figure/loss-iter.png', dpi=300)
plt.show()

# 認識精度の推移
plt.plot(epoch_list, train_acc_list, label="train")
plt.plot(epoch_list, test_acc_list, label="test", ls="--")
plt.xlim(left=1, right=max(epoch_list))
plt.ylim(bottom=0)
plt.xlabel("epoch")
plt.ylabel('accuracy')
plt.savefig('figure/accuracy.png', dpi=300)
plt.show()