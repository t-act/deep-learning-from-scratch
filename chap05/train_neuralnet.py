import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet
import matplotlib.pyplot as plt

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

iter_list = []
epoch_list = []
train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

i_epoch = 0

for i in range(iters_num):
    iter_list.append(i)

    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network.gradient(x_batch, t_batch)

    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        i_epoch += 1
        epoch_list.append(i_epoch)
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

# 損失関数プロット
plt.plot(iter_list, train_loss_list)
plt.xlim(left=0, right=iters_num)
plt.ylim(bottom=0)
plt.xlabel("iteration")
plt.ylabel('loss')
plt.savefig('figure/loss-iter-005.png', dpi=300)
plt.show()

# 認識精度の推移
plt.plot(epoch_list, train_acc_list, label="train")
plt.plot(epoch_list, test_acc_list, label="test", ls="--")
plt.xlim(left=1, right=max(epoch_list))
plt.ylim(bottom=0)
plt.xlabel("epoch")
plt.ylabel('accuracy')
plt.savefig('figure/accuracy-005.png', dpi=300)
plt.show()