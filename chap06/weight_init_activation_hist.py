import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.random.randn(1000, 100)

node_num = 100
hidden_layer_size = 5
activations = {}
weight_init = 1/np.sqrt(node_num)

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]
    
    w = np.random.randn(node_num, node_num) * weight_init

    z = np.dot(x, w)
    a = sigmoid(z)
    #a = np.tanh(z)
    activations[i] = a

# ヒストグラムを描画
for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1) + "-layer")
    if i != 0: plt.yticks([], [])
    # plt.xlim(0.1, 1)
    # plt.ylim(0, 7000)
    plt.hist(a.flatten(), 30, range=(0,1))
plt.savefig(f'figure/weight_init_hist_Xavie_tanh.png', dpi=300)
plt.show()