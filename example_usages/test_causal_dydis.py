import sys
import os


import pandas as pd
from ylearn.exp_dataset.gen import dygen, gen, count_accuracy
from ylearn.causal_discovery import CausalDiscovery, GolemDiscovery, DagmaDiscovery, DyCausalDiscovery, DygolemCausalDiscovery

X = gen(n=500, d=10)
X = pd.DataFrame(X, columns=[f'x{i}' for i in range(X.shape[1])])
print("-------------------------notears model-------------------------")
cd = CausalDiscovery(hidden_layer_dim=[3])
est_dict = cd(X, threshold=0.01, return_dict=True)
print(est_dict)

print("-------------------------golem model-------------------------")
gocd = GolemDiscovery()
est_go_dict = gocd(X, threshold=0.01, return_dict=True)
print(est_go_dict)

print("-------------------------DAGMA model-------------------------")
dagmacd = DagmaDiscovery()
est_go_dict = dagmacd(X, threshold=0.01, return_dict=True)
print(est_go_dict)


X, W_true, P_true = dygen(n=500, d=5, step=10, order=1)
print(X.shape, W_true.shape, P_true.shape)

X = pd.DataFrame(X, columns=[f'x{i}' for i in range(X.shape[1])])
print("-------------------------Dynotears model-------------------------")
dycd = DyCausalDiscovery()
w_est_dy, p_est_dy = dycd(X, threshold=0.3, order=1, step=10, return_dict=True)
acc_dy = count_accuracy(W_true, w_est_dy != 0)
print(acc_dy)
for k in range(P_true.shape[2]):
    acc_dy = count_accuracy(P_true[:, :, k], p_est_dy[:, :, k] != 0)
    print(acc_dy)
print("-------------------------Dygolem model-------------------------")
dygo = DygolemCausalDiscovery()
w_est_dy, p_est_dy = dygo(X, threshold=0.3, order=1, step=10, return_dict=True)
acc_dy = count_accuracy(W_true, w_est_dy != 0)
print(acc_dy)
for k in range(P_true.shape[2]):
    acc_dy = count_accuracy(P_true[:, :, k], p_est_dy[:, :, k] != 0)
    print(acc_dy)