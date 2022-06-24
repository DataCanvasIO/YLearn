import torch
import numpy as np


def func(dat, pho=10, alpha=1, lambda1=1):
    cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(dat, device=cuda).float()
    n = x.size()[0]
    batch_num = x.size()[1]
    w = torch.rand(n, n, device=cuda, requires_grad=True)

    def h(w):
        return torch.trace(torch.matrix_exp(w * w)) / n - 1

    def linear_loss(w):
        res = x - torch.matmul(w, x)
        return torch.sum(res * res) / (n * batch_num)

    def tot_loss(w, pho, alpha):
        return linear_loss(w) + h(w) * h(w) * 0.5 * pho + alpha * h(w) + lambda1 * torch.sum(w * w) / (n * n)

    optimizer = torch.optim.SGD([w], lr=0.1, momentum=0.9)

    def local_minimize(pho, alpha):
        for i in range(200):
            optimizer.zero_grad()
            l = tot_loss(w, pho, alpha)
            l.backward()
            optimizer.step()
            # print(i,w)

    h_ = h(w.clone().detach())
    for _ in range(100):
        local_minimize(pho, alpha)
        alpha = alpha + pho * h(w.clone().detach())
        print(w)

    return w.detach().cpu().numpy()


def reg(x, y):
    cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(x, device=cuda).float()
    y = torch.tensor(y, device=cuda).float()
    size = x.size()[0]
    batch_num = x.size()[1]
    w = torch.ones([size], device=cuda, requires_grad=True).float()
    wt = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], device=cuda).float()
    optimizer = torch.optim.SGD([w], lr=0.01, momentum=0.9)

    def l(w, x, y):
        res = y - torch.matmul(w, x)
        loss = torch.sum(res * res) / (size * batch_num)
        return loss

    for i in range(10000):
        optimizer.zero_grad()
        loss = l(w, x, y)
        loss.backward()
        optimizer.step()
        # w=torch.tensor(w-0.1*w.grad,device=cuda,requires_grad=True).float()

        # w = torch.tensor(w - 0.5 * w.grad,requires_grad=True).float()
        # print(w)
        print(loss)
    print(w)


def dat_gen():
    x1 = np.random.random_sample(20000)
    x2 = x1 + 0.001 * np.random.random_sample(20000)
    x3 = x2 - 0.001 * np.random.random_sample(20000)
    return np.array([x2, x1, x3])


# def dat_gen1(n, coef12=1, coef32=2, eps=1e-3, change_order=False):
#     x2 = np.random.normal(size=(n, ))
#     x1 = eps * np.random.normal(size=(n, )) + coef12 * x2
#     x3 = eps * np.random.normal(size=(n, )) + coef32 * x2
#     if change_order:
#         return np.array([x1, x2, x3])
#     else:
#         return np.array([x2, x1, x3])


if __name__ == 'main':
    dat = dat_gen()
    print(f'chain result {func(dat)}')

# dat = dat_gen1()
# print(f'folk result {func(dat)}')

# dat = dat_gen1()
# print(f'folk new order {func(dat)}')