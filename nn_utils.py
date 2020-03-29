import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import datetime
import time
import pandas as pd


class AE_basic(nn.Module):
    def __init__(self, nodes, no_last_bias=False):
        super(AE_basic, self).__init__()
        n_layers = len(nodes)
        ins_n_outs = []
        en_modulelist = nn.ModuleList()
        de_modulelist = nn.ModuleList()
        for ii in range(n_layers // 2):
            ins = nodes[ii]
            outs = nodes[ii + 1]
            ins_n_outs.append((ins, outs))
            en_modulelist.append(nn.Linear(ins, outs))
            en_modulelist.append(nn.Tanh())
        for ii in range(n_layers // 2):
            ii += n_layers // 2
            ins = nodes[ii]
            outs = nodes[ii + 1]
            de_modulelist.append(nn.Linear(ins, outs))
            de_modulelist.append(nn.Tanh())

        de_modulelist = de_modulelist[:-1]  # Remove Tanh activation from output layer
        if no_last_bias:
            de_modulelist = de_modulelist[:-1]
            de_modulelist.append(nn.Linear(nodes[-2], nodes[-1], bias=False))

        self.encoder = nn.Sequential(*en_modulelist)

        self.decoder = nn.Sequential(*de_modulelist)

        node_string = ''
        for layer in nodes:
            node_string = node_string + str(layer) + '-'
        node_string = node_string[:-1]
        self.node_string = node_string

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        return self.decode(self.encode(x))

    def get_node_string(self):
        return self.node_string

class AE_3D_200(nn.Module):
    def __init__(self, n_features=4):
        super(AE_3D_200, self).__init__()
        self.en1 = nn.Linear(n_features, 200)
        self.en2 = nn.Linear(200, 100)
        self.en3 = nn.Linear(100, 50)
        self.en4 = nn.Linear(50, 3)
        self.de1 = nn.Linear(3, 50)
        self.de2 = nn.Linear(50, 100)
        self.de3 = nn.Linear(100, 200)
        self.de4 = nn.Linear(200, n_features)
        self.tanh = nn.Tanh()

    def encode(self, x):
        return self.en4(self.tanh(self.en3(self.tanh(self.en2(self.tanh(self.en1(x)))))))

    def decode(self, x):
        return self.de4(self.tanh(self.de3(self.tanh(self.de2(self.tanh(self.de1(self.tanh(x))))))))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def describe(self):
        return 'in-200-100-50-3-50-100-200-out'

class AE_big(nn.Module):
    def __init__(self, n_features=4):
        super(AE_big, self).__init__()
        self.en1 = nn.Linear(n_features, 8)
        self.en2 = nn.Linear(8, 6)
        self.en3 = nn.Linear(6, 4)
        self.en4 = nn.Linear(4, 3)
        self.de1 = nn.Linear(3, 4)
        self.de2 = nn.Linear(4, 6)
        self.de3 = nn.Linear(6, 8)
        self.de4 = nn.Linear(8, n_features)
        self.tanh = nn.Tanh()

    def encode(self, x):
        return self.en4(self.tanh(self.en3(self.tanh(self.en2(self.tanh(self.en1(x)))))))

    def decode(self, x):
        return self.de4(self.tanh(self.de3(self.tanh(self.de2(self.tanh(self.de1(self.tanh(x))))))))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def describe(self):
        return 'in-8-6-4-3-4-6-8-out'


# Some helper functions
def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True, pin_memory=True),
        DataLoader(valid_ds, batch_size=bs * 2, pin_memory=True),
    )


def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


def fit(epochs, model, loss_func, opt, train_dl, valid_dl, device):
    since = time.time()
    epochs_train_loss = []
    epochs_val_loss = []
    for epoch in range(epochs):
        running_train_loss = 0.
        epoch_start = time.perf_counter()
        model.train()
        for xb, yb in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            loss, lenxb = loss_batch(model, loss_func, xb, yb, opt)
            running_train_loss += np.multiply(loss, lenxb)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb_tmp.to(device), yb_tmp.to(device)) for xb_tmp, yb_tmp in valid_dl]
            )
        train_loss = running_train_loss / len(train_dl.dataset)
        epochs_train_loss.append(train_loss)
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        epochs_val_loss.append(val_loss)
        if(epoch % 1 == 0):
            current_time = time.perf_counter()
            delta_t = current_time - epoch_start
            # print('Epoch ' + str(epoch) + ':', 'Validation loss = ' + str(val_loss) + ' Time: ' + str(datetime.timedelta(seconds=delta_t)))
            print('Epoch: {:d} Train Loss: {:.3e} Val Loss: {:.3e}, Time: {}'.format(epoch, train_loss, val_loss, str(datetime.timedelta(seconds=delta_t))))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return pd.DataFrame({'Epoch': np.arange(epochs), 'train_loss': np.array(epochs_train_loss), 'val_loss': np.array(epochs_val_loss), 'epoch_time': delta_t})


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, x, y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss

