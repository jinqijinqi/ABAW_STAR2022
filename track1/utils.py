import os
import numpy as np
import torch
import torch.nn as nn
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def readtxt(file_name):
    data = []
    file = open(file_name, 'r')  #打开文件
    file_data = file.readlines() #读取所有行
    for row in file_data[1:]:
        row = row.replace('\n', '')
        tmp_list = row.split(',') #按‘，’切分每行的数据
        #tmp_list[-1] = tmp_list[-1].replace('\n',',') #去掉换行符
        data.append(tmp_list) #将每行数据插入data中
    return data


def save_checkpoint(state, filepath='./weights',
                    filename='latest.pth'):
    os.makedirs(filepath, exist_ok=True)
    save_path = os.path.join(filepath, filename)
    torch.save(state, save_path)


class AULoss(nn.Module):
    """
    Lin's Concordance correlation coefficient
    """

    def __init__(self, ignore=-1):
        super(AULoss, self).__init__()
        self.ignore = ignore
        #self.loss_fn = nn.BCEWithLogitsLoss()
        #[1, 2, 1, 1, 1, 1, 1, 6, 6, 5, 1, 5]
        self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.Tensor([1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 1, 2]).to(torch.cuda.current_device()))

    def forward(self, y_pred, y_true):
        """

        Args:
            y_pred: Nx12
            y_true: Nx12

        Returns:

        """
        index = y_true != self.ignore
        valid_sample_index = index.t()[0]
        valid_y_true = y_true[valid_sample_index][:]
        valid_y_pred = y_pred[valid_sample_index][:]
        device = y_true.device
        #loss = 0
        '''
        for i in range(y_true.shape[1]):
            index_i = index[:, i]
            y_true_i = y_true[:, i][index_i]
            y_pred_i = y_pred[:, i][index_i]
            if y_true_i.size(0) == 0:
                loss += torch.tensor(0.0, requires_grad=True).to(device)
                continue
            print(y_pred_i.shape, y_true_i.shape)
            loss += self.loss_fn(y_pred_i, y_true_i)
        '''
        loss = self.loss_fn(valid_y_pred, valid_y_true).mean()
        return loss


class CCCLoss(nn.Module):
    """
    Lin's Concordance correlation coefficient
    """

    def __init__(self, ignore=-5.0):
        super(CCCLoss, self).__init__()
        self.ignore = ignore

    def forward(self, y_pred, y_true):
        """
        y_true: shape of (N, )
        y_pred: shape of (N, )
        """
        batch_size = y_pred.size(0)
        device = y_true.device
        index = y_true != self.ignore
        index.requires_grad = False

        y_true = y_true[index]
        y_pred = y_pred[index]
        if y_true.size(0) <= 1:
            loss = torch.tensor(0.0, requires_grad=True).to(device)
            return loss
        x_m = torch.mean(y_pred)
        y_m = torch.mean(y_true)

        x_std = torch.std(y_true)
        y_std = torch.std(y_pred)

        v_true = y_true - y_m
        v_pred = y_pred - x_m

        s_xy = torch.sum(v_pred * v_true)

        numerator = 2 * s_xy
        denominator = x_std ** 2 + y_std ** 2 + (x_m - y_m) ** 2 + 1e-8

        ccc = numerator / (denominator * batch_size)

        loss = torch.mean(1 - ccc)

        return loss


class MTLoss:
    def __init__(self):
        self.loss_EX = nn.CrossEntropyLoss(weight = torch.tensor([0.1,1,1,1,0.12,0.3,0.2,0.1]).cuda(),ignore_index=8)
        self.loss_AU = AULoss()
        self.loss_VA = CCCLoss()

    def get_ex_loss(self, y_pred, y_true):
        y_pred = y_pred[:, 12:20]
        y_true = y_true.view(-1)
        loss = self.loss_EX(y_pred, y_true)
        return loss

    def get_au_loss(self, y_pred, y_true):
        # y_pred = torch.sigmoid(y_pred[:, :12])
        loss = self.loss_AU(y_pred[:, :12], y_true)
        return loss

    def get_va_loss(self, y_pred, y_true):
        y_pred_v = torch.tanh(y_pred[:, 20])
        y_pred_a = torch.tanh(y_pred[:, 21])
        # print(y_pred_v)
        # print(y_true[:, 0])
        loss = self.loss_VA(y_pred_v, y_true[:, 0]) + self.loss_VA(y_pred_a, y_true[:, 1])
        return loss

    def get_mt_loss(self, y_pred, y_true,normalize=False):  # multi-task loss
        loss_ex = self.get_ex_loss(y_pred, y_true['EX'])
        loss_au = self.get_au_loss(y_pred, y_true['AU'])
        loss_va = self.get_va_loss(y_pred, y_true['VA'])
        if normalize:
            valid_ex_label_num = np.sum(y_true['EX'].detach().cpu().numpy() != 8)
            if valid_ex_label_num != 0:
                loss_ex = loss_ex / valid_ex_label_num
            else:
                device = y_true.device
                loss_ex = torch.tensor(0.0, requires_grad=True).to(device)

            valid_au_label_num = np.sum((y_true['AU'].detach().cpu().numpy() != -1))
            if valid_au_label_num != 0:
                loss_au = loss_au / valid_au_label_num
            else:
                device = y_true.device
                loss_au = torch.tensor(0.0, requires_grad=True).to(device)

            valid_va_label_num = np.sum(y_true['VA'].detach().cpu().numpy() != -5.0)
            if valid_va_label_num != 0:
                loss_va = loss_va / valid_va_label_num
            else:
                device = y_true.device
                loss_va = torch.tensor(0.0, requires_grad=True).to(device)

        return [loss_ex, loss_au, loss_va]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count