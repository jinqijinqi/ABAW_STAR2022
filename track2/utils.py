from tqdm import tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

def plot_history(epochs, Acc, Loss, lr):
    plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    epoch_list = range(1,epochs + 1)
    plt.plot(epoch_list, Loss['train_loss'])
    plt.plot(epoch_list, Loss['val_loss'])
    plt.xlabel('epoch')
    plt.ylabel('Loss Value')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('vis/history_Loss.png')
    plt.show()
    
    plt.plot(epoch_list, Acc['train_acc'])
    plt.plot(epoch_list, Acc['val_acc'])
    plt.xlabel('epoch')
    plt.ylabel('Acc Value')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('vis/history_Acc.png')
    plt.show()
    
    plt.plot(epoch_list, lr)
    plt.xlabel('epoch')
    plt.ylabel('Train LR')
    plt.savefig('vis/history_Lr.png')
    plt.show()

def acc_f1_score(y_true, y_pred, ignore_index=None, normalize=False, average='macro', **kwargs):
    """Multi-class f1 score and accuracy"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if ignore_index is not None:
        leave = y_true != ignore_index
        y_true = y_true[leave]
        y_pred = y_pred[leave]
    f1 = f1_score(y_true=y_true, y_pred=y_pred, average=average, **kwargs)
    acc = accuracy_score(y_true=y_true, y_pred=y_pred, normalize=normalize)
    return acc, f1

class AccF1Metric(object):
    def __init__(self, ignore_index, average='macro'):
        self.ignore_index = ignore_index
        self.average = average
        self.y_pred = []
        self.y_true = []

    def update(self, y_pred, y_true):
        y_pred = torch.softmax(y_pred, dim=1)
        y_pred = torch.argmax(y_pred, dim=1).detach().cpu().numpy().reshape(-1)
        self.y_pred.append(y_pred)
        self.y_true.append(y_true.detach().cpu().numpy())

    def clear(self):
        self.y_true = []
        self.y_pred = []

    def get(self):
        # y_true = np.stack(self.y_true, axis=0).reshape(-1)
        # y_pred = np.stack(self.y_pred, axis=0).reshape(-1)
        y_true = np.concatenate(self.y_true)
        y_pred = np.concatenate(self.y_pred)
        acc, f1 = acc_f1_score(y_true=y_true, y_pred=y_pred,
                               average=self.average,
                               normalize=True,
                               ignore_index=self.ignore_index)
        return acc, f1

# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9) 
# criterion = nn.CrossEntropyLoss()
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=3,factor=0.5,min_lr=1e-6)
def train(model,trainloader, valloader, optimizer , criterion, scheduler , epochs = 10 , path = './model.pth',  writer = False, verbose = False, logs = False, pretrain = ''):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if pretrain != '':
        print('Load weights {}.'.format(pretrain))
        model.load_state_dict(torch.load(pretrain))

    best_acc = 0
    train_acc_list, val_acc_list = [],[]
    train_loss_list, val_loss_list = [],[]
    lr_list  = []
    epoch_step = len(trainloader)
    epoch_step_val = len(valloader)
    
    if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集，或者减小batchsize")
    
    for epoch in range(epochs):
        train_loss = 0
        train_metric = AccF1Metric(ignore_index=None)
        val_loss = 0
        val_metric = AccF1Metric(ignore_index=None)
        if torch.cuda.is_available():
            model = model.to(device)
        model.train()
        print('Start Train')
        with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{epochs}',postfix=dict,mininterval=0.3) as pbar:
            for step,data in enumerate(trainloader,start=0):
                im,label = data
                im = im.to(device)
                label = label.to(device)
                #---------------------
                #  释放内存
                #---------------------
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                #----------------------#
                #   清零梯度
                #----------------------#
                optimizer.zero_grad()
                #----------------------#
                #   前向传播forward
                #----------------------#
                outputs = model(im)
                #----------------------#
                #   计算损失
                #----------------------#
                loss = criterion(outputs,label)
                train_loss += loss.data
                train_metric.update(outputs,label)
                #----------------------#
                #   反向传播
                #----------------------#
                # backward
                loss.backward()
                # 更新参数
                optimizer.step()
                lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix(**{'Train Loss' : train_loss.item()/(step+1),
                                    'Lr'   : lr})
                pbar.update(1)
        train_loss = train_loss.item() / len(trainloader)
        train_acc, train_f1 = train_metric.get()
        if verbose:
            train_acc_list.append(train_acc)
            train_loss_list.append(train_loss)

        # 记录学习率
        lr = optimizer.param_groups[0]['lr']
        if verbose:
            lr_list.append(lr)
        
        # 更新学习率
        scheduler.step(train_loss)
        
        print('Finish Train')

        model.eval()
        print('Start Validation')
        #--------------------------------
        #   相同方法，同train
        #--------------------------------
        with tqdm(total=epoch_step_val,desc=f'Epoch {epoch + 1}/{epochs}',postfix=dict,mininterval=0.3) as pbar2:
            for step,data in enumerate(valloader,start=0):
                im,label = data
                im = im.to(device)
                label = label.to(device)
                with torch.no_grad():
                    if step >= epoch_step_val:
                        break
                    
                    # 释放内存
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    #----------------------#
                    #   前向传播
                    #----------------------#
                    outputs = model(im)
                    loss = criterion(outputs,label)
                    val_loss += loss.data
                    # probs, pred_y = outputs.data.max(dim=1) # 得到概率
                    # test_acc += (pred_y==label).sum().item()
                    # total += label.size(0)
                    val_metric.update(outputs,label)
                    
                    pbar2.set_postfix(**{'Val Loss': val_loss.item() / (step + 1)})
                    pbar2.update(1)

        val_loss = val_loss.item() / len(valloader)
        val_acc, val_f1 = val_metric.get()
        if verbose:
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)
        print('Finish Validation')

        print("Epoch [{:>3d}/{:>3d}]  Train Acc: {:>3.2f}%  F1: {:>3.2f}% Loss: {:>.6f} || Val Acc: {:>3.2f}%  F1: {:>3.2f}% Loss: {:>.6f} || Learning Rate:{:>.6f}"
              .format(epoch+1,epochs,train_acc,train_f1,train_loss,val_acc,val_f1,val_loss,lr))
        
        # ====================== 使用 tensorboard ==================:
        if writer:
            writer.add_scalars('Loss', {'train': train_loss.item(),
                            'valid': val_loss.item()}, epoch+1)
            writer.add_scalars('Acc', {'train': train_acc.item() ,
                            'valid': val_acc.item()}, epoch+1)
#               writer.add_scalars('Learning Rate',lr,i+1)
        # =========================================================
        train_metric.clear()
        val_metric.clear()
        #------------------------------------------------------------#
        #       保存日志模型，中断以后可以继续训练
        #------------------------------------------------------------#
        if logs:
            torch.save(model.state_dict(), 'logs/ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, train_loss, val_loss))
        
        #-----------------------------------------------------------#
        # 如果取得更好的准确率，就保存模型
        #-----------------------------------------------------------#
        if path != '':
            if val_acc > best_acc:
                torch.save(model.state_dict(),'./model/'+path)
                best_acc = val_acc
    torch.cuda.empty_cache()
    if verbose:
        Acc = {}
        Loss = {}
        Acc['train_acc'] = train_acc_list
        Acc['val_acc'] = val_acc_list
        Loss['train_loss'] = train_loss_list
        Loss['val_loss'] = val_loss_list
        Lr = lr_list
        return Acc, Loss, Lr

def train_freeze(model,trainloader, valloader, optimizer , criterion, scheduler , epochs = 10 , path = './model.pth',  writer = False, verbose = False, logs = False, pretrain = ''):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if pretrain != '':
        print('Load weights {}.'.format(pretrain))
        model.load_state_dict(torch.load(pretrain))

    best_acc = 0
    train_acc_list, val_acc_list = [],[]
    train_loss_list, val_loss_list = [],[]
    lr_list  = []
    epoch_step = len(trainloader)
    epoch_step_val = len(valloader)
    
    if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集，或者减小batchsize")
    
    for epoch in range(epochs):
        train_loss = 0
        train_metric = AccF1Metric(ignore_index=None)
        val_loss = 0
        val_metric = AccF1Metric(ignore_index=None)
        if torch.cuda.is_available():
            model = model.to(device)
        model.train()
        #model.backbone.eval()
        model.resnet.eval()
        model.hrnet.backbone.eval()
        print('Start Train')
        with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{epochs}',postfix=dict,mininterval=0.3) as pbar:
            for step,data in enumerate(trainloader,start=0):
                im,label = data
                im = im.to(device)
                label = label.to(device)
                #---------------------
                #  释放内存
                #---------------------
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                #----------------------#
                #   清零梯度
                #----------------------#
                optimizer.zero_grad()
                #----------------------#
                #   前向传播forward
                #----------------------#
                outputs = model(im)
                #----------------------#
                #   计算损失
                #----------------------#
                loss = criterion(outputs,label)
                train_loss += loss.data
                train_metric.update(outputs,label)
                #----------------------#
                #   反向传播
                #----------------------#
                # backward
                loss.backward()
                # 更新参数
                optimizer.step()
                lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix(**{'Train Loss' : train_loss.item()/(step+1),
                                    'Lr'   : lr})
                pbar.update(1)
        train_loss = train_loss.item() / len(trainloader)
        train_acc, train_f1 = train_metric.get()
        if verbose:
            train_acc_list.append(train_acc)
            train_loss_list.append(train_loss)

        # 记录学习率
        lr = optimizer.param_groups[0]['lr']
        if verbose:
            lr_list.append(lr)
        
        # 更新学习率
        scheduler.step(train_loss)
        
        print('Finish Train')

        model.eval()
        print('Start Validation')
        #--------------------------------
        #   相同方法，同train
        #--------------------------------
        with tqdm(total=epoch_step_val,desc=f'Epoch {epoch + 1}/{epochs}',postfix=dict,mininterval=0.3) as pbar2:
            for step,data in enumerate(valloader,start=0):
                im,label = data
                im = im.to(device)
                label = label.to(device)
                with torch.no_grad():
                    if step >= epoch_step_val:
                        break
                    
                    # 释放内存
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    #----------------------#
                    #   前向传播
                    #----------------------#
                    outputs = model(im)
                    loss = criterion(outputs,label)
                    val_loss += loss.data
                    # probs, pred_y = outputs.data.max(dim=1) # 得到概率
                    # test_acc += (pred_y==label).sum().item()
                    # total += label.size(0)
                    val_metric.update(outputs,label)
                    
                    pbar2.set_postfix(**{'Val Loss': val_loss.item() / (step + 1)})
                    pbar2.update(1)

        val_loss = val_loss.item() / len(valloader)
        val_acc, val_f1 = val_metric.get()
        if verbose:
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)
        print('Finish Validation')

        print("Epoch [{:>3d}/{:>3d}]  Train Acc: {:>3.2f}%  F1: {:>3.2f}% Loss: {:>.6f} || Val Acc: {:>3.2f}%  F1: {:>3.2f}% Loss: {:>.6f} || Learning Rate:{:>.6f}"
              .format(epoch+1,epochs,train_acc,train_f1,train_loss,val_acc,val_f1,val_loss,lr))
        
        # ====================== 使用 tensorboard ==================:
        if writer:
            writer.add_scalars('Loss', {'train': train_loss.item(),
                            'valid': val_loss.item()}, epoch+1)
            writer.add_scalars('Acc', {'train': train_acc.item() ,
                            'valid': val_acc.item()}, epoch+1)
#               writer.add_scalars('Learning Rate',lr,i+1)
        # =========================================================
        train_metric.clear()
        val_metric.clear()
        #------------------------------------------------------------#
        #       保存日志模型，中断以后可以继续训练
        #------------------------------------------------------------#
        if logs:
            torch.save(model.state_dict(), 'logs/ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, train_loss, val_loss))
        
        #-----------------------------------------------------------#
        # 如果取得更好的准确率，就保存模型
        #-----------------------------------------------------------#
        if path != '':
            if val_acc > best_acc:
                torch.save(model,'./model/'+path)
                best_acc = val_acc
    torch.cuda.empty_cache()
    if verbose:
        Acc = {}
        Loss = {}
        Acc['train_acc'] = train_acc_list
        Acc['val_acc'] = val_acc_list
        Loss['train_loss'] = train_loss_list
        Loss['val_loss'] = val_loss_list
        Lr = lr_list
        return Acc, Loss, Lr