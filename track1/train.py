import torch
import opts
import os
import logging
import time
import numpy as np
from collections import defaultdict
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from utils import setup_seed, AverageMeter, save_checkpoint, MTLoss
from dataloader.ABAW4dataset import trainData
from metrics import AccF1Metric, CCCMetric, MultiLabelAccF1
from models import resnet
from models import sformer
from models import streaming
from models import hrnet

class EarlyStopper(object):

    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            torch.save(model.state_dict(), self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False


@torch.no_grad()
def evaluate(model, loader, loader_iter, device):
    model.eval()
    bar = tqdm(loader, desc=f'Validation', colour='green', position=0, leave=False)
    metric_ex = AccF1Metric(ignore_index=8)
    metric_va = CCCMetric(ignore_index=-5.0)
    metric_au = MultiLabelAccF1(ignore_index=-1)
    total_loss = 0
    lfn = MTLoss()
    scores = defaultdict()
    for step, data in enumerate(bar):
        label_ex = data['EX'].to(device)
        label_ex[label_ex == -1] = 8
        labels = {
            'VA': data['VA'].float().to(device),
            'AU': data['AU'].float().to(device),
            'EX': label_ex,
        }
        x = data['image'].to(device)
        result = model(x)  # batchx22 12 + 8 + 2
        logits_ex = result[:, 12:20]
        logits_au = result[:, :12]
        logits_va = result[:, 20:22] #tanh??
        losses = lfn.get_mt_loss(result, labels, normalize=False)
        loss = losses[0] + losses[1] + losses[2]
        total_loss += loss.item()

        pred = torch.argmax(logits_ex, dim=1).detach().cpu().numpy().reshape(-1)
        label = label_ex.detach().cpu().numpy().reshape(-1)

        metric_ex.update(pred, label)
        metric_va.update(y_pred=torch.tanh(logits_va).detach().cpu().numpy(), y_true=labels['VA'].detach().cpu().numpy())
        metric_au.update(y_pred=np.round(torch.sigmoid(logits_au).detach().cpu().numpy()), y_true=labels['AU'].detach().cpu().numpy())

        acc_ex = accuracy_score(y_true=label, y_pred=pred)
        # bar.set_postfix(data_fetch_time=data_time, batch_loss=loss.item(), avg_loss=total_loss / (step + 1), acc=acc_ex)

    acc_ex, f1_ex = metric_ex.get()
    acc_au, f1_au = metric_au.get()
    scores['EX'] = {'EX:acc': acc_ex, 'f1': f1_ex, 'score': f1_ex}
    scores['AU'] = {'AU:acc': acc_au, 'f1': f1_au, 'score': f1_au}
    scores['VA'] = {'VA:ccc_v': metric_va.get()[0],'ccc_a': metric_va.get()[1], 'score': metric_va.get()[2]}
    model.train()
    metric_va.clear()
    metric_au.clear()
    metric_ex.clear()
    return scores, loader_iter


def step_lr_adjust(optimizer, epoch, init_lr=1e-4, step_size=20, gamma=0.1):
    lr = init_lr * gamma ** (epoch // step_size)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def cycle_lr_adjust(optimizer, epoch, base_lr=1e-7, max_lr=1e-4, step_size=200, gamma=1):
    cycle = np.floor(1 + epoch/(2  * step_size))
    x = np.abs(epoch/step_size - 2 * cycle + 1)
    scale =  gamma ** (epoch // (2 * step_size))
    lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1-x)) * scale
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def train(args, model, dataloader, optimizer, epochs, device):
    early_stopper = EarlyStopper(num_trials=args['early_stop_step'], save_path=f'{args["checkpoint_path"]}/best.pth')
    start_epoch = 0
    if args['resume'] == True:
        start_epoch = args['start_epoch']
    learning_rate = args['learning_rate']
    lfn = MTLoss()
    for epoch in range(start_epoch, epochs):
        if epoch == 50:
            learning_rate = learning_rate*0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
        if epoch == 100:
            learning_rate = learning_rate*0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
        bar = tqdm(dataloader['train'], desc=f'Training Epoch:{epoch}', colour='blue', position=0, leave=True)
        logging.info('Epoch {}, lr {}'.format(epoch+1, optimizer.param_groups[0]['lr']))
        total_loss, ex_loss_record, au_loss_record, va_loss_record = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        for step, data in enumerate(bar):
            optimizer.zero_grad()
            x = data['image'].to(device)
            label_ex = data['EX'].to(device)
            label_ex[label_ex == -1] = 8
            labels = {
                'VA': data['VA'].float().to(device),
                'AU': data['AU'].float().to(device),
                'EX': label_ex
            }
            result = model(x)
            losses = lfn.get_mt_loss(result, labels, normalize=True)
            loss = 3*losses[0] + losses[1] + losses[2]
            ex_loss_record.update(losses[0].item())
            au_loss_record.update(losses[1].item())
            va_loss_record.update(losses[2].item())

            loss.backward()
            optimizer.step()
            total_loss.update(loss.item())
            bar.set_postfix(total=total_loss.avg, ex=ex_loss_record.avg, au=au_loss_record.avg, va=va_loss_record.avg)
        logging.info(f'Total Loss,{total_loss.avg}, Ex:{ex_loss_record.avg}, AU:{au_loss_record.avg}, VA:{va_loss_record.avg}')

        save_checkpoint(state=model.state_dict(), filepath=args["checkpoint_path"], filename='latest.pth')

        # eval
        val_loader_iter = iter(dataloader['val'])
        scores, val_loader_iter = evaluate(model, dataloader['val'], val_loader_iter, device)
        score_str = ''
        total_score = 0
        for task in ['EX', 'AU', 'VA']:
            score_dict = scores[task]
            for k, v in score_dict.items():
                score_str += f'{k}:{v:.3},'
            total_score = total_score + score_dict["score"]

        print(f'Training,{args["task"]}, Epoch:{epoch}, {score_str}')
        logging.info(f'Training,{args["task"]}, Epoch:{epoch}, {score_str}')

        if not early_stopper.is_continuable(model, total_score):
            print(f'validation: best score: {early_stopper.best_accuracy}')
            logging.info(f'validation: best score: {early_stopper.best_accuracy}')
            break


def main(args):
    setup_seed(args.get('seed'))
    task = args.get('task')
    print(f'Task: {task}')
    # print('Model:', opt['model_name'])
    # print('Modality:', opt['modality'])
    # print('clip size', opt['n_frames'], opt['image_size'])
    log_file_name = r'./log.txt'
    logging.basicConfig(filename=log_file_name, level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger()

    datasets, datasets_size = trainData(args)

    if 'resnet' in opt['model_name']:
        model = getattr(resnet, 'se_resnet_18')(num_classes=22)  # 12 + 8 + 2
    elif 'sformer' in opt['model_name']:
        model = sformer.SpatialFormer()
    elif 'streaming' in opt['model_name']:
        emb = hrnet.hrnet18(pretrain_path='hrnet.pth')
        model = streaming.Streaming(emb)
    model = model.cuda()

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])
    train(args, model, datasets, optimizer, epochs=args['epochs'], device=torch.cuda.current_device())


if __name__ == '__main__':
    opt = opts.parse_opt()
    torch.cuda.set_device(opt.gpu_id)
    opt = vars(opt)
    main(opt)
