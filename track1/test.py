import torch
import opts
import logging
import numpy as np
from collections import defaultdict
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from utils import setup_seed, AverageMeter, save_checkpoint, MTLoss
from dataloader.ABAW4dataset import TestData
from metrics import AccF1Metric, CCCMetric, MultiLabelAccF1
from models import resnet
from models import sformer
from models import streaming
from models import hrnet


@torch.no_grad()
def evaluate(args, model, loader):
    model.eval()
    bar = tqdm(loader, desc=f'Validation', colour='green', position=0, leave=False)
    # metric_ex = AccF1Metric(ignore_index=8)
    # metric_va = CCCMetric(ignore_index=-5.0)
    # metric_au = MultiLabelAccF1(ignore_index=-1)
    # scores = defaultdict()
    with open(r'./MTL_predictions.txt', 'w') as f:
        f.write('image,valence,arousal,expression,aus' + '\n')
        for step, data in enumerate(bar):
            x = data['image'].to('cuda:0')
            result = model(x)  # batchx22 12 + 8 + 2
            logits_ex = result[:, 12:20]
            logits_au = result[:, :12]
            logits_va = result[:, 20:22] #tanh??

            pred = dict()
            pred['ex'] = torch.argmax(logits_ex, dim=1).detach().cpu().numpy().reshape(-1)
            pred['va'] = np.around(torch.tanh(logits_va).detach().cpu().numpy(), 3)
            pred['au'] = np.round(torch.sigmoid(logits_au).detach().cpu().numpy())

            for i in range(len(data)):
                f.write(data['file'][i] + ',' + str(pred['va'][i][0]) + ',' + str(pred['va'][i][1]) + ',' +
                        str(pred['ex'][i]))
                for j in range(12):
                    f.write(',' + str(int(pred['au'][i][j])))
                f.write('\n')

            # metric_ex.update(pred, label)
            # metric_va.update(y_pred=torch.tanh(logits_va).detach().cpu().numpy(), y_true=labels['VA'].detach().cpu().numpy())
            # metric_au.update(y_pred=np.round(torch.sigmoid(logits_au).detach().cpu().numpy()), y_true=labels['AU'].detach().cpu().numpy())

            # acc_ex = accuracy_score(y_true=label, y_pred=pred)
            # bar.set_postfix(data_fetch_time=data_time, batch_loss=loss.item(), avg_loss=total_loss / (step + 1), acc=acc_ex)

        # acc_ex, f1_ex = metric_ex.get()
        # acc_au, f1_au = metric_au.get()
        # scores['EX'] = {'EX:acc': acc_ex, 'f1': f1_ex, 'score': f1_ex}
        # scores['AU'] = {'AU:acc': acc_au, 'f1': f1_au, 'score': f1_au}
        # scores['VA'] = {'VA:ccc_v': metric_va.get()[0],'ccc_a': metric_va.get()[1], 'score': metric_va.get()[2]}
        # model.train()
        # metric_va.clear()
        # metric_au.clear()
        # metric_ex.clear()
        # return scores, loader_iter


def main(args):
    setup_seed(args.get('seed'))
    task = args.get('task')
    print(f'Task: {task}')
    # print('Model:', opt['model_name'])
    # print('Modality:', opt['modality'])
    # print('clip size', opt['n_frames'], opt['image_size'])

    datasets, datasets_size = TestData(args)

    if 'resnet' in opt['model_name']:
        model = getattr(resnet, 'se_resnet_18')(num_classes=22)  # 12 + 8 + 2
    elif 'sformer' in opt['model_name']:
        model = sformer.SpatialFormer()
    elif 'streaming' in opt['model_name']:
        emb = hrnet.hrnet18(pretrain_path='hrnet.pth')
        model = streaming.Streaming(emb)

    model.load_state_dict(torch.load(r'F:\lhc\ABAW4\track1\ckpt\best.pth'))
    model = model.cuda()

    evaluate(args, model, datasets)


if __name__ == '__main__':
    opt = opts.parse_opt()
    torch.cuda.set_device(opt.gpu_id)
    opt = vars(opt)
    main(opt)
