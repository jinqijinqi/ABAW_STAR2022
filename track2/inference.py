import torch.nn as nn
import torchvision
import os
from utils import *
import numpy as np
import argparse
import hrnet
import sformer
import dualformer
import autoaugment
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm

def inference(test_set,model,model_name):
    test_transform = transforms.Compose([transforms.Resize(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    file_list = os.listdir(test_set)
    model.eval()
    with open('track2_'+model_name+'.txt','w+') as writer:
        writer.write('image,expression\n')
        with torch.no_grad():
            for img_file in tqdm(file_list):
                img = Image.open(os.path.join(test_set,img_file))
                img = test_transform(img).cuda()
                img = torch.unsqueeze(img, dim=0)
                # predict class
                output = torch.squeeze(model(img))
                predict = torch.softmax(output, dim=0)
                predict_cla = torch.argmax(predict).data.cpu().numpy()
                writer.write(img_file+','+str(int(predict_cla))+'\n')

def main(opt):
    if opt.model == "HRNet": 
        model = hrnet.hrnet18(pretrain_path=opt.weights,emb_dim=6,classify=True).cuda()
    elif opt.model == "ResNet":
        model = sformer.get_sformer(pretrain_path=opt.weights).cuda()
    elif opt.model == "Dual":
        model = dualformer.DualFormer('AffectNet_res18_acc0.6285.pth','hrnet_model_epoch3.pth').cuda()
        weight = torch.load(opt.weights).state_dict()
        model.load_state_dict(weight)
    print("You Choose {} model".format(opt.model))

    inference(os.path.join(opt.data,'track_2_test_data'),model,opt.model)

if __name__ == '__main__':
    # python train.py
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes', type=int, default=6, help='the number of classes')
    parser.add_argument('--model', type=str, default='ResNet', help='select model to train')
    parser.add_argument('--data', type=str, default=r'K:\ABAW4\track2\training_set_synthetic_images', help='data 文件路径')
    parser.add_argument('--weights',type=str,default=r'K:\ABAW4\track2\LSD_ResNet\model\resnet_model_epoch4.pth')
    
    opt = parser.parse_args()
    print(opt)
    main(opt)
