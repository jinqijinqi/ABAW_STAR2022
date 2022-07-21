import os
import lmdb
import numpy as np
# import pickle
import cv2
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils import readtxt
import pickle
from .autoaugment import ImageNetPolicy


def TestData(args):
    data_transforms = {
        'test': transforms.Compose([
            transforms.Resize((args['image_size'], args['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])}

    image_dataset = ABAW4Dataset(args, 'test', data_transforms)
    dataloders = DataLoader(image_dataset,
                            batch_size=args['batch_size'],
                            num_workers=args['num_workers'],
                            shuffle=False)
    dataset_sizes = len(image_dataset)
    return dataloders, dataset_sizes


def trainData(args):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((args['image_size'], args['image_size'])),
            ImageNetPolicy(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((args['image_size'], args['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ]),
    }

    image_datasets = {}
    image_datasets['train'] = ABAW4Dataset(args, 'train', data_transforms)
    image_datasets['val'] = ABAW4Dataset(args, 'val', data_transforms)

    dataloaders = {x: DataLoader(image_datasets[x],
                                 batch_size=args.get('batch_size'),
                                 num_workers=args.get('num_workers'),
                                 shuffle=True) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    return dataloaders, dataset_sizes


class ABAW4Dataset(Dataset):
    def __init__(self, opt, mode, data_transforms):
        super().__init__()
        self.task = opt.get('task')
        self.opt = opt
        self.mode = mode
        self.transforms = data_transforms[mode]
        assert self.task in ['ALL', 'EX', 'AU', 'VA']
        print('Constructing '+mode+' dataset')
        if mode == 'train':
            self.lmdb_label_path = opt.get('train_lmdb_dir')
            self.label_path = opt.get('train_label_path')
        elif mode == 'val':
            self.lmdb_label_path = opt.get('train_lmdb_dir')
            self.label_path = opt.get('val_label_path')
        elif mode == 'test':
            self.dir_path = opt.get('test_dir')
            self.label_path = opt.get('test_label_path')
        try:
            self.env_image = lmdb.open(os.path.join(self.lmdb_label_path, '.croped_jpeg'), create=False, lock=False,
                                       readonly=True)
        except:
            print('fail to open image lmdb')
        # self.video2orignal = pickle.load(open(os.path.join(self.video_dir, 'video2orignal.pkl'), 'rb'))
        self.imgs = self._make_dataset()
        print('Construct dataset finished')

    def __decodejpeg(self, jpeg):
        x = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        return x

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        if self.mode != 'test':
            [file, valence, arousal, expression, aus] = self.imgs[index]
            image = self.lmdb2image(file)
            image = Image.fromarray(image)
            image = self.transforms(image)
            sample = {
                'image': image,
                'VA': np.array([valence, arousal]),
                'EX': expression,
                'AU': np.array(aus)
            }
        else:
            [file, file_path] = self.imgs[index]
            # image = self.lmdb2image(file)
            # image = Image.fromarray(image)
            image = Image.open(file_path)
            image = self.transforms(image)
            sample = {
                'image': image,
                'file': file
            }
        return sample

    def lmdb2image(self, video_frame):
        try:
            with self.env_image.begin(write=False) as txn:
                jpeg = np.frombuffer(txn.get(video_frame.encode()), dtype='uint8')
                image = self.__decodejpeg(jpeg)
                return image
        except:
            #print('No image for:', key)
            return None

    def _make_dataset(self):
        if self.mode != 'test':
            if os.path.exists(self.label_path.replace('txt', 'pkl')):
                with open(self.label_path.replace('txt', 'pkl'), 'rb') as f:
                    items = pickle.load(f)
            else:
                labels = readtxt(self.label_path)
                img_id = [item[0] for item in labels]
                items = []
                for file in img_id:
                    try:
                        index_ = img_id.index(file)
                    except:
                        continue
                    [valence, arousal, expression] = float(labels[index_][1]), float(labels[index_][2]), int(labels[index_][3])
                    aus = [int(i) for i in labels[index_][4:]]
                    item = (file, valence, arousal, expression, aus)
                    items.append(item)
                print('Writing label to pickle')
                with open(self.label_path.replace('txt', 'pkl'), 'wb') as f:
                    pickle.dump(items, f)
        else:
            items = []
            for case in os.listdir(self.dir_path):
                case_path = os.path.join(self.dir_path, case)
                for filename in os.listdir(case_path):
                    file = case + '/' + filename
                    if file in [r'40-30-1280x720/06926.jpg']:
                        continue
                    file_path = os.path.join(case_path, filename)
                    item = (file, file_path)
                    items.append(item)
        return items
