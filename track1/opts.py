import argparse


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', '-t', type=str, default='ALL', help='task, one of [ALL, EX, AU, VA]')
    parser.add_argument('--train_dir', '-td', type=str, default='F:\lhc\ABAW4\MTL\dataset\cropped_aligned')
    parser.add_argument('--test_dir', '-vd', type=str, default=r'F:\lhc\ABAW4\MTL\dataset\test_aligned')
    parser.add_argument('--train_lmdb_dir', '-tld', type=str, default='F:\lhc\ABAW4\MTL\dataset\cropped_aligned_lmdb')
    parser.add_argument('--test_lmdb_dir', '-vld', type=str, default=r'F:\lhc\ABAW4\MTL\dataset\test_aligned_lmdb')
    parser.add_argument('--train_label_path', '-tl', type=str, default=r'F:\lhc\ABAW4\MTL\dataset\training_set_annotations.txt')
    parser.add_argument('--val_label_path', '-vl', type=str, default=r'F:\lhc\ABAW4\MTL\dataset\validation_set_annotations.txt')
    parser.add_argument('--test_label_path', '-tlp', type=str, default=r'F:\lhc\ABAW4\track1\dataloader\test.txt')
    # parser.add_argument('--exp_dir', '-ed', type=str, default='experiments/avformer')
    # parser.add_argument('--cache_dir', '-cd', type=str, default='./data/cached_data')
    parser.add_argument('--image_size', type=int, default=256)
    # parser.add_argument('--modality', '-md', type=str, default='ALL')#A;V;M
    parser.add_argument('--checkpoint_path', '-ckpt', type=str, default='./ckpt')

    model = parser.add_argument_group(title='Model Parameters')
    model.add_argument('--dropout_rate', type=float, default=0.2)
    model.add_argument('--model_name', '-mn', type=str, default='streaming')

    training = parser.add_argument_group(title='Training Parameters')
    training.add_argument('--seed', default=123, type=int, help='for reproducibility')
    training.add_argument('--learning_rate', '-lr', default=5e-4, type=float, help='the initial learning rate')
    training.add_argument('--n_warmup_steps', type=int, default=0,
                          help='the number of warmup steps towards the initial lr')
    training.add_argument('--grad_clip', type=float, default=-1,
                          help='clip gradients at this value. it will not be used if the value is lower than 0')
    training.add_argument('--weight_decay', type=float, default=1e-2, help='Strength of weight regularization')
    training.add_argument('-e', '--epochs', type=int, default=200, help='number of epochs')
    training.add_argument('-b', '--batch_size', type=int, default=48, help='minibatch size')
    training.add_argument('--num_workers', type=int, default=0)
    training.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    training.add_argument('--early_stop_step', type=int, default=30)
    training.add_argument('--resume', action='store_true')
    training.add_argument('--start_epoch', type=int, default=15)
    training.add_argument('--eval_step', type=int, default=1)

    dataloader = parser.add_argument_group(title='Dataloader Parameters')
    dataloader.add_argument('--n_frames', '--clip_len', type=int, default=16, help='the number of frames per clip')
    dataloader.add_argument('--dilation', type=int, default=3, help='dilation')
    dataloader.add_argument('--downsample_rate', '-ds', type=int, default=100)
    dataloader.add_argument('--audio_len_secs', '-als', type=int, default=10)
    dataloader.add_argument('--audio_shift_secs', '-ass', type=int, default=5)
    dataloader.add_argument('--n_mels', type=int, default=64)
    args = parser.parse_args()

    return args
