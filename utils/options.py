import os
import time
import argparse
import torch


# global variables
opt, device = None, None

parser = argparse.ArgumentParser()

# basic parameters
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0; 0,1,2; 0,2. use -1 for CPU')
parser.add_argument('--workers', type=int, default=2, help='number of workers to load dataset')
parser.add_argument('--exper_name', type=str, default=str(time.strftime('%Y-%m-%d_%H-%M', time.localtime())), help='experiment name')

# path parameters
parser.add_argument('--root', type=str, default='/content/drive/MyDrive', help='root dir of this project')
parser.add_argument('--exper_dir', type=str, default='', help='dir of one experiment')
parser.add_argument('--options_path', type=str, default='', help='path of options')
parser.add_argument('--log_path', type=str, default='', help='path of log information')
parser.add_argument('--checkpoints_save_dir', type=str, default='', help='dir of saving checkopints')
parser.add_argument('--train_pics_save_dir', type=str, default='', help='dir of saving pictures in training')
parser.add_argument('--val_pics_save_dir', type=str, default='', help='dir of saving pictures in validation')
parser.add_argument('--test_pics_save_dir', type=str, default='', help='dir of saving pictures in test')
parser.add_argument('--experiment_pics_save_dir', type=str, default='', help='dir of saving pictures in experiment')
parser.add_argument('--loss_save_path', type=str, default='', help='path of saving the curve of loss function')

# saving frequence parameters
parser.add_argument('--log_freq', type=int, default=10, help='frequency of saving log information')
parser.add_argument('--result_pic_freq', type=int, default=200, help='frequency of saving result pictures in the first epoch')

# dataset parameters
parser.add_argument('--image_size', type=int, default=128, help='size of images')
parser.add_argument('--dataset_size_train', type=int, default=25000, help='size of training dataset')
parser.add_argument('--dataset_size_val', type=int, default=500, help='size of training dataset')
parser.add_argument('--dataset_size_test', type=int, default=1000, help='size of training dataset')
parser.add_argument('--dataset_dir', type=str, default='', help='dir of dataset')

# model parameters
parser.add_argument('--cover_dependent', action='store_true', help='DDH(True) or UDH(False)')
parser.add_argument('--without_key', action='store_true', help='without key or not')
parser.add_argument('--channel_cover', type=int, default=3, help='number of channels for cover images')
parser.add_argument('--channel_secret', type=int, default=3, help='number of channels for secret images')
parser.add_argument('--channel_key', type=int, default=3, help='number of channels for embedded key')
parser.add_argument('--num_downs', type=int, default=5, help='number of down submodules in U-Net')
parser.add_argument('--norm_type', type=str, default='batch', help='type of normalization layer')
parser.add_argument('--loss', type=str, default='l2', help='loss function [l1 | l2]')
parser.add_argument('--num_secrets', type=int, default=1, help='the number of secret images to be hidden')

# training parameters
parser.add_argument('--epochs', type=int, default=80, help='epochs for training')
parser.add_argument('--batch_size', type=int, default=25, help='batch size')
parser.add_argument('--beta', type=float, default=0.75, help='weight of true reveal')
parser.add_argument('--gamma', type=float, default=0.5, help='weight of fake reveal')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_policy', type=str, default='step', help='learning rate policy [step | linear | plateau | cosine]')
parser.add_argument('--lr_decay_freq', type=int, default=30, help='frequency of decaying lr in `step` mode')
parser.add_argument('--decay_num', type=int, default=2, help='decay number for lr in `step` mode')
parser.add_argument('--shuffle_secret', action='store_true', help='hide nosie image as secret in training')

# test parameters
parser.add_argument('--test', action='store_true', help='test mode')
parser.add_argument('--load_checkpoint', action='store_true', help='load checkpoint')
parser.add_argument('--checkpoint_type', type=str, default='best', help='type of the checkpint file [best | newest]')
parser.add_argument('--checkpoint_path', type=str, default='', help='path of one checkpint file')

# key parameters
parser.add_argument('--redundance', type=int, default=-1, help='redundance size of key; e.g. `32` for mapping it to a 3*32*32 tensor; `-1` for simple duplication')
parser.add_argument('--generation_type', type=str, default='random_generation', help='generation type of a fake key [random | gradual | custom | ELSE (e.g. random_generation)]')
parser.add_argument('--modified_bits', type=int, default=0, help='number of modified bits for test')
parser.add_argument('--key', type=str, default='', help='true key')
parser.add_argument('--fake_key', type=str, default='', help='fake key')

# set global variables

opt, _ = parser.parse_known_args()  # opt = parser.parse_args()
device = torch.device("cuda:0" if opt.gpu_ids != -1 and torch.cuda.is_available() else "cpu")

# check
_ngpu = len(opt.gpu_ids.split(','))
assert _ngpu <= torch.cuda.device_count(), "There are not enough GPUs!"

_r = opt.redundance
assert (_r == -1) or (_r % 2 == 0 and _r >= 8), "Unexpected redundance size!"

if opt.test:
    opt.load_checkpoint = True
    assert opt.load_checkpoint, "Test mode must load the checkpoint file!"
    opt.generation_type = 'random_generation'

if opt.generation_type == 'custom':
    assert opt.fake_key != '', "A custom fake key should be given with the custom generation type!"

if not opt.test:
    assert opt.modified_bits == 0, "Do not modify true key in training mode"

# default path
opt.dataset_dir = opt.root + '/dataset'
opt.exper_dir = opt.root +  '/sdh/exper_info/' + opt.exper_name
opt.options_path = opt.exper_dir +  '/options.txt'
opt.log_path = opt.exper_dir +  '/log.txt'
opt.checkpoints_save_dir = opt.exper_dir + '/checkpoints'
opt.train_pics_save_dir = opt.exper_dir + '/train_pics'
opt.val_pics_save_dir = opt.exper_dir + '/val_pics'
opt.test_pics_save_dir = opt.exper_dir + '/test_pics'
opt.experiment_save_dir = opt.exper_dir + '/exp_pics'
opt.loss_save_path = opt.exper_dir + '/train_loss.png'
opt.checkpoint_path = opt.exper_dir + '/checkpoints/checkpoint_%s.pth.tar' % opt.checkpoint_type
