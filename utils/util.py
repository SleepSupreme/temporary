import os
import time
import copy
import random
import hashlib
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torchvision.utils as vutils
from skimage.metrics import peak_signal_noise_ratio as _PSNR
from skimage.metrics import structural_similarity as _SSIM

from .options import opt, device, parser


class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count


def md5(key: str) -> torch.Tensor:
    """Hash and binarize the `key` by MD5 algorithm."""
    hash_key = hashlib.md5(key.encode(encoding='UTF-8')).digest()
    binary_key = ''.join(format(x, '08b') for x in hash_key)
    tensor_key = torch.Tensor([float(x) for x in binary_key]).to(device)
    return tensor_key


def random_key(length: int) -> torch.Tensor:
    """Generate a key with `length` bits randomly."""
    return torch.Tensor([float(torch.randn(1)<0) for _ in range(length)]).to(device)


def modify_key(key: torch.Tensor, num: int) -> torch.Tensor:
    """Flip `num` bits in `key` randomly and return a modified key."""
    fake_key = copy.deepcopy(key)
    indices = random.sample(range(len(key)), num)
    fake_key[indices] = -key[indices] + 1  # 0/1 flip
    return fake_key


def weights_init(m):
    """Weights initialization for a network."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out') 
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)


def print_log(log_info, log_path=opt.log_path, console=True, rewrite=False):
    """Print log information to the console and log files.
    
    Parameters:
        log_info (str) -- message that will be saved
        log_path (str) -- path to save the log_info
        consolr (bool) -- print the message in the console or not
        rewrite (bool) -- rewrite the file or not
    """
    if console:  # print the info into the console
        print(log_info)
    # write the log information into a log file
    if not os.path.exists(log_path):
        fp = open(log_path, "w")
        fp.writelines(log_info + "\n")
        fp.close()
    else:
        mode = 'w' if rewrite else 'a+'
        with open(log_path, mode) as f:
            f.writelines(log_info + '\n')


def print_network(net, save_path=opt.options_path):
    """Print `net` information."""
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print_log(str(net), save_path, console=False)
    print_log('Total number of parameters: %d\n' % num_params, save_path, console=False)


def save_options(save_path=opt.options_path):
    """Save options as a .txt file to `save_path`."""
    message = ''
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    # options info should be the first message in the log file
    # so set the `rewrite` as True
    print_log(message, save_path, console=False, rewrite=True)


def save_checkpoint(state, is_best, save_path=opt.checkpoints_save_dir):
    """Save checkpoint files for training.
    
    Parameters:
        state (dict)    -- state of a network that will be saved
        is_best (bool)  -- the state is the best or not
        save_path (str) -- path to save the checkpoint
    """
    if is_best:  # best
        filename = '%s/checkpoint_best.pth.tar' % save_path
    else:  # newest
        filename = '%s/checkpoint_newest.pth.tar' % save_path
    torch.save(state, filename)


def save_image(input_image, image_path, save_all=False, start=0):
    """Save a 3D or 4D torch.Tensor as image(s) to the disk.
    
    Parameters:
        input_image (Tensor) -- a 3D or 4D tensor
        image_path (str)     -- path of a dir (to save all images) or file (to save one iamge)
        save_all (bool)      -- save all images in the batch (4D) or not
        start (int)          -- the begining number to name a saved image
    """
    # make dir if not exist
    if not save_all:
        save_dir, _ = os.path.split(image_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    else:  # save the whole batch images
        save_dir = image_path
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    if isinstance(input_image, torch.Tensor):  # detach the tensor from current graph
        image_tensor = input_image.detach()
    else:
        raise TypeError("Type of the input should be `torch.Tensor`")
    
    if not save_all:
        if image_tensor.dim() == 4:
            image_tensor = image_tensor[0]  # the first one in a batch
        elif image_tensor.dim() != 3:
            raise TypeError('input_image should be 3D or 4D, but get a [%d]D tensor' % len(image_tensor.shape))
        
        image_numpy = image_tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()  # add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        image_pil = Image.fromarray(image_numpy)
        image_pil.save(image_path)
    else:
        if image_tensor.dim() != 4:
            raise TypeError('input_image should be 4D if set `save_all` to True, but get a [%d]D tensor' % len(image_tensor.shape))
        for i in range(image_tensor.shape[0]):
            image_tensor_ = image_tensor[i]
            image_numpy = image_tensor_.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()  # add 0.5 after unnormalizing to [0, 255] to round to nearest integer
            image_pil = Image.fromarray(image_numpy)
            image_pil.save(image_path + '/%d.png' % (i+start))


def save_result_pic(batch_size, cover, container, secret_list, rev_secret_list, rev_secret_, epoch, i, save_path):
    """Save a batch of result pictures.
    
    Parameters:
        batch_size (int)          -- batch size of the following tensors
        cover (Tensor)            -- cover image batch
        container (Tensor)        -- container image batch
        secret_list (list)         -- list of secret image batch(es) (maybe hiding several images)
        rev_secret_list (list)     -- list of revealed secrer image batch(es)
        rev_secret_ (Tensor/None) -- revealed image batch with fake keys
        epoch (int/None)          -- epoch number in training stage or None in val or test
        i (int)                   -- iteration number
        save_path (str)           -- path to save the result picture
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    if epoch is None:
        result_name = '%s/result_pic_batch%04d.png' % (save_path, i)
    else:
        result_name = '%s/result_pic_epoch%03d_batch%04d.png' % (save_path, epoch, i)

    cover_gap = container - cover
    cover_gap = (cover_gap*10 + 0.5).clamp_(0.0, 1.0)
    show_cover = torch.cat((cover, container, cover_gap), dim=0)
    if cover_gap.shape[1] == 1:  # gray
        show_cover = show_cover.repeat(1, 3, 1, 1)

    secret_gap_0 = rev_secret_list[0] - secret_list[0]
    secret_gap_set = [(secret_gap_0*10 + 0.5).clamp_(0.0, 1.0)]
    show_secret = torch.cat((secret_list[0], rev_secret_list[0], secret_gap_set[0]), dim=0)
    for i in range(1, opt.num_secrets):
        secret_gap_i = rev_secret_list[i] - secret_list[i]
        secret_gap_set.append((secret_gap_i*10 + 0.5).clamp_(0.0, 1.0))
        show_secret = torch.cat((show_secret, secret_list[i], rev_secret_list[i], secret_gap_set[i]), dim=0)
    if secret_gap_0.shape[1] == 1:  # gray
        show_secret = show_secret.repeat(1, 3, 1, 1)
    
    show_all = torch.cat((show_cover, show_secret), dim=0)
    if rev_secret_ is not None:
        rev_secret_ = rev_secret_.repeat(1, 3//opt.channel_secret, 1, 1)
        show_all = torch.cat((show_all, (rev_secret_*50).clamp_(0.0, 1.0)), dim=0)

    # vutils.save_image(show_all, result_name, batch_size, padding=1, normalize=False)
    grid = vutils.make_grid(show_all, nrow=batch_size, padding=1, normalize=False)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(result_name)


def save_loss_pic(h_losses_list, r_losses_list, r_losses_list_, save_path=opt.loss_save_path):
    """Save loss picture for Hnet and Rnet.
    
    Parameters:
        h_losses_list (list)  -- list of the losses of Hnet
        r_losses_list (list)  -- list of the losses of Rnet with the true key
        r_losses_list_ (list) -- list of the losses of Rnet with fake keys
        save_path (str)       -- path to save the loss picture
    """
    plt.title('Training Loss for Hnet and Rnet')
    plt.xlabel('epoch')
    plt.ylabel('MSE loss')
    plt.plot(list(range(1, len(h_losses_list)+1)), h_losses_list, label='H loss')
    plt.plot(list(range(1, len(r_losses_list)+1)), r_losses_list, label='R loss')
    if len(r_losses_list_) != 0:
        plt.plot(list(range(1, len(r_losses_list_)+1)), r_losses_list_, label='R loss (fake)')
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def PSNR(batch_image0, batch_image1):
    """Compute PSNR value between two tensor batches: `batch_image0` and `batch_image1`."""
    if isinstance(batch_image0, torch.Tensor) and isinstance(batch_image1, torch.Tensor):  # detach the tensor from current graph
        batch_image0_tensor = batch_image0.detach()
        batch_image1_tensor = batch_image1.detach()
    else:
        raise TypeError("Type of the input should be `torch.Tensor`")
    assert batch_image0_tensor.shape == batch_image1_tensor.shape, "Batches to compute PSNR should have the same shape!"
    
    SUM, b = 0.0, batch_image0_tensor.shape[0]
    for i in range(b):
        image0_numpy = batch_image0_tensor[i].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        image1_numpy = batch_image1_tensor[i].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

        SUM += _PSNR(image0_numpy, image1_numpy)
    return SUM / b


def SSIM(batch_image0, batch_image1):
    """Compute SSIM value between two tensor batches: `batch_image0` and `batch_image1`."""
    if isinstance(batch_image0, torch.Tensor) and isinstance(batch_image1, torch.Tensor):  # detach the tensor from current graph
        batch_image0_tensor = batch_image0.detach()
        batch_image1_tensor = batch_image1.detach()
    else:
        raise TypeError("Type of the input should be `torch.Tensor`")
    assert batch_image0_tensor.shape == batch_image1_tensor.shape, "Batches to compute SSIM should have the same shape!"
    
    SUM, b = 0.0, batch_image0_tensor.shape[0]
    for i in range(b):
        image0_numpy = batch_image0_tensor[i].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        image1_numpy = batch_image1_tensor[i].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

        SUM += _SSIM(image0_numpy, image1_numpy, multichannel=True)
    return SUM / b


def create_dirs():
    """Create `checkpoints_save_dir`, `train_pics_save_dir`, `val_pics_save_dir` in training mode.
    Create `test_pics_save_dir` in test mode.
    """
    try:
        if not opt.test:
            if not os.path.exists(opt.checkpoints_save_dir):
                os.makedirs(opt.checkpoints_save_dir)
            if not os.path.exists(opt.train_pics_save_dir):
                os.makedirs(opt.train_pics_save_dir)
            if not os.path.exists(opt.val_pics_save_dir):
                os.makedirs(opt.val_pics_save_dir)
            save_options()
        else:
            if not os.path.exists(opt.test_pics_save_dir):
                os.makedirs(opt.test_pics_save_dir)
    except OSError:
        print("XXXXXXXX mkdir failed XXXXXXXX")
