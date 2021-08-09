import time
import numpy as np

from utils.options import opt, device
from utils.util import *
from .networks import UnetGenerator, RevealNet, EncodingNet


def load_nets():
    """Return prepared hiding net `Hnet`, reveal net `Rnet` and encoding net `Enet`."""
    if opt.cover_dependent:
        cover_dim = opt.channel_cover
        output_function_H = 'sigmoid'
    else:
        cover_dim = 0
        output_function_H = 'tanh'
    
    if opt.without_key:
        input_nc_H = cover_dim + opt.num_secrets * opt.channel_secret
        input_nc_R = opt.channel_cover
    else:
        input_nc_H = cover_dim + opt.num_secrets * (opt.channel_secret+opt.channel_key)
        input_nc_R = opt.channel_cover + opt.channel_key

    Hnet = UnetGenerator(
        input_nc=input_nc_H,
        output_nc=opt.channel_cover,
        num_downs=opt.num_downs,
        norm_type=opt.norm_type,
        output_function=output_function_H
    )
    Rnet = RevealNet(
        input_nc=input_nc_R,
        output_nc=opt.channel_secret,
        norm_type=opt.norm_type,
        output_function='sigmoid'
    )

    Enet = EncodingNet(opt.image_size, opt.channel_key, opt.redundance, opt.batch_size)

    Hnet.apply(weights_init)
    Rnet.apply(weights_init)
    Enet.apply(weights_init)
    Hnet = torch.nn.DataParallel(Hnet).to(device)
    Rnet = torch.nn.DataParallel(Rnet).to(device)
    Enet = torch.nn.DataParallel(Enet).to(device)
    return Hnet, Rnet, Enet


def load_checkpoints(Hnet, Rnet, Enet):
    """Load checkpoints for Hnet, Rnet and Enet."""
    checkpoint = torch.load(opt.checkpoint_path)
    Hnet.load_state_dict(checkpoint['H_state_dict'])
    Rnet.load_state_dict(checkpoint['R_state_dict'])
    if opt.redundance != -1:
        Enet.load_state_dict(checkpoint['E_state_dict'])


def adjust_learning_rate(optimizer, epoch, decay_num=5):
    """Set the learning rate to the initial LR decayed by `decay_num` every `lr_decay_freq` epochs."""
    lr = opt.lr * (1/decay_num ** (epoch // opt.lr_decay_freq))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def forward_pass(cover, secret, Hnet, Rnet, Enet,
                    criterion, epoch=None, modified_bits=0):
    """Forward propagation for hiding and reveal network and calculate losses and APD.
    
    Parameters:
        cover (Tensor)      -- cover image in batch
        secret (Tensor)     -- secret image in batch
        Hnet (Module)       -- hiding network
        Rnet (Module)       -- reveal network
        Enet (Module)       -- encoding network
        criterion (Loss)    -- loss function
        epoch (int/None)    -- epoch number
        modidied_bits (int) -- numbers of modified bits on the true key
    """
    cover, secret = cover.to(device), secret.to(device)
    (b, c, h, w), (_, c_s, h_s, w_s) = cover.shape, secret.shape
    assert h == h_s and w == w_s, "Cover and secret images shoule be in the same shape!"

    secret_list, rev_secret_list = [], []
    for i in range(opt.num_secrets):
        secret_list.append(secret[i*b:(i+1)*b, :, :, :])

    # using key
    count_diff = 0  # different bits between true and fake key
    if not opt.without_key:
        # set true key(s)
        key_list, encode_key_list = [], []  # original key(s); encoded key(s)
        # custom true key(s)
        if opt.key != '':
            if '[' not in opt.key:
                key_list.append(md5(opt.key))
            elif opt.key != '':
                rkeys = eval(opt.key)
                for rk in rkeys:
                    key_list.append(md5(rk))

            assert len(key_list) == opt.num_secrets, "Numbers of keys should be the same as secret images!"
        # random true key(s)
        else:
            for i in range(opt.num_secrets):
                key_list.append(random_key(w))

        # set for fake keys
        fake_key, encode_fake_key = None, None
        if opt.generation_type == 'random':
            assert opt.num_secrets == 1
            num = np.random.randint(1, 129)
            fake_key = modify_key(key_list[0], num)
        elif opt.generation_type == 'gradual':
            RANGE, low, high = [128,64,32,16,4,1], None, None
            assert opt.num_secrets == 1 and epoch is not None
            if epoch < 40:
                low, high = RANGE[epoch//10], RANGE[epoch//10 + 1]
            else:
                low, high = RANGE[-2], RANGE[-1]
            num = np.random.randint(low, high)
            fake_key = modify_key(key_list[0], num)
        elif opt.generation_type == 'custom':
            fake_key = md5(opt.fake_key)
        else:
            fake_key = random_key(w)
            # make sure fake key is not equal to all the true keys
            for k in key_list:
                while torch.equal(k, fake_key):
                    fake_key = random_key(w)

        if opt.num_secrets == 1:
            count_diff = int(torch.sum(torch.abs(fake_key - key_list[0])).item)

        # redundant keys
        for k in key_list:
            encode_key_list.append(Enet(k))
        encode_fake_key = Enet(fake_key)

    # hiding net
    if opt.cover_dependent:
        if opt.without_key:
            H_input = cover
            for i in range(opt.num_secrets):
                H_input = torch.cat((H_input, secret_list[i]), dim=1)
        else:
            H_input = cover
            for i in range(opt.num_secrets):
                H_input = torch.cat((H_input, secret_list[i], encode_key_list[i]), dim=1)
    else:
        if opt.without_key:
            H_input = secret_list[0]
            for i in range(1, opt.num_secrets):
                H_input = torch.cat((H_input, secret_list[i]), dim=1)
        else:
            H_input = torch.cat((secret_list[0], encode_key_list[0]), dim=1)
            for i in range(1, opt.num_secrets):
                H_input = torch.cat((H_input, secret_list[i], encode_key_list[i]), dim=1)
    H_output = Hnet(H_input)

    if opt.cover_dependent:
        container = H_output
    else:
        container = H_output + cover
    
    H_loss, R_loss, R_loss_ = criterion(container, cover), 0.0, 0.0

    # reveal net with true key(s)
    if opt.without_key:
        R_output = Rnet(container)
        for i in range(opt.num_secrets):
            rev_secret_list.append(R_output[:, i*c_s:(i+1)*c_s, :, :])
            R_loss += criterion(secret_list[i], rev_secret_list[i])
    else:
        for i in range(opt.num_secrets):
            new_key = modify_key(key_list[i], modified_bits)
            new_encoded_key = Enet(new_key)
            
            R_output = Rnet(torch.cat((container, new_encoded_key), dim=1))
            rev_secret_list.append(R_output)
            R_loss += criterion(secret_list[i], rev_secret_list[i])
    R_loss /= opt.num_secrets

    # reveal net with fake key
    rev_secret_, R_loss_, R_diff_ = None, 0, 0
    if not opt.without_key:
        rev_secret_ = Rnet(torch.cat((container, encode_fake_key), dim=1))
        R_loss_ = criterion(rev_secret_, torch.zeros(rev_secret_.shape))
        R_diff_ = rev_secret_.abs().mean() * 255

    # L1 metric (APD)
    H_diff = (container-cover).abs().mean() * 255
    
    R_diff = 0
    for i in range(opt.num_secrets):
        R_diff += (rev_secret_list[i]-secret_list[i]).abs().mean() * 255
    R_diff /= opt.num_secrets

    return cover, container, secret_list, rev_secret_list, rev_secret_,\
            H_loss, R_loss, R_loss_, H_diff, R_diff, R_diff_, count_diff


def train(train_loader_cover, train_loader_secret, val_loader_cover, val_loader_secret,
            Hnet, Rnet, Enet, criterion, optimizer, scheduler):
    """Train Hnet and Rnet and schedule learning rate by the validation results.
    
    Parameters:
        train_loader_cover (DataLoader)  -- loader for training cover images
        train_loader_secret (DataLoader) -- loader for training secret images
        val_loader_cover (DataLoader)    -- loader for val cover images
        val_loader_secret (DataLoader)   -- loader for val secret images
        Hnet (Module)                    -- hiding network
        Rnet (Module)                    -- reveal network
        Enet (Module)                    -- encoding network
        criterion (Loss)                 -- loss function
        optimizer (Module)               -- optimizer for nets
        scheduler (Scheduler)            -- scheduler for optimizer to set dynamic learning rate
    """
    min_loss = float('inf')
    h_loss_list, r_loss_list, r_loss_list_ = [], [], []
    batch_size = opt.batch_size
    iters_per_epoch = opt.dataset_size_train // opt.batch_size
    print("######## TRAIN BEGIN ########")
    
    for epoch in range(opt.start_epoch, opt.epochs):
        if opt.start_epoch != 0:
            assert opt.load_checkpoint, "Load checkpoint to continue training!"
        adjust_learning_rate(optimizer, epoch)
        # ATTENTION: must zip loaders in epoch's iter
        train_loader = zip(train_loader_cover, train_loader_secret)
        val_loader = zip(val_loader_cover, val_loader_secret)

        # training info
        batch_time, data_time = AverageMeter(), AverageMeter()
        H_losses, R_losses, R_losses_ = AverageMeter(), AverageMeter(), AverageMeter()
        Sum_losses = AverageMeter()
        H_diffs, R_diffs, R_diffs_ = AverageMeter(), AverageMeter(), AverageMeter()
        Diff_bits = AverageMeter()

        # turn on training mode
        Hnet.train()
        Rnet.train()
        Enet.train()

        start_time = time.time()

        # go through the training dataset
        for i, (cover, secret) in enumerate(train_loader, start=1):
            data_time.update(time.time() - start_time)

            cover, container, secret_list, rev_secret_list, rev_secret_,\
                H_loss, R_loss, R_loss_, H_diff, R_diff, R_diff_, count_diff\
                    = forward_pass(cover, secret, Hnet, Rnet, Enet, criterion,
                        epoch=epoch, modified_bits=0
            )

            H_losses.update(H_loss.item(), batch_size)
            R_losses.update(R_loss.item(), batch_size)
            H_diffs.update(H_diff.item(), batch_size)
            R_diffs.update(R_diff.item(), batch_size)
            Diff_bits.update(count_diff, 1)
            if not opt.without_key:
                R_losses_.update(R_loss_.item(), batch_size)
                R_diffs_.update(R_diff_.item(), batch_size)

            loss = H_loss + opt.beta*R_loss + opt.gamma*R_loss_
            Sum_losses.update(loss.item(), batch_size)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - start_time)
            start_time = time.time()

            log = "[%02d/%d] [%04d/%d]\tH_loss: %.6f R_loss: %.6f R_loss_:%.6f H_diff: %.4f R_diff: %.4f R_diff_: %.4f Diff_bits: %.4f data_time: %.4f batch_time: %.4f" % (
                epoch, opt.epochs, i, iters_per_epoch,
                H_losses.val, R_losses.val, R_losses_.val,
                H_diffs.val, R_diffs.val, R_diffs_.val,
                Diff_bits.val, data_time.val, batch_time.val
            )

            if i % opt.log_freq == 0:
                print(log)
            if epoch == 0 and i % opt.result_pic_freq == 0:
                save_result_pic(
                    batch_size, cover, container,
                    secret_list, rev_secret_list, rev_secret_,
                    epoch, i, opt.train_pics_save_dir
                )

        # end of an epoch
        save_result_pic(
            batch_size, cover, container,
            secret_list, rev_secret_list, rev_secret_,
            epoch, i, opt.train_pics_save_dir
        )

        epoch_log = "Training Epoch[%02d]\tSumloss=%.6f Hloss=%.6f Rloss=%.6f Rloss_=%.6f Hdiff=%.4f Rdiff=%.4f Rdiff_=%.4f Diff_bits=%.4f lr= %.6f Epoch Time=%.4f" % (
            epoch, Sum_losses.avg,
            H_losses.avg, R_losses.avg, R_losses_.avg,
            H_diffs.avg, R_diffs.avg, R_diffs.avg,
            Diff_bits.avg,
            optimizer.param_groups[0]['lr'],
            batch_time.sum
        )
        print_log(epoch_log)

        h_loss_list.append(H_losses.avg)
        r_loss_list.append(R_losses.avg)
        if not opt.without_key:
            r_loss_list_.append(R_losses_.avg)
        save_loss_pic(h_loss_list, r_loss_list, r_loss_list_)

        val_Sum_loss, val_H_loss, val_R_loss, val_R_loss_, val_H_diff, val_R_diff, val_R_diff_\
            = inference(val_loader_cover, val_loader_secret,
                Hnet, Rnet, Enet, criterion,
                save_num=1, mode='val', epoch=epoch
        )

        scheduler.step(val_Sum_loss)

        is_best = val_Sum_loss < min_loss
        min_loss = min(min_loss, val_Sum_loss)

        state = 'best' if is_best else 'newest'
        print_log("Save the %s checkpoint: epoch%03d" % (state, epoch+1))
        save_checkpoint(
            {
                'epoch': epoch+1,
                'H_state_dict': Hnet.state_dict(),
                'R_state_dict': Rnet.state_dict(),
                'E_state_dict': Enet.state_dict(),
                'optimizer': optimizer.state_dict()
            }, is_best=is_best
        )
        print("######## TRAIN END ########")


def inference(data_loader_cover, data_loader_secret, Hnet, Rnet, Enet, criterion,\
                save_num=1, mode='test', epoch=None):
    """Validate or test the performance of Hnet and Rnet.

    Parameters:
        data_loader_cover (DataLoader)  -- data loader for cover images
        data_loader_secret (DataLoader) -- data loader for secret images
        Hnet (Module)                   -- hiding network
        Rnet (Module)                   -- reveal network
        Enet (Module)                   -- encoding network
        criterion (Loss)                -- loss function
        save_num (int)                  -- numbers of saved images
        mode (str)                      -- mode of inference [val | test]
        epoch (int/None)                -- epoch number
    """
    assert mode in ['val', 'test'], 'Mode is expected to be either `val` or `test`'

    batch_size = opt.batch_size

    print("\n#### %s begin ####" % mode)
    
    # inference info
    H_losses, R_losses, R_losses_ = AverageMeter(), AverageMeter(), AverageMeter()
    Sum_losses = AverageMeter()
    H_diffs, R_diffs, R_diffs_ = AverageMeter(), AverageMeter(), AverageMeter()
    Diff_bits = AverageMeter()

    # turn on val mode
    Hnet.eval()
    Rnet.eval()

    data_loader = zip(data_loader_cover, data_loader_secret)
    for i, (cover, secret) in enumerate(data_loader, start=1):
        cover, container, secret_list, rev_secret_list, rev_secret_,\
            H_loss, R_loss, R_loss_, H_diff, R_diff, R_diff_, count_diff\
                = forward_pass(cover, secret, Hnet, Rnet, Enet, criterion,
                    epoch=epoch, modified_bits=0
        )

        H_losses.update(H_loss.item(), batch_size)
        R_losses.update(R_loss.item(), batch_size)
        H_diffs.update(H_diff.item(), batch_size)
        R_diffs.update(R_diff.item(), batch_size)
        Diff_bits.update(count_diff, 1)
        if not opt.without_key:
            R_losses_.update(R_loss_.item(), batch_size)
            R_diffs_.update(R_diff_.item(), batch_size)

        loss = H_loss + opt.beta*R_loss + opt.gamma*R_loss_
        Sum_losses.update(loss.item(), batch_size)

        if i <= save_num:
            if mode == 'test':
                save_result_pic(
                    batch_size, cover, container,
                    secret_list, rev_secret_list, rev_secret_,
                    epoch=None, i=i,
                    save_path=opt.test_pics_save_dir
                )
            else:
                save_result_pic(
                    batch_size, cover, container,
                    secret_list, rev_secret_list, rev_secret_,
                    epoch=epoch, i=i,
                    save_path=opt.val_pics_save_dir
                )

    if mode == 'test':
        log = "Test\tSumloss=%.6f Hloss=%.6f Rloss=%.6f Rloss_=%.6f Hdiff=%.4f Rdiff=%.4f Rdiff_=%.4f Diff_bits=%.4f" % (
            Sum_losses.avg, H_losses.avg, R_losses.avg, R_losses_.avg,
            H_diffs.avg, R_diffs.avg, R_diffs_.avg, Diff_bits.avg
        )
    else:
        log = "Validation Epoch[%02d]\tSumloss=%.6f Hloss=%.6f Rloss=%.6f Rloss_=%.6f Hdiff=%.4f Rdiff=%.4f Rdiff_=%.4f Count=%.4f" % (
            epoch, Sum_losses.avg,
            H_losses.avg, R_losses.avg, R_losses_.avg,
            H_diffs.avg, R_diffs.avg, R_diffs_.avg, Diff_bits.avg
        )
    print_log(log)
    print("#### %s end ####\n" % mode)
    return Sum_losses.avg, H_losses.avg, R_losses.avg, R_losses_.avg, H_diffs.avg, R_diffs.avg, R_diffs_.avg
