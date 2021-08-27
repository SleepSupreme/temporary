import time
import numpy as np
from torch.optim import lr_scheduler

from utils.options import opt, device
from utils.util import *
from .networks import *


def get_nets(htype=opt.htype, rtype=opt.rtype, invertible=opt.invertible, cover_dependent=opt.cover_dependent, without_key=opt.without_key, fill_zeros=opt.fill_zeros):
    """According to options return several structured networks: hiding, reveal and encoding net."""
    if not invertible:
        if cover_dependent:
            cover_dim = opt.channel_cover
            output_function_H = 'sigmoid'
        else:
            cover_dim = 0
            output_function_H = 'tanh'
        
        if without_key:
            input_nc_H = cover_dim + opt.num_secrets * opt.channel_secret
            input_nc_R = opt.channel_cover
        else:
            input_nc_H = cover_dim + opt.num_secrets * (opt.channel_secret+opt.channel_key)
            input_nc_R = opt.channel_cover + opt.channel_key

        if fill_zeros:  # reserve excess channels for keys; the same as the channels with opt.without_key==False
            input_nc_H = cover_dim + opt.num_secrets * (opt.channel_secret+opt.channel_key)
            input_nc_R = opt.channel_cover + opt.channel_key

        if htype == 'unet':
            Hnet = Unet(
                input_nc=input_nc_H,
                output_nc=opt.channel_cover,
                num_downs=opt.num_downs,
                norm_type=opt.norm_type,
                output_function=output_function_H
            )
        else:
            raise NotImplementedError('Hiding network structure [%s] is not found!' % htype)

        if rtype == 'vanilla':
            Rnet = VanillaCNN(
                input_nc=input_nc_R,
                output_nc=opt.channel_secret,
                norm_type=opt.norm_type,
                output_function='sigmoid'
            )
        else:
            raise NotImplementedError('Reveal network structure [%s] is not found!' % rtype)

        Hnet.apply(weights_init)
        Rnet.apply(weights_init)
        Hnet = torch.nn.DataParallel(Hnet).to(device)
        Rnet = torch.nn.DataParallel(Rnet).to(device)
        HRnet = nn.ModuleList([Hnet, Rnet])
    
    else:  # invertible network
        input_nc = opt.channel_cover + opt.channel_secret * opt.num_secrets
        split_nc = opt.channel_cover
        
        HRnet = InvHidingNet(
            input_nc, split_nc, N=opt.num_inv, n=opt.num_sub
        )
        # weights are already initialized
        HRnet = torch.nn.DataParallel(HRnet).to(device)

    Enet = EncodingNet(opt.image_size, opt.channel_key, opt.redundance, opt.batch_size)
    Enet.apply(weights_init)
    Enet = torch.nn.DataParallel(Enet).to(device)
    return HRnet, Enet


def load_checkpoint(HRnet, Enet, optimizer=None, scheduler=None, invertible=opt.invertible, checkpoint_path=opt.checkpoint_path):
    """Load checkpoint from `checkpoint_path`."""
    checkpoint = torch.load(checkpoint_path)
    if not invertible:
        HRnet[0].load_state_dict(checkpoint['H_state_dict'])
        HRnet[1].load_state_dict(checkpoint['R_state_dict'])
    else:
        HRnet.load_state_dict(checkpoint['I_state_dict'])
    try:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        Enet.load_state_dict(checkpoint['E_state_dict'])
    except (ValueError, KeyError, AttributeError):  # in case did not save checkpoints (properly) or the inputs are None
        pass
    return HRnet, Enet, optimizer, scheduler


def get_scheduler(optimizer):
    """Return a learning rate scheduler by `opt.lr_policy` for `optimizer`."""
    if opt.lr_policy == 'fixed':
        def lambda_rule(epoch):
            return 1.0
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            epoch_init_lr = 30  # number of epochs with initial learning rate
            epoch_decay_lr = opt.epochs - epoch_init_lr  # number of epochs to linearly decay learning rate
            lr_l = 1.0 - max(0, epoch + 1 - epoch_init_lr) / float(epoch_decay_lr + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_freq, gamma=1/opt.decay_num)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5)
    elif opt.lr_policy == 'cosine':
        T_max_epoch = 20  # 2 * T_max_epoch == period of the cosine scheduler
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max_epoch, eta_min=0)
    else:
        raise NotImplementedError('learning rate policy [%s] is not implemented' % opt.lr_policy)
    return scheduler


def forward_pass(cover, secret, HRnet, Enet, criterion, epoch=None,
                modified_bits=opt.modified_bits, invertible=opt.invertible, cover_dependent=opt.cover_dependent,
                without_key=opt.without_key, key=opt.key, generation_type=opt.generation_type, fake_key=opt.fake_key):
    """Forward propagation for hiding and reveal network and calculate losses and APD.
    
    Parameters:
        cover (Tensor)         -- cover image in batch
        secret (Tensor)        -- secret image in batch
        HRnet                  -- hiding and reveal network
        Enet                   -- encoding network
        criterion              -- loss function
        epoch (int/None)       -- epoch number
        modidied_bits (int)    -- numbers of modified bits on the true key
        invertible (bool)      -- use invertible network or not
        cover_dependent (bool) -- cover-dependent deep hiding or not
        without_key (bool)     -- hiding data without key or not
        key (str)              -- custom true key or a null string
        generation_type (str)  -- generation type for fake keys
        fake_key (str)         -- custom fake key or a null string
    """
    cover, secret = cover.to(device), secret.to(device)
    (b, c, h, w), (_, c_s, h_s, w_s) = cover.shape, secret.shape
    assert h == h_s and w == w_s, "Cover and secret images shoule be in the same shape!"

    secret_list, rev_secret_list = [], []
    for i in range(opt.num_secrets):
        secret_list.append(secret[i*b:(i+1)*b, :, :, :])

    # using key
    count_diff = 0  # different bits between true and fake key
    if not without_key:
        # set true key(s)
        key_list, encode_key_list = [], []  # original key(s); encoded key(s)
        if key != '':  # custom true key(s)
            if '[' not in key:
                key_list.append(md5(key))
            elif key != '':
                rkeys = eval(key)
                for rk in rkeys:
                    key_list.append(md5(rk))

            assert len(key_list) == opt.num_secrets, "Numbers of keys should be the same as secret images!"
        else:  # random true key(s)
            for i in range(opt.num_secrets):
                key_list.append(random_key(w))

        # set fake keys
        fake_key, encode_fake_key = None, None
        if generation_type == 'uniform':
            assert opt.num_secrets == 1
            num = np.random.randint(1, 129)
            fake_key = modify_key(key_list[0], num)
        elif generation_type == 'gradual':
            RANGE = [128,96,80,64,48,32,16,8]
            assert opt.epochs == len(RANGE), "Length of the list should be the same as training epochs"
            num = RANGE[epoch//10]
            fake_key = modify_key(key_list[0], num)
        elif generation_type == 'custom':
            fake_key = md5(fake_key)
        elif generation_type == 'random':
            fake_key = random_key(w)
            # make sure fake key is not equal to all the true keys
            for k in key_list:
                while torch.equal(k, fake_key):
                    fake_key = random_key(w)
        else:
            return NotImplementedError('generation type [%s] is not implemented' % generation_type)

        if opt.num_secrets == 1:
            count_diff = int(torch.sum(torch.abs(fake_key - key_list[0])).item())

        # redundant keys
        for k in key_list:
            encode_key_list.append(Enet(k))
        encode_fake_key = Enet(fake_key)

    if not invertible:
        Hnet, Rnet = HRnet[0], HRnet[1]
        # hiding net
        if cover_dependent:
            if without_key:
                H_input = cover
                for i in range(opt.num_secrets):
                    H_input = torch.cat((H_input, secret_list[i]), dim=1)
            else:
                H_input = cover
                for i in range(opt.num_secrets):
                    H_input = torch.cat((H_input, secret_list[i], encode_key_list[i]), dim=1)
        else:
            if without_key:
                H_input = secret_list[0]
                for i in range(1, opt.num_secrets):
                    H_input = torch.cat((H_input, secret_list[i]), dim=1)
            else:
                H_input = torch.cat((secret_list[0], encode_key_list[0]), dim=1)
                for i in range(1, opt.num_secrets):
                    H_input = torch.cat((H_input, secret_list[i], encode_key_list[i]), dim=1)
        H_output = Hnet(H_input)

        if cover_dependent:
            container = H_output
        else:
            container = H_output + cover
        
        H_loss, R_loss, R_loss_ = criterion(container, cover), 0.0, 0.0

        # reveal net (with (modified) true key(s))
        if without_key:
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

        # reveal net (with fake key)
        rev_secret_, R_loss_, R_diff_ = None, 0, 0
        if not without_key:
            rev_secret_ = Rnet(torch.cat((container, encode_fake_key), dim=1))
            R_loss_ = criterion(rev_secret_, torch.zeros(rev_secret_.shape).to(device))
            R_diff_ = rev_secret_.abs().mean() * 255
    
    else:  # invertible network
        HR_input = cover
        for i in range(opt.num_secrets):
            HR_input = torch.cat((HR_input, secret_list[i]), dim=1)
        
        HR_output = HRnet(HR_input)
        container = HR_output[:, :c, :, :]
        z = HR_output[:, c:, :, :]
        
        CONSTANT = torch.ones(z.shape).to(device)*0.5
        H_loss = 32*criterion(container, cover) + criterion(z, CONSTANT)

        criterion_l1 = nn.L1Loss().to(device)
        HR_input_inv = torch.cat((container, CONSTANT), dim=1)
        HR_output_inv = HRnet(HR_input_inv, rev=True)
        cover_inv = HR_output_inv[:, :c, :, :]

        R_loss = criterion_l1(cover_inv, cover)
        for i in range(opt.num_secrets):
            secret_inv = HR_output_inv[:, c+i*c_s:c+(i+1)*c_s, :, :]
            R_loss = criterion_l1(secret_inv, secret_list[i])
            rev_secret_list.append(secret_inv)
        
        rev_secret_, R_loss_, R_diff_ = None, 0, 0
        if epoch is None:  # test mode
            CONSTANT_FAKE = torch.zeros(z.shape).to(device)
            CONSTANT_FAKE[:, :, :64, :64] = 1
            CONSTANT_FAKE[:, :, 64:, 64:] = 1
            rev_secret_ = HRnet(torch.cat((container, CONSTANT_FAKE), dim=1), rev=True)[:, -c_s:, :, :]
            R_loss_ = criterion(rev_secret_, secret_list[0])
            R_diff_ = (rev_secret_ - secret_list[0]).abs().mean() * 255


    # L1 metric (APD)
    H_diff = (container-cover).abs().mean() * 255
    
    R_diff = 0
    for i in range(opt.num_secrets):
        R_diff += (rev_secret_list[i]-secret_list[i]).abs().mean() * 255
    R_diff /= opt.num_secrets

    return cover, container, secret_list, rev_secret_list, rev_secret_,\
            H_loss, R_loss, R_loss_, H_diff, R_diff, R_diff_, count_diff


def train(train_loader_cover, train_loader_secret, val_loader_cover, val_loader_secret,
            HRnet, Enet, criterion, optimizer, scheduler):
    """Train HRnet and schedule learning rate by the validation results.
    
    Parameters:
        train_loader_cover  -- loader for training cover images
        train_loader_secret -- loader for training secret images
        val_loader_cover    -- loader for val cover images
        val_loader_secret   -- loader for val secret images
        HRnet               -- hiding and reveal network
        Enet                -- encoding network
        criterion           -- loss function
        optimizer           -- optimizer for nets
        scheduler           -- scheduler for optimizer to set dynamic learning rate
    """
    min_loss = float('inf')
    h_loss_list, r_loss_list, r_loss_list_ = [], [], []
    batch_size = opt.batch_size
    iters_per_epoch = opt.dataset_size_train // opt.batch_size
    print("######## TRAIN BEGIN ########")

    if opt.continue_train:
        checkpoint = torch.load(opt.checkpoint_path)
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0

    for epoch in range(start_epoch, opt.epochs):
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
        if not opt.invertible:
            HRnet[0].train()
            HRnet[1].train()
        else:
            HRnet.train()
        Enet.train()

        start_time = time.time()

        # go through the training dataset
        for i, (cover, secret) in enumerate(train_loader, start=1):
            data_time.update(time.time() - start_time)

            if opt.shuffle_secret:
                secret = shuffle_pix(secret)
            if opt.noise_secret:
                secret = torch.rand(secret.shape)

            cover, container, secret_list, rev_secret_list, rev_secret_,\
                H_loss, R_loss, R_loss_, H_diff, R_diff, R_diff_, count_diff\
                    = forward_pass(cover, secret, HRnet, Enet, criterion, epoch=epoch)

            H_losses.update(H_loss.item(), batch_size)
            R_losses.update(R_loss.item(), batch_size)
            H_diffs.update(H_diff.item(), batch_size)
            R_diffs.update(R_diff.item(), batch_size)
            Diff_bits.update(count_diff, 1)
            if not opt.without_key:
                R_losses_.update(R_loss_.item(), batch_size)
                R_diffs_.update(R_diff_.item(), batch_size)

            if not opt.invertible:
                loss = H_loss + opt.beta*R_loss + opt.gamma*R_loss_
            else:
                loss = H_loss + R_loss
            Sum_losses.update(loss.item(), batch_size)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - start_time)
            start_time = time.time()

            log = "[%02d/%d] [%04d/%d]\tH_loss: %.6f R_loss: %.6f R_loss_:%.6f H_diff: %.4f R_diff: %.4f R_diff_: %.4f Diff_bits: %03d data_time: %.4f batch_time: %.4f" % (
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

        epoch_log = "Training Epoch[%02d]\tSumloss=%.6f Hloss=%.6f Rloss=%.6f Rloss_=%.6f Hdiff=%.4f Rdiff=%.4f Rdiff_=%.4f Diff_bits=%03d lr= %.6f Epoch Time=%.4f" % (
            epoch, Sum_losses.avg,
            H_losses.avg, R_losses.avg, R_losses_.avg,
            H_diffs.avg, R_diffs.avg, R_diffs_.avg,
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
            = test(val_loader_cover, val_loader_secret,
                HRnet, Enet, criterion,
                save_num=1, mode='val', epoch=epoch
        )

        if opt.lr_policy == 'plateau':
            scheduler.step(val_Sum_loss)
        else:
            scheduler.step()

        is_best = val_Sum_loss < min_loss
        min_loss = min(min_loss, val_Sum_loss)

        state = 'best' if is_best else 'newest'
        print_log("Save the %s checkpoint: epoch%03d\n" % (state, epoch))
        
        save_dict = {
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        if not opt.invertible:
            save_dict['H_state_dict'] = HRnet[0].state_dict()
            save_dict['R_state_dict'] = HRnet[1].state_dict()
        else:
            save_dict['I_state_dict'] = HRnet.state_dict()
        if opt.redundance != 1:
            save_dict['E_state_dict'] = Enet.state_dict()
        save_checkpoint(save_dict, is_best)

    print("######## TRAIN END ########")


def test(data_loader_cover, data_loader_secret, HRnet, Enet, criterion,\
                save_num=1, mode='test', epoch=None, modified_bits=opt.modified_bits):
    """Validate or test the performance of HRnet.

    Parameters:
        data_loader_cover   -- data loader for cover images
        data_loader_secret  -- data loader for secret images
        HRnet               -- hiding and reveal network
        Enet                -- encoding network
        criterion           -- loss function
        save_num (int)      -- numbers of saved images
        mode (str)          -- mode of this function [val | test]
        epoch (int/None)    -- epoch number
        modidied_bits (int) -- numbers of modified bits on the true key
    """
    assert mode in ['val', 'test'], 'Mode is expected to be either `val` or `test`'

    batch_size = opt.batch_size
    
    if mode == 'test':
        try:
            from lpips import LPIPS
        except ModuleNotFoundError:
            os.system("pip install lpips")
            from lpips import LPIPS
        loss_fn_alex = LPIPS(net='alex')

    print("\n#### %s begin ####" % mode)
    
    # val/test info
    H_losses, R_losses, R_losses_ = AverageMeter(), AverageMeter(), AverageMeter()
    Sum_losses = AverageMeter()
    H_diffs, R_diffs, R_diffs_ = AverageMeter(), AverageMeter(), AverageMeter()
    Diff_bits = AverageMeter()

    # turn on val mode
    if not opt.invertible:
        HRnet[0].eval()
        HRnet[1].eval()
    else:
        HRnet.eval()

    data_loader = zip(data_loader_cover, data_loader_secret)
    for i, (cover, secret) in enumerate(data_loader, start=1):
        if opt.shuffle_secret:
            secret = shuffle_pix(secret)
        if opt.noise_secret:
            secret = torch.rand(secret.shape)

        cover, container, secret_list, rev_secret_list, rev_secret_,\
            H_loss, R_loss, R_loss_, H_diff, R_diff, R_diff_, count_diff\
                = forward_pass(cover, secret, HRnet, Enet, criterion, epoch=epoch)

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
                    save_path=opt.test_pics_save_dir, diff_bits=opt.modified_bits
                )
            else:
                save_result_pic(
                    batch_size, cover, container,
                    secret_list, rev_secret_list, rev_secret_,
                    epoch=epoch, i=i,
                    save_path=opt.val_pics_save_dir
                )

        # evaluate PSNR, SSIM and LPIPS metrics
        if mode == 'test':
            H_PSNRs, R_PSNRs = AverageMeter(), AverageMeter()
            H_SSIMs, R_SSIMs = AverageMeter(), AverageMeter()
            H_LPIPS, R_LPIPS = AverageMeter(), AverageMeter()

            h_psnr, r_psnr = PSNR(cover, container), 0
            h_ssim, r_ssim = SSIM(cover, container), 0
            h_lpips, r_lpips = loss_fn_alex(cover.cpu(), container.cpu()).mean(), 0
            for j in range(opt.num_secrets):
                r_psnr += PSNR(secret_list[j], rev_secret_list[j])
                r_ssim += SSIM(secret_list[j], rev_secret_list[j])
                r_lpips += loss_fn_alex(secret_list[j].cpu(), rev_secret_list[j].cpu()).mean()

            H_PSNRs.update(h_psnr, batch_size)
            H_SSIMs.update(h_ssim, batch_size)
            H_LPIPS.update(h_lpips, batch_size)
            R_PSNRs.update(r_psnr/opt.num_secrets, batch_size)
            R_SSIMs.update(r_ssim/opt.num_secrets, batch_size)
            R_LPIPS.update(r_lpips/opt.num_secrets, batch_size)

    if mode == 'test':
        if modified_bits != 0:
            log = "%s Modified_bits: %d\n" % (opt.checkpoint_type, modified_bits)
        else:
            log = "%s\n" % opt.checkpoint_type
        log  += "H_APD=%.4f H_PSNR=%.4f H_SSIM=%.4f H_LPIPS=%.4f R_APD=%.4f R_PSNR=%.4f R_SSIM=%.4f R_LPIPS=%.4f R_APD_=%.4f Diff_bits=%03d" % (
            H_diffs.avg, H_PSNRs.avg, H_SSIMs.avg, H_LPIPS.avg,
            R_diffs.avg, R_PSNRs.avg, R_SSIMs.avg, R_LPIPS.avg,
            R_diffs_.avg, Diff_bits.avg
        )
    else:
        log = "Validation Epoch[%02d]\tSumloss=%.6f Hloss=%.6f Rloss=%.6f Rloss_=%.6f Hdiff=%.4f Rdiff=%.4f Rdiff_=%.4f Diff_bits=%03d" % (
            epoch, Sum_losses.avg,
            H_losses.avg, R_losses.avg, R_losses_.avg,
            H_diffs.avg, R_diffs.avg, R_diffs_.avg, Diff_bits.avg
        )
    print_log(log)
    print("#### %s end ####\n" % mode)
    return Sum_losses.avg, H_losses.avg, R_losses.avg, R_losses_.avg, H_diffs.avg, R_diffs.avg, R_diffs_.avg
