import torch
from torch import nn, optim
from torch.backends import cudnn

from utils.options import opt, device
from utils.util import create_dirs, print_network
from data.load_data import get_loaders
from models.tools import get_nets, get_scheduler, load_checkpoint, train, test


def main():
    if opt.gpu_ids != -1 and torch.cuda.is_available():
        print("---- GPU ----")
    else:
        print("---- CPU ----")
    
    cudnn.benchmark = True  # speed up

    print("Creating new dirs...")
    create_dirs()
    
    print("Making data loaders...")
    if not opt.test:
        (train_loader_cover, train_loader_secret),\
            (val_loader_cover, val_loader_secret) = get_loaders()
    else:
        (test_loader_cover, test_loader_secret) = get_loaders()

    print("Constructing nets...")
    HRnet, Enet = get_nets()

    if not opt.invertible:
        params = list(HRnet[0].parameters()) + list (HRnet[1].parameters())
    else:
        params = list(HRnet.parameters())
    if opt.redundance != -1:
        params += list(Enet.parameters())
    if not opt.invertible:
        optimizer = optim.Adam(params, lr=opt.lr, betas=(0.5, 0.999))
    else:
        optimizer = optim.Adamax(params, lr=opt.lr, betas=(0.9, 0.999))
    scheduler = get_scheduler(optimizer)

    if opt.checkpoint_name != '':
        print("Loading checkpoints...")
        if opt.continue_train:  # continue training; checkpoint_name == exper_name
            HRnet, Enet, optimizer, scheduler = load_checkpoint(HRnet, Enet, optimizer, scheduler)
        elif opt.test:  # test mode; checkpoint_name == exper_name
            HRnet, Enet, _, _ = load_checkpoint(HRnet, Enet)
        else:  # load weights from other well-trained networks
            HRnet, Enet, _, _ = load_checkpoint(HRnet, Enet)

    if opt.loss == 'l1':
        criterion = nn.L1Loss().to(device)
    if opt.loss == 'l2':
        criterion = nn.MSELoss().to(device)

    if not opt.test:
        if not opt.invertible:
            print_network(HRnet[0])
            print_network(HRnet[1])
        else:
            print_network(HRnet)
        if opt.redundance != -1:
            print_network(Enet)

        train(
            train_loader_cover, train_loader_secret,
            val_loader_cover, val_loader_secret,
            HRnet, Enet, criterion,
            optimizer, scheduler
        )
    else:
        test(
            test_loader_cover, test_loader_secret,
            HRnet, Enet, criterion,
            save_num=5, mode='test', epoch=None
        )


if __name__ == '__main__':
    main()
