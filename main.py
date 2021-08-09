import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.backends import cudnn

from utils.options import opt, device
from utils.util import create_dirs, print_network
from data.load_data import make_loaders
from models.utils import load_nets, load_checkpoints, train, inference


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
            (val_loader_cover, val_loader_secret) = make_loaders()
    else:
        (test_loader_cover, test_loader_secret) = make_loaders()

    print("Constructing nets...")
    Hnet, Rnet, Enet = load_nets()

    if opt.load_checkpoint:
        print("Loading checkpoints for nets...")
        load_checkpoints(Hnet, Rnet, Enet)

    if opt.loss == 'l1':
        criterion = nn.L1Loss().cuda()
    if opt.loss == 'l2':
        criterion = nn.MSELoss().cuda()

    if not opt.test:
        print_network(Hnet)
        print_network(Rnet)
        params = list(Hnet.parameters()) + list (Rnet.parameters())
        if opt.redundance != -1:
            print_network(Enet)
            params += list(Enet.parameters())
        optimizer = optim.Adam(params, lr=opt.lr, betas=(0.5, 0.999))
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=True)

        print("Begining training...")
        train(
            train_loader_cover, train_loader_secret,
            val_loader_cover, val_loader_secret,
            Hnet, Rnet, Enet, criterion,
            optimizer, scheduler
        )
    else:
        print("Begining test...")
        inference(
            test_loader_cover, test_loader_secret,
            Hnet, Rnet, Enet, criterion,
            save_num=1, mode='test', epoch=None
        )


if __name__ == '__main__':
    main()
