import os
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.options import opt
from .image_folder import ImageFolder


def get_loaders(test=opt.test):
    """Return `(train_loader_cover, train_loader_secret), (val_loader_cover, val_loader_secret)` in training mode.
    Return `(test_loader_cover, test_loader_secret)` in test mode.
    """
    assert opt.image_size % 32 == 0, "Image size should be be divisible by 32!"

    assert opt.dataset_dir, "Dataset dir doesn't exist!"
    train_dir = os.path.join(opt.dataset_dir, 'train')
    val_dir = os.path.join(opt.dataset_dir, 'val')
    test_dir = os.path.join(opt.dataset_dir, 'test')

    transform_color = transforms.Compose([
        transforms.Resize([opt.image_size, opt.image_size]),
        transforms.ToTensor()
    ])
    transform_gray = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize([opt.image_size, opt.image_size]),
        transforms.ToTensor()
    ])
    if opt.channel_cover == 3:
        transform_cover = transform_color
    else:
        transform_cover = transform_gray
    if opt.channel_secret == 3:
        transform_secret = transform_color
    else:
        transform_secret = transform_gray

    if not test:
        # train
        train_dataset_cover = ImageFolder(train_dir, opt.dataset_size_train, transform_cover)
        train_dataset_secret = ImageFolder(train_dir, opt.dataset_size_train, transform_secret)
        for i in range(1, opt.num_secrets):
            train_dataset_secret += ImageFolder(train_dir, opt.dataset_size_train, transform_secret)
        
        train_loader_cover = DataLoader(
            train_dataset_cover,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.workers
        )
        train_loader_secret = DataLoader(
            train_dataset_secret,
            batch_size=opt.batch_size*opt.num_secrets,
            shuffle=True,
            num_workers=opt.workers
        )
        # val
        val_dataset_cover = ImageFolder(val_dir, opt.dataset_size_val, transform_cover)
        val_dataset_secret = ImageFolder(val_dir, opt.dataset_size_val, transform_secret)
        for i in range(1, opt.num_secrets):
            val_dataset_secret += ImageFolder(val_dir, opt.dataset_size_val, transform_secret)

        val_loader_cover = DataLoader(
            val_dataset_cover,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.workers
        )
        val_loader_secret = DataLoader(
            val_dataset_secret,
            batch_size=opt.batch_size*opt.num_secrets,
            shuffle=False,  # do not shuffle secret image when in val mode
            num_workers=opt.workers
        )
        return (train_loader_cover, train_loader_secret), (val_loader_cover, val_loader_secret)
    else:
        # test
        test_dataset_cover = ImageFolder(test_dir, opt.dataset_size_test, transform_cover)
        test_dataset_secret = ImageFolder(test_dir, opt.dataset_size_test, transform_secret)
        for i in range(1, opt.num_secrets):
            test_dataset_secret += ImageFolder(test_dir, opt.dataset_size_test, transform_secret)
        
        test_loader_cover = DataLoader(
            test_dataset_cover,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.workers
        )
        test_loader_secret = DataLoader(
            test_dataset_secret,
            batch_size=opt.batch_size*opt.num_secrets,
            shuffle=False,  # do not shuffle secret image when in test mode
            num_workers=opt.workers
        )
        return (test_loader_cover, test_loader_secret)
