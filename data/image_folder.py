import os
from PIL import Image
from torch.utils import data


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    """Judge a file is an image file or not."""
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    """Open the image in `path` as RGB form."""
    return Image.open(path).convert('RGB')


def make_dataset(dir, max_dataset_size=float('inf')):
    """Get `max_dataset_size` image file paths at most in `dir` and its subdir."""
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


class ImageFolder(data.Dataset):
    """A modified ImageFolder class.
    
    Modified by the official PyTorch ImageFolder class (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
    so that this class can load images from BOTH current directory AND its subdirectories.
    """
    def __init__(self, root, max_dataset_size=float('inf'), transform=None, return_paths=False, loader=default_loader):
        images = make_dataset(root, max_dataset_size)
        if len(images) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))
        
        self.root = root
        self.images = images
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.images[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        if self.return_paths:
            return image, path
        else:
            return image
    
    def __len__(self):
        return len(self.images)
