from torchvision import datasets, transforms
from torch.utils.data import Dataset
from base import BaseDataLoader
import torch


class FashionMNISTDataLoader(BaseDataLoader):
    """
    FashionMNIST data loading using BaseDataLoader

    :parameter
    mode : used for determining 'background model' training or 'semantic model' training
    """
    def __init__(self, data_dir, img_size, batch_size, mode='background', shuffle=True, validation_split=0.0, num_workers=1, training=True):
        if mode == 'background':
            mutate = FashionMNISTDataLoader.mutate
            rescaling = lambda x: (x - .5) * 2.
            trsfm = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                mutate,
                rescaling
            ])
        else:
            rescaling = lambda x: (x - .5) * 2.
            trsfm = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                rescaling
            ])

        self.data_dir = data_dir
        self.dataset = datasets.FashionMNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

    @staticmethod
    def mutate(img, corruption_rate=0.1):
        img = img * 255
        mask = (torch.rand_like(img) < corruption_rate).float()
        corruption = torch.randint(0, 255, img.shape).float()
        img = img * (1 - mask) + corruption * mask
        img = img / 255

        return img


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading using BaseDataLoader
    """
    def __init__(self, data_dir, img_size, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        rescaling = lambda x: (x - .5) * 2.
        trsfm = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            rescaling
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class CIFAR10DataLoader(BaseDataLoader):
    """
    CIFAR10 data loading using BaseDataLoader

    :parameter
    mode : used for determining 'background model' training or 'semantic model' training
    """
    def __init__(self, data_dir, batch_size, mode='background',shuffle=True, validation_split=0.0, num_workers=1, training=True):
        if mode =='background':
            mutate = CIFAR10DataLoader.mutate_x()
            rescaling = lambda x: (x - .5) * 2.
            trsfm = transforms.Compose([
                transforms.ToTensor(),
                mutate,
                rescaling
            ])
        else:
            rescaling = lambda x: (x - .5) * 2.
            trsfm = transforms.Compose([
                transforms.ToTensor(),
                rescaling
            ])

        self.data_dir = data_dir
        self.dataset = datasets.CIFAR10(self.data_dir, train=training, download=True, transforms=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

    @staticmethod
    def mutate_x(img, corruption_rate=0.1):
        img = img * 255
        mask = (torch.randn_like(img) < corruption_rate).float()
        corruption = torch.rand_int(0, 255, img.shape).float()
        img = img * mask + corruption * (1 - mask)
        img = img / 255

        return img


class SVHNDataLoader(BaseDataLoader):
    """
    SVHN data loading using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        rescaling = lambda x: (x - .5) * 2.
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            rescaling
        ])
        self.data_dir = data_dir
        self.dataset = datasets.SVHN(self.data_dir, train=training, download=True, transforms=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
