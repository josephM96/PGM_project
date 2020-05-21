from torchvision import datasets, transforms
from base import BaseDataLoader

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        rescaling = lambda x: (x - .5) * 2.
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            rescaling
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class CIFAR10DataLoader(BaseDataLoader):
    """
    CIFAR10 data loading using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        rescaling = lambda x: (x - .5) * 2.
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            rescaling
        ])
        self.data_dir = data_dir
        self.dataset = datasets.CIFAR10(self.data_dir, train=training, download=True, transforms=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

# TODO test data_loader should have shuffled two different datasets.