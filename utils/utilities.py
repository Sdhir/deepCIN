import logging
from typing import Optional
import torch
from torch.utils.data.dataset import Dataset  # For custom datasets
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

from albumentations import Compose
from albumentations.augmentations import transforms as ab_transforms
from albumentations.pytorch import ToTensor


def initialize_logger(output_file):
    '''
    The function create a logger to log into the console and a file
    it takes one parameter:
        output_file: The full path to a file to store all log messages

    In the main code:
        from utilities import initialize_logger
        initialize_logger('/path/to/log/file.log')
        # then use it as follow
        logging.debug("debug message")
        logging.info("info message")
        logging.warning("warning message")
        logging.error("error message")
        logging.critical("critical message")

    '''
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # create console handler and set level to info
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    # "%(asctime)s - %(levelname)s - %(message)s',
    # datefmt='%m/%d/%Y %I:%M:%S %p"
    formatter = logging.Formatter("(%(levelname)s) %(asctime)s : %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # create error file handler and set level to error
    handler = logging.FileHandler(output_file, "w", encoding=None, delay="true")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def one_hot(labels: torch.Tensor,
            num_classes: int,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            eps: Optional[float] = 1e-6) -> torch.Tensor:
    r"""Converts an integer label 2D tensor to a one-hot 3D tensor.

    Args:
        labels (torch.Tensor) : tensor with labels of shape :math:`(N, H, W)`,
                                where N is batch siz. Each value is an integer
                                representing correct classification.
        num_classes (int): number of classes in labels.
        device (Optional[torch.device]): the desired device of returned tensor.
         Default: if None, uses the current device for the default tensor type
         (see torch.set_default_tensor_type()). device will be the CPU for CPU
         tensor types and the current CUDA device for CUDA tensor types.
        dtype (Optional[torch.dtype]): the desired data type of returned
         tensor. Default: if None, infers data type from values.

    Returns:
        torch.Tensor: the labels in one hot tensor.

    Examples::
        >>> labels = torch.LongTensor([[[0, 1], [2, 0]]])
        >>> kornia.losses.one_hot(labels, num_classes=3)
        tensor([[[[1., 0.],
                  [0., 1.]],
                 [[0., 1.],
                  [0., 0.]],
                 [[0., 0.],
                  [1., 0.]]]]
    """
    if not torch.is_tensor(labels):
        raise TypeError("Input labels type is not a torch.Tensor. Got {}"
                        .format(type(labels)))
    if not len(labels.shape) == 3:
        raise ValueError("Invalid depth shape, we expect BxHxW. Got: {}"
                         .format(labels.shape))
    if not labels.dtype == torch.int64:
        raise ValueError(
            "labels must be of the same dtype torch.int64. Got: {}" .format(
                labels.dtype))
    if num_classes < 1:
        raise ValueError("The number of classes must be bigger than one."
                         " Got: {}".format(num_classes))
    batch_size, height, width = labels.shape
    one_hot = torch.zeros(batch_size, num_classes, height, width,
                          device=device, dtype=dtype)
    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps

def matrix_mul(input, weight, bias=False):
    feature_list = []
    for feature in input:
        feature = torch.mm(feature, weight)
        if isinstance(bias, torch.nn.parameter.Parameter):
            feature = feature + bias.expand(feature.size()[0], bias.size()[1])
        feature = torch.tanh(feature).unsqueeze(0)
        feature_list.append(feature)

    return torch.cat(feature_list, 0).squeeze()

def element_wise_mul(input1, input2):

    feature_list = []
    for feature_1, feature_2 in zip(input1, input2):
        feature_2 = feature_2.unsqueeze(1).expand_as(feature_1)
        feature = feature_1 * feature_2
        feature_list.append(feature.unsqueeze(0))
    output = torch.cat(feature_list, 0)

    return torch.sum(output, 0).unsqueeze(0)

#%% with albumentations

class CustomDataset(Dataset):
    def __init__(self, im_paths=None, labels=None, phase=None, resize=False):
        """
        Args:
            im_paths (numpy): image_data
            y (numpy): label data
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.im_paths = im_paths
        self.labels = labels
        self.resize = resize
        self.albumentations_transform = {
                'train': Compose([
                    ab_transforms.HorizontalFlip(p=0.2),
                    ab_transforms.VerticalFlip(p=0.2),
                    ab_transforms.Rotate(limit=180,p=0.2),
                    ab_transforms.HueSaturationValue(p=0.1),
                    ab_transforms.RandomContrast(p=0.1),
                    ab_transforms.GaussianBlur(blur_limit=3, p=0.2),
                    ab_transforms.GaussNoise(p=0.05),
                    ab_transforms.CLAHE(p=0.2),
                    ab_transforms.Normalize(mean=[0.5944, 0.4343, 0.5652],std=[0.2593,0.2293,0.2377]),
                    ToTensor()
                ]),
                'val': Compose([
                    ab_transforms.Normalize(mean=[0.5944, 0.4343, 0.5652],std=[0.2593,0.2293,0.2377]),
                    ToTensor()
                ]),
        }
        if phase == 'train':
            self.transform = self.albumentations_transform['train']
        else:
            self.transform = self.albumentations_transform['val']
    
    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, index):
        single_image_label = self.labels[index] 
        # Read image from DIR to PIL image
        img_as_bgr = cv2.imread(self.im_paths[index])
        if self.resize: #resize
            height = 704
            width = 64
            img_as_bgr = cv2.resize(img_as_bgr,(width,height), interpolation = cv2.INTER_CUBIC)
        img_as_rgb = cv2.cvtColor(img_as_bgr, cv2.COLOR_BGR2RGB)
        img_as_rgb = np.rot90(img_as_rgb)
        # Transform image to tensor
        if self.transform is not None:
            # Apply transformations
            augmented  = self.transform(image=img_as_rgb)
            # Convert numpy array to PIL Image
            img_as_tensor = augmented['image']
        # Return image and the label
        return (img_as_tensor, single_image_label)


#%% with albumentations for Image level classification
albumentations_transform_im = {
        'train': Compose([
            ab_transforms.Normalize(mean=[0.5944, 0.4343, 0.5652],std=[0.2593,0.2293,0.2377]),
            ToTensor()
        ]),
        'val': Compose([
            ab_transforms.Normalize(mean=[0.5944, 0.4343, 0.5652],std=[0.2593,0.2293,0.2377]),
            ToTensor()
        ]),
}
class Img_CustomDataset(Dataset):
    def __init__(self, im_paths, labels, phase=None):
        """
        Args:
            im_paths (list): image_data
            y (list): label data
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.im_paths = im_paths
        self.labels = labels
        if phase == 'train':
            self.transform = albumentations_transform_im['train']
        else:
            self.transform = albumentations_transform_im['val']
    
    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, index):
        single_image_label = self.labels[index] 
        # Read image from DIR to PIL image
        for i,vs_dir in enumerate(self.im_paths[index]):
            vs_as_bgr = cv2.imread(vs_dir)
            vs_as_rgb = cv2.cvtColor(vs_as_bgr, cv2.COLOR_BGR2RGB)
            vs_as_rgb = np.rot90(vs_as_rgb)
            # Apply transformations
            augmented  = self.transform(image=vs_as_rgb)
            # Convert numpy array to PIL Image
            vs_as_tensor = augmented['image']
            vs_as_tensor = vs_as_tensor.unsqueeze(0)
            if not i:
                img_as_tensor = vs_as_tensor
            else:
                img_as_tensor = torch.cat((img_as_tensor, vs_as_tensor), 0)
        # Return image and the label
        return (img_as_tensor, single_image_label)
