import numpy as np
from .autograd import Tensor
import gzip
import struct
import logging

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        # 对每一行进行翻转
        # if flip_img:
        #     flipped_image = []
        #     for row in img:
        #         flipped_row = row[::-1]  # 反转当前行
        #         flipped_image.append(flipped_row)
        #     img = flipped_image
        # return img
        if flip_img:
            # 选中第一维所有，对每一行进行翻转，选中第三维所有
            img = img[:, ::-1, :]
        return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        ### BEGIN YOUR SOLUTION
        h,w,c = img.shape
        padded_img = np.zeros((h + 2*self.padding, w + 2*self.padding, c))

        padded_img[self.padding:self.padding+h, self.padding:self.padding+w, :] = img

        new_x, new_y = self.padding + shift_x, self.padding + shift_y
        crop_img = padded_img[new_x:new_x+h, new_y:new_y+w]
        return crop_img
        ### END YOUR SOLUTION


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                                           range(batch_size, len(dataset), batch_size))
        else:
            arr = np.arange(len(dataset))
            np.random.shuffle(arr)
            self.ordering = np.array_split(arr, range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        self.idx = -1
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        self.idx += 1
        if self.idx >= len(self.ordering):
            self.idx = -1
            raise StopIteration

        samples = []
        samples = self.dataset[self.ordering[self.idx]]
        samples = [Tensor(x) for x in samples]
        return samples
        ### END YOUR SOLUTION


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        self.image_filename = image_filename
        self.label_filename = label_filename

        with gzip.open(image_filename, 'rb') as f:
            # read first 16 bytes and interprets them as C struct values
            # '>' means big endian
            # 'I' character indicates that the next 4 bytes should be interpreted as an unsigned integer (4 bytes each).
            magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
            image_size = rows * cols
            # create ndarray from read buffer, convert into dtype=float32
            data = np.frombuffer(f.read(), dtype=np.uint8).astype(np.float32)
            # normalization uint8 ranges in [0,255], so divide values by 255.0
            data = data / 255.0
            data = data.reshape(num_images, image_size)

        with gzip.open(label_filename, 'rb') as f:
            magic, num_items = struct.unpack('>II', f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        
        data = data.reshape((-1, rows, cols, 1))

        self.data = data
        self.labels = labels
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        img = self.data[index]
        super().apply_transforms(img)
        return img, self.labels[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.data.shape[0]
        ### END YOUR SOLUTION

class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])
