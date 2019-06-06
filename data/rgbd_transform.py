import torch
import numpy as np
import random
import scipy.ndimage as ndimage
import numbers

class ArrayToTensor(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""

    def __call__(self, array):
        assert (isinstance(array, np.ndarray))
        # handle numpy array
        try:
            tensor = torch.from_numpy(array).permute(2, 0, 1)
        except:
            tensor = torch.from_numpy(np.expand_dims(array, axis=2)).permute(2, 0, 1)
        # put it from HWC to CHW format
        return tensor.float()



class TensorCenterCrop(object):

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, inputs):
        _, h, w = inputs.shape

        th, tw = self.size
        x = int(round((w - tw) / 2.))
        y = int(round((h - th) / 2.))
        inputs = inputs[:, y: y + th, x: x + tw]
        return inputs

class Scale_Single(object):
    """ Rescales a single object, for example only the ground truth dpeth map """

    def __init__(self, size, order=2):
        self.size = size
        self.order = order

    def __call__(self, inputs):
        h, w = inputs.shape

        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return inputs

        if w < h:
            ratio = self.size / w
        else:
            ratio = self.size / h

        inputs = ndimage.interpolation.zoom(inputs, ratio, order=self.order)

        return inputs

class Scale_Depth_Tensor(object):
    """ Rescales a single object, for example only the ground truth dpeth map """

    def __init__(self, size, order=2):
        self.size = size
        self.order = order

    def __call__(self, inputs):
        h, w = inputs.shape

        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return inputs

        if w < h:
            ratio = self.size / w
        else:
            ratio = self.size / h

        inputs = ndimage.interpolation.zoom(inputs.numpy(), ratio, order=self.order)

        return torch.tensor(inputs)


class Scale(object):
    """ Rescales the inputs and target arrays to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation order: Default: 2 (bilinear)
    """

    def __init__(self, size, order=2):
        self.size = size
        self.order = order

    def __call__(self, inputs, target_depth=None, target_label=None):
        h, w, _ = inputs.shape

        if (w <= h and w == self.size) or (h <= w and h == self.size):
            if target_depth is not None and target_label is not None:
                return inputs, target_depth, target_label
            elif target_depth is not None:
                return inputs, target_depth
            elif target_label is not None:
                return inputs, target_label

        if w < h:
            ratio = self.size / w
        else:
            ratio = self.size / h

        inputs = np.stack((ndimage.interpolation.zoom(inputs[:, :, 0], ratio, order=self.order),
                           ndimage.interpolation.zoom(inputs[:, :, 1], ratio, order=self.order), \
                           ndimage.interpolation.zoom(inputs[:, :, 2], ratio, order=self.order)), axis=2)

        if target_label is not None and target_depth is not None:

            target_label = ndimage.interpolation.zoom(target_label, ratio, order=self.order)
            target_depth = ndimage.interpolation.zoom(target_depth, ratio, order=self.order)
            return inputs, target_depth, target_label

        elif target_depth is not None:
            target_depth = ndimage.interpolation.zoom(target_depth, ratio, order=self.order)
            return inputs, target_depth

        elif target_label is not None:
            target_label = ndimage.interpolation.zoom(target_label, ratio, order=self.order)
            return inputs, target_label

        else:
            return inputs


class RandomCropRGBD(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, sample):

        sample['rgb'], sample['depth'] = self._crop(sample['rgb'], sample['depth'], self.size)

        # print('RGBD cropped')
        # if 'center' in sample:
        #     h, w = self.dptsize
        #     for i in range(sample['center'].shape[0]):
        #         if sample['center'][i,0] > h or sample['center'][i,1] > w:
        #             sample['center'][i] = 0
        #             sample['ord'][i] = 0
        return sample

    def _crop(self, tensor, dpt, crop_size):
        _, h, w = tensor.shape
        th, tw = crop_size
        if w == tw and h == th:
            return tensor, dpt

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        # x1 = 0
        # y1 = 0
        return tensor[:, y1: y1 + th, x1: x1 + tw], dpt[:, y1: y1 + th, x1: x1 + tw]

class RandomHorizontalFlipRGBD(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __call__(self, sample):
        if random.random() < 1.5:
            sample['rgb'] = torch.flip(sample['rgb'], [2])
            sample['depth'] = torch.flip(sample['depth'], [2])
            # if 'center' in sample:
                # print(sample['depth'].shape)
                # _,_,w = sample['depth'].size()
                # print(sample['center'].shape)
                # sample['center'][:,1] = w - sample['center'][:,1]
            # print('RGBD flipped')
        return sample