# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Augment operators."""

# import functools
# import random

# import tensorflow as tf
import functools
import random
import numpy as np
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import torch

# import simclr.data_util as simclr_ops


def base_augment(is_training=True, **kwargs):
    """Applies base (random resize and crop) augmentation."""
    size, crop_size = kwargs['size'], int(0.875 * kwargs['size'])
    if is_training:
        return [
            ('resize', {
                'size': size
            }),
            ('crop', {
                'size': crop_size
            }),
        ]
    return [('resize', {'size': size})]


def resize_augment(is_training=True, **kwargs):
    """Applies resize augmentation."""
    del is_training
    size = kwargs['size']
    return [('resize', {'size': size})]


def crop_augment(is_training=True, **kwargs):
    """Applies resize and random crop augmentation."""
    size, crop_size = kwargs['size'], kwargs['crop_size']
    if is_training:
        return [
            ('resize', {
                'size': size
            }),
            ('crop', {
                'size': crop_size
            }),
        ]
    return [('resize', {'size': size})]


def shift_augment(is_training=True, **kwargs):
    """Applies resize and random shift augmentation."""
    size, pad_size = kwargs['size'], int(0.125 * kwargs['size'])
    if is_training:
        return [
            ('resize', {
                'size': size
            }),
            ('shift', {
                'pad': pad_size
            }),
        ]
    return [('resize', {'size': size})]


def crop_and_resize_augment(is_training=True, **kwargs):
    """Applies random crop and resize augmentation."""
    size = kwargs['size']
    min_scale = kwargs['min_scale'] if 'min_scale' in kwargs else 0.5
    if is_training:
        return [
            ('crop_and_resize', {
                'size': size,
                'min_scale': min_scale
            }),
        ]
    return [('resize', {'size': size})]


def hflip_augment(is_training=True, **kwargs):
    """Applies random horizontal flip."""
    del kwargs
    if is_training:
        return [('hflip', {})]
    return []


def vflip_augment(is_training=True, **kwargs):
    """Applies random vertical flip."""
    del kwargs
    if is_training:
        return [('vflip', {})]
    return []


def rotate90_augment(is_training=True, **kwargs):
    """Applies rotation by 90 degree."""
    del kwargs
    if is_training:
        return [('rotate90', {})]
    return []


def rotate180_augment(is_training=True, **kwargs):
    """Applies rotation by 180 degree."""
    del kwargs
    if is_training:
        return [('rotate180', {})]
    return []


def rotate270_augment(is_training=True, **kwargs):
    """Applies rotation by 270 degree."""
    del kwargs
    if is_training:
        return [('rotate270', {})]
    return []


def jitter_augment(is_training=True, **kwargs):
    """Applies random color jitter augmentation."""
    if is_training:
        brightness = kwargs['brightness'] if 'brightness' in kwargs else 0.125
        contrast = kwargs['contrast'] if 'contrast' in kwargs else 0.4
        saturation = kwargs['saturation'] if 'saturation' in kwargs else 0.4
        hue = kwargs['hue'] if 'hue' in kwargs else 0
        return [('jitter', {
            'brightness': brightness,
            'contrast': contrast,
            'saturation': saturation,
            'hue': hue
        })]
    return []


def gray_augment(is_training=True, **kwargs):
    """Applies random grayscale augmentation."""
    if is_training:
        prob = kwargs['prob'] if 'prob' in kwargs else 0.2
        return [('gray', {'prob': prob})]
    return []


def blur_augment(is_training=True, **kwargs):
    """Applies random blur augmentation."""
    if is_training:
        prob = kwargs['prob'] if 'prob' in kwargs else 0.5
        return [('blur', {'prob': prob})]
    return []


class Resize(object):
    """Applies resize."""
    def __init__(self, size, method=InterpolationMode.BILINEAR):
        self.size = self._check_input(size)
        self.method = method

    def _check_input(self, size):
        if isinstance(size, int):
            size = (size, size)
        elif isinstance(size, (list, tuple)) and len(size) == 1:
            size = size * 2
        else:
            raise TypeError(
                'size must be an integer or list/tuple of integers')
        return size

    def __call__(self, image, is_training=True):
        return transforms.Resize(self.size, self.method)(image) if is_training else image
        # return tf.image.resize(image, self.size, method=self.method) if is_training else image


class RandomCrop(object):
    """Applies random crop without padding."""
    def __init__(self, size):
        self.size = self._check_input(size)

    def _check_input(self, size):
        """Checks input size is valid."""
        if isinstance(size, int):
            size = (3, size, size)
        elif isinstance(size, (list, tuple)):
            if len(size) == 1:
                size = (3, ) + tuple(size) * 2
            elif len(size) == 2:
                size = (3, ) + tuple(size)
        else:
            raise TypeError(
                'size must be an integer or list/tuple of integers')
        return size

    def __call__(self, image, is_training=True):
        return transforms.RandomCrop(self.size)(image) if is_training else image
        # return tf.image.random_crop(image, self.size) if is_training else image


class RandomShift(object):
    """Applies random shift."""
    def __init__(self, pad):
        self.pad = self._check_input(pad)

    def _check_input(self, size):
        """Checks input size is valid."""
        if isinstance(size, int):
            size = (size, size)
        elif isinstance(size, (list, tuple)):
            if len(size) == 1:
                size = tuple(size) * 2
            elif len(size) > 2:
                size = tuple(size[1:])
        else:
            raise TypeError(
                'size must be an integer or list/tuple of integers')
        return size

    def __call__(self, image, is_training=True):
        if is_training:
            img_size = image.shape[-3:]
            # image = tf.pad(image,
            #                [[self.pad[0]] * 2, [self.pad[1]] * 2, [0] * 2],
            #                mode='REFLECT')
            # image = tf.image.random_crop(image, img_size)
            image = transforms.Pad([self.pad[0], self.pad[1]], padding_mode='reflect')(image)
            image = transforms.RandomCrop(img_size)(image)
            
        return image


class RandomCropAndResize(object):
    """Applies random crop and resize."""
    def __init__(self, size, min_scale=0.5):
        self.min_scale = min_scale
        self.size = self._check_input(size)

    def _check_input(self, size):
        """Checks input size is valid."""
        if isinstance(size, int):
            size = (size, size)
        elif isinstance(size, (list, tuple)) and len(size) == 1:
            size = size * 2
        else:
            raise TypeError(
                'size must be an integer or list/tuple of integers')
        return size

    def __call__(self, image, is_training=True):
        if is_training:
            # crop and resize
            width = int(np.ceil(np.random.uniform(low=image.size(-1) * self.min_scale, high=image.size(-1))))
            # # width = tf.random.uniform(shape=[],
            # #                           minval=tf.cast(image.shape[0] *
            # #                                          self.min_scale,
            # #                                          dtype=tf.int32),
            # #                           maxval=image.shape[0] + 1,
            # #                           dtype=tf.int32)
            # size = (width, np.minimum(width, image.size(-1)))  # tf.minimum(width, image.shape[1]), image.shape[2])
            # # print("size: ", size)
            # print("width: ", width)
            image = transforms.RandomCrop(size=(width, width))(image)
            image = transforms.Resize(self.size)(image)
            # image = transforms.RandomCrop(size)(image)
            # image = tf.image.random_crop(image, size)
            # # image = tf.image.resize(image, size=self.size)
            # image = transforms.Resize(self.size)(image)
        return image    
        # return transforms.RandomResizedCrop(self.size, scale=(self.min_scale, 1))(image) if is_training else image
            # transforms.RandomResizedCrop
        # return image


class RandomFlipLeftRight(object):
    """Applies random horizontal flip."""
    def __init__(self):
        pass

    def __call__(self, image, is_training=True):
        return transforms.RandomHorizontalFlip()(image) if is_training else image
        # return tf.image.random_flip_left_right(image) if is_training else image


class RandomFlipUpDown(object):
    """Applies random vertical flip."""
    def __init__(self):
        pass

    def __call__(self, image, is_training=True):
        # return tf.image.random_flip_up_down(image) if is_training else image
        return transforms.RandomVerticalFlip()(image) if is_training else image


class Rotate90(object):
    """Applies rotation by 90 degree."""
    def __init__(self):
        pass

    def __call__(self, image, is_training=True):
        return torch.rot90(image, k=1, dims=(-2, -1)) if is_training else image
        # return tf.image.rot90(image, k=1) if is_training else image


class Rotate180(object):
    """Applies rotation by 180 degree."""
    def __init__(self):
        pass

    def __call__(self, image, is_training=True):
        return torch.rot90(image, k=2, dims=(-2, -1)) if is_training else image
        # return tf.image.rot90(image, k=2) if is_training else image


class Rotate270(object):
    """Applies rotation by 270 degree."""
    def __init__(self):
        pass

    def __call__(self, image, is_training=True):
        return torch.rot90(image, k=3, dims=(-2, -1)) if is_training else image
        
        # return tf.image.rot90(image, k=3) if is_training else image


# class ColorJitter(object):
#     """Applies random color jittering."""
#     def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
#         self.brightness = brightness
#         self.contrast = contrast
#         self.saturation = saturation
#         self.hue = hue
        
#     def __call__(self, image, is_training=True):
    # transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)(image) if is_training else image

"""
Pytorch implementation of RGB convert to HSV, and HSV convert to RGB,
RGB or HSV's shape: (B * C * H * W)
RGB or HSV's range: [0, 1)
"""
class RGB_HSV(object):
    def __init__(self, eps=0):
        super(RGB_HSV, self).__init__()
        self.eps = eps

    def rgb_to_hsv(self, img):

        hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(img.device)

        hue[ img[:,2]==img.max(1)[0] ] = 4.0 + ( (img[:,0]-img[:,1]) / ( img.max(1)[0] - img.min(1)[0] + self.eps) ) [ img[:,2]==img.max(1)[0] ]
        hue[ img[:,1]==img.max(1)[0] ] = 2.0 + ( (img[:,2]-img[:,0]) / ( img.max(1)[0] - img.min(1)[0] + self.eps) ) [ img[:,1]==img.max(1)[0] ]
        hue[ img[:,0]==img.max(1)[0] ] = (0.0 + ( (img[:,1]-img[:,2]) / ( img.max(1)[0] - img.min(1)[0] + self.eps) ) [ img[:,0]==img.max(1)[0] ]) % 6

        hue[img.min(1)[0]==img.max(1)[0]] = 0.0
        hue = hue/6

        saturation = ( img.max(1)[0] - img.min(1)[0] ) / ( img.max(1)[0] + self.eps )
        saturation[ img.max(1)[0]==0 ] = 0

        value = img.max(1)[0]
        
        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)
        hsv = torch.cat([hue, saturation, value],dim=1)
        return hsv

    def hsv_to_rgb(self, hsv):
        h,s,v = hsv[:,0,:,:],hsv[:,1,:,:],hsv[:,2,:,:]
        #对出界值的处理
        h = h%1
        s = torch.clamp(s,0,1)
        v = torch.clamp(v,0,1)
  
        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)
        
        hi = torch.floor(h * 6)
        f = h * 6 - hi
        p = v * (1 - s)
        q = v * (1 - (f * s))
        t = v * (1 - ((1 - f) * s))
        
        hi0 = hi==0
        hi1 = hi==1
        hi2 = hi==2
        hi3 = hi==3
        hi4 = hi==4
        hi5 = hi==5
        
        r[hi0] = v[hi0]
        g[hi0] = t[hi0]
        b[hi0] = p[hi0]
        
        r[hi1] = q[hi1]
        g[hi1] = v[hi1]
        b[hi1] = p[hi1]
        
        r[hi2] = p[hi2]
        g[hi2] = v[hi2]
        b[hi2] = t[hi2]
        
        r[hi3] = p[hi3]
        g[hi3] = q[hi3]
        b[hi3] = v[hi3]
        
        r[hi4] = t[hi4]
        g[hi4] = p[hi4]
        b[hi4] = v[hi4]
        
        r[hi5] = v[hi5]
        g[hi5] = p[hi5]
        b[hi5] = q[hi5]
        
        r = r.unsqueeze(1)
        g = g.unsqueeze(1)
        b = b.unsqueeze(1)
        rgb = torch.cat([r, g, b], dim=1)
        return rgb


class ColorJitter(object):
    """Applies random color jittering."""
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness)
        self.contrast = self._check_input(contrast, center=1)
        self.saturation = self._check_input(saturation, center=1)
        self.hue = self._check_input(hue, bound=0.5)
        self.rgb_hsv = RGB_HSV()

    def _check_input(self, value, center=None, bound=None):
        if bound is not None:
            value = min(value, bound)
        if center is not None:
            value = [center - value, center + value]
            if value[0] == value[1] == center:
                return None
        elif value == 0:
            return None
        return value
    
    def random_bightness(self, image):
        delta = np.random.uniform(low=-self.brightness, high=self.brightness)
        return image + delta
    
    def random_contrast(self, image):
        contrast_factor = np.random.uniform(low=self.contrast[0], high=self.contrast[1])
        mean = torch.mean(image, dim=(-2, -1), keepdim=True)
        return (image-mean) * contrast_factor + mean
    
    def random_saturation(self, image):
        saturation_factor = np.random.uniform(low=self.saturation[0], high=self.saturation[1])
        size = image.size()
        c, h, w = size[-3:]
        image = self.rgb_hsv.rgb_to_hsv(image.reshape(-1, c, h, w))
        image[:, 1, :, :] = image[:, 1, :, :] * saturation_factor
        image = self.rgb_hsv.hsv_to_rgb(image)
        return image.reshape(size)
        
    def random_hue(self, image):
        delta = np.random.uniform(low=-self.hue, high=self.hue)
        size = image.size()
        c, h, w = size[-3:]
        image = self.rgb_hsv.rgb_to_hsv(image.reshape(-1, c, h, w))
        image[:, 0, :, :] = image[:, 1, :, :] + delta
        image = self.rgb_hsv.hsv_to_rgb(image)
        return image.reshape(size)
        
    def __get_transforms(self):
        transform_list = []
        if self.brightness is not None:
            transform_list.append(functools.partial(self.random_bightness))
        if self.contrast is not None:
            transform_list.append(functools.partial(self.random_contrast))
        if self.saturation is not None:
            transform_list.append(functools.partial(self.random_saturation))
        if self.hue is not None:
            transform_list.append(functools.partial(self.random_hue))
        random.shuffle(transform_list)
        return transform_list
        
    def __call__(self, image, is_training=True):
        if is_training:
            transform_list = self.__get_transforms()
            for transform in transform_list:
                image = transform(image)
        return image
    
# class ColorJitter(object):
#     """Applies random color jittering."""
#     def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
#         self.brightness = self._check_input(brightness)
#         self.contrast = self._check_input(contrast, center=1)
#         self.saturation = self._check_input(saturation, center=1)
#         self.hue = self._check_input(hue, bound=0.5)

#     def _check_input(self, value, center=None, bound=None):
#         if bound is not None:
#             value = min(value, bound)
#         if center is not None:
#             value = [center - value, center + value]
#             if value[0] == value[1] == center:
#                 return None
#         elif value == 0:
#             return None
#         return value

#     def _get_transforms(self):
#         """Gets a randomly ordered sequence of color transformation."""
#         transforms = []
#         if self.brightness is not None:
#             transforms.append(
#                 functools.partial(tf.image.random_brightness,
#                                   max_delta=self.brightness))
#         if self.contrast is not None:
#             transforms.append(
#                 functools.partial(tf.image.random_contrast,
#                                   lower=self.contrast[0],
#                                   upper=self.contrast[1]))
#         if self.saturation is not None:
#             transforms.append(
#                 functools.partial(tf.image.random_saturation,
#                                   lower=self.saturation[0],
#                                   upper=self.saturation[1]))
#         if self.hue is not None:
#             transforms.append(
#                 functools.partial(tf.image.random_hue, max_delta=self.hue))
#         random.shuffle(transforms)
#         return transforms

#     def __call__(self, image, is_training=True):
#         if not is_training:
#             return image
#         num_concat = image.shape[2] // 3
#         if num_concat == 1:
#             for transform in self._get_transforms():
#                 image = transform(image)
#         else:
#             images = tf.split(image, num_concat, axis=-1)
#             for transform in self._get_transforms():
#                 images = [transform(image) for image in images]
#             image = tf.concat(images, axis=-1)
#         return image


class RandomGrayScale(object):
    def __init__(self, prob):
        self.prob = prob
    
    def __call__(self, image, is_training=True):
        return transforms.RandomGrayscale(self.prob)(image) if is_training else image


# class RandomGrayScale(object):
#     """Applies random grayscale augmentation."""
#     def __init__(self, prob):
#         self.prob = prob

#     def __call__(self, image, is_training=True):
#         return tf.cond(
#             tf.random.uniform([]) > self.prob,
#             lambda: image, lambda: tf.image.grayscale_to_rgb(
#                 tf.image.rgb_to_grayscale(image)))

class RandomBlur(object):
    
    def __init__(self, prob):
        self.prob = prob
    
    def __call__(self, image, is_training=True):
        if is_training and np.random.uniform(0, 1) < self.prob:
            kernel_size = (image.shape[-1] // 10) // 2 * 2 + 1
            return transforms.GaussianBlur(kernel_size=kernel_size)(image) if is_training else image
        else:
            return image


# class RandomBlur(object):
#     """Applies random blur augmentation."""
#     def __init__(self, prob=0.5):
#         self.prob = prob

#     def __call__(self, image, is_training=True):
#         if is_training:
#             return image
#         return simclr_ops.random_blur(image,
#                                       image.shape[0],
#                                       image.shape[1],
#                                       p=self.prob)
