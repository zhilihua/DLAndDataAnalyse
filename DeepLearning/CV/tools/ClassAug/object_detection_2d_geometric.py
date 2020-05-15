from __future__ import division
import numpy as np
import cv2

class Resize:
    '''
    将图像调整为指定的高度和宽度（以像素为单位）。
    '''

    def __init__(self,
                 height,
                 width,
                 interpolation_mode=cv2.INTER_LINEAR):
        '''
        Arguments:
            height (int): 输出图像的期望高度（以像素为单位）。
            width (int): 输出图像的期望宽度（以像素为单位）。
            interpolation_mode (int, optional): 表示有效的OpenCV插值模式的整数。
        '''

        self.out_height = height
        self.out_width = width
        self.interpolation_mode = interpolation_mode

    def __call__(self, image, return_inverter=False):
        image = cv2.resize(image,
                           dsize=(self.out_width, self.out_height),
                           interpolation=self.interpolation_mode)

        return image

class ResizeRandomInterp:
    '''
    使用随机选择的插值模式将图像调整为特定的高度和宽度（以像素为单位）。
    '''

    def __init__(self,
                 height,
                 width,
                 interpolation_modes=[cv2.INTER_NEAREST,
                                      cv2.INTER_LINEAR,
                                      cv2.INTER_CUBIC,
                                      cv2.INTER_AREA,
                                      cv2.INTER_LANCZOS4]):
        '''
        Arguments:
            height (int): 输出图像的期望高度（以像素为单位）。
            width (int): 输出图像的期望宽度（以像素为单位）。
            interpolation_mode (int, optional): 表示有效的OpenCV插值模式的整数。
        '''
        if not (isinstance(interpolation_modes, (list, tuple))):
            raise ValueError("`interpolation_mode` must be a list or tuple.")
        self.height = height
        self.width = width
        self.interpolation_modes = interpolation_modes
        self.resize = Resize(height=self.height,
                             width=self.width)

    def __call__(self, image, return_inverter=False):
        self.resize.interpolation_mode = np.random.choice(self.interpolation_modes)

        return self.resize(image, return_inverter)

class Flip:
    '''
    水平或垂直翻转图像。
    '''
    def __init__(self,
                 dim='horizontal'):
        '''
        Arguments:
            dim (str, optional): 可以是“水平”和“垂直”之一。
        '''
        if not (dim in {'horizontal', 'vertical'}): raise ValueError("`dim` can be one of 'horizontal' and 'vertical'.")
        self.dim = dim

    def __call__(self, image, return_inverter=False):

        if self.dim == 'horizontal':
            image = image[:,::-1]
            return image

        else:
            image = image[::-1]
            return image

class RandomFlip:
    '''
    水平或垂直随机翻转图像。 随机性仅指图像是否被翻转。
    '''
    def __init__(self,
                 dim='horizontal',
                 prob=0.5):
        '''
        Arguments:
            dim (str, optional): 可以是“水平”和“垂直”之一。
            prob (float, optional): 执行此操作的概率`(1 - prob)`。
        '''
        self.dim = dim
        self.prob = prob
        self.flip = Flip(dim=self.dim)

    def __call__(self, image, labels=None):
        p = np.random.uniform(0, 1)
        if p >= (1.0-self.prob):
            return self.flip(image)
        else:
            return image