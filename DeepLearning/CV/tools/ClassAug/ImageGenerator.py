"""
进行数据增广的整理流程
"""
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import inspect

from DeepLearning.CV.tools.ClassAug.object_detection_2d_photometric import ConvertColor, \
    ConvertDataType, ConvertTo3Channels, \
    RandomBrightness, RandomContrast, RandomHue, RandomSaturation, RandomChannelSwap
from DeepLearning.CV.tools.ClassAug.object_detection_2d_geometric import ResizeRandomInterp, RandomFlip

class PhotometricDistortions:
    '''
    执行“ train_transform_param”指令定义的光学变换。
    '''

    def __init__(self):

        self.convert_RGB_to_HSV = ConvertColor(current='RGB', to='HSV')
        self.convert_HSV_to_RGB = ConvertColor(current='HSV', to='RGB')
        self.convert_to_float32 = ConvertDataType(to='float32')
        self.convert_to_uint8 = ConvertDataType(to='uint8')
        self.convert_to_3_channels = ConvertTo3Channels()
        self.random_brightness = RandomBrightness(lower=-32, upper=32, prob=0.5)
        self.random_contrast = RandomContrast(lower=0.5, upper=1.5, prob=0.5)
        self.random_saturation = RandomSaturation(lower=0.5, upper=1.5, prob=0.5)
        self.random_hue = RandomHue(max_delta=18, prob=0.5)
        self.random_channel_swap = RandomChannelSwap(prob=0.0)

        self.sequence1 = [self.convert_to_3_channels,
                          self.convert_to_float32,
                          self.random_brightness,
                          self.random_contrast,
                          self.convert_to_uint8,
                          self.convert_RGB_to_HSV,
                          self.convert_to_float32,
                          self.random_saturation,
                          self.random_hue,
                          self.convert_to_uint8,
                          self.convert_HSV_to_RGB,
                          self.random_channel_swap]

        self.sequence2 = [self.convert_to_3_channels,
                          self.convert_to_float32,
                          self.random_brightness,
                          self.convert_to_uint8,
                          self.convert_RGB_to_HSV,
                          self.convert_to_float32,
                          self.random_saturation,
                          self.random_hue,
                          self.convert_to_uint8,
                          self.convert_HSV_to_RGB,
                          self.convert_to_float32,
                          self.random_contrast,
                          self.convert_to_uint8,
                          self.random_channel_swap]

    def __call__(self, image):
        if np.random.choice(2):

            for transform in self.sequence1:
                image = transform(image)
            return image
        else:

            for transform in self.sequence2:
                image = transform(image)
            return image

class DataAugmentation:
    '''
    实现的数据增强管道。
    '''

    def __init__(self,
                 img_height=300,
                 img_width=300):
        '''
        Arguments:
            height (int): 输出图像的期望高度（以像素为单位）。
            width (int): 输出图像的期望宽度（以像素为单位）。
            background (list/tuple, optional): 一个三元组，指定转换图像的背景像素的RGB颜色值。
        '''


        self.photometric_distortions = PhotometricDistortions()
        self.random_flip = RandomFlip(dim='horizontal', prob=0.5)

        self.resize = ResizeRandomInterp(height=img_height,
                                         width=img_width,
                                         interpolation_modes=[cv2.INTER_NEAREST,
                                                              cv2.INTER_LINEAR,
                                                              cv2.INTER_CUBIC,
                                                              cv2.INTER_AREA,
                                                              cv2.INTER_LANCZOS4])

        self.sequence = [self.photometric_distortions,
                         self.random_flip,
                         self.resize]

    def __call__(self, image, return_inverter=False):

        inverters = []

        for transform in self.sequence:
            if return_inverter and ('return_inverter' in inspect.signature(transform).parameters):
                image, inverter = transform(image, return_inverter=True)
                inverters.append(inverter)
            else:
                image = transform(image)

        if return_inverter:
            return image, inverters[::-1]
        else:
            return image

if __name__ == '__main__':
    #进行数据增广,并设定循环次数
    for i in range(10):
        img = np.array(Image.open('datas/2007_000027.jpg'))

        try:
            #==================显示原图=====================
            plt.subplot(1, 2, 1)
            plt.imshow(img)

            plt.axis('off')     #去掉坐标轴
            plt.title("original image")   #添加标题
            #==============================================

            #==================显示增广图==================
            plt.subplot(1, 2, 2)
            DataAugmen = DataAugmentation()
            img = DataAugmen(img)

            plt.imshow(img)

            plt.axis('off')
            plt.title("augment image")

            plt.show()
        except Exception:
            continue