import pathlib
import random
from tensorflow.python.keras import backend
from tensorflow.python.util import tf_inspect
from keras_preprocessing import image
from DeepLearning.CV.tools.ClassAug.ImageGenerator import DataAugmentation
from PIL import Image
import numpy as np
from DeepLearning.CV.tools.ClassAug.object_detection_2d_geometric import ResizeRandomInterp
import cv2
import math

######################################数据预处理函数#######################################
def PreDataForVGG(image): #vgg的数据预处理
    """
    :param image:图片数据
    :param label: 图片的标签
    :return: 处理后的数据和标签
    """
    image = image.astype(np.float32)
    #进行归一化处理
    image /= 127.5
    image -= 1.
    return image
##########################################################################################

#####################################封装自带数据增强######################################
#自定义数据增强模块（继承自带的模块，添加多标签分类）
class dataGenerator(image.ImageDataGenerator):
    #继承所有现有方法
    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 zca_epsilon=1e-6,
                 rotation_range=0,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 brightness_range=None,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 data_format=None,
                 validation_split=0.0,
                 dtype=None):
        if data_format is None:
            data_format = backend.image_data_format()
        kwargs = {}
        if 'dtype' in tf_inspect.getfullargspec(
                image.ImageDataGenerator.__init__)[0]:
            if dtype is None:
                dtype = backend.floatx()
            kwargs['dtype'] = dtype
        super(dataGenerator, self).__init__(
            featurewise_center=featurewise_center,
            samplewise_center=samplewise_center,
            featurewise_std_normalization=featurewise_std_normalization,
            samplewise_std_normalization=samplewise_std_normalization,
            zca_whitening=zca_whitening,
            zca_epsilon=zca_epsilon,
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            brightness_range=brightness_range,
            shear_range=shear_range,
            zoom_range=zoom_range,
            channel_shift_range=channel_shift_range,
            fill_mode=fill_mode,
            cval=cval,
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip,
            rescale=rescale,
            preprocessing_function=preprocessing_function,
            data_format=data_format,
            validation_split=validation_split,
            **kwargs)
        self.num_samples = 0     #为了获取数据的总条数
##########################################################################################
#=========================定义自己的实现方法===========================
    #自定义数据读入
    def readData(self, path, shuffle=True):
        print("读取数据")
        data_path = pathlib.Path(path)
        all_image_paths = list(data_path.glob('*/*'))  #读取所有的图片路径
        all_image_paths = [str(path) for path in all_image_paths]
        self.num_samples = len(all_image_paths)
        if shuffle:
            random.shuffle(all_image_paths)  #打乱所有的图片

        #设置数据标签
        label_names = sorted(item.name for item in data_path.glob('*/') if item.is_dir())
        label_to_index = dict((name, index) for index, name in enumerate(label_names))
        #为图片打标签
        all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                            for path in all_image_paths]

        #返回数据和标签
        return all_image_paths, all_image_labels, label_to_index

    #自定义增强方式
    def augment(self, isAug, path, batch, shuffle, funcForData, img_height=300, img_width=300):
        self.index = 0
        DataAugmen = DataAugmentation(img_height, img_width)
        image_paths, labels, label_to_index = self.readData(path)
        while True:
            batch_images = []
            batch_labels = []
            for i in range(batch):
                #读取数据
                Myimage = np.array(Image.open(image_paths[self.index]))
                if isAug:
                    Myimage = DataAugmen(Myimage)  #对数据进行增强
                else:
                    Myimage = ResizeRandomInterp(height=img_height,
                                         width=img_width,
                                         interpolation_modes=[cv2.INTER_NEAREST,
                                                              cv2.INTER_LINEAR,
                                                              cv2.INTER_CUBIC,
                                                              cv2.INTER_AREA,
                                                              cv2.INTER_LANCZOS4])(Myimage)

                # 对数据进行预处理
                Myimage = funcForData(Myimage)
                batch_images.append(Myimage)
                batch_labels.append(labels[self.index])
                self.index += 1

                if (self.index == len(image_paths)):
                    if shuffle:
                        # 重新打乱
                        state = np.random.get_state()
                        np.random.shuffle(image_paths)
                        np.random.set_state(state)
                        np.random.shuffle(labels)
                    self.index = 0
                    break
            batch_images = np.array(batch_images)
            batch_labels = np.array(batch_labels)
            yield batch_images, batch_labels
#######################################################################################
#############################添加根据比例切割数据集###############################
    def readDataScale(self, path, scale=0.1):
        print("读取数据")
        data_path = pathlib.Path(path)
        all_image_paths = list(data_path.glob('*/*'))  # 读取所有的图片路径
        all_image_paths = [str(path) for path in all_image_paths]

        random.shuffle(all_image_paths)  # 打乱所有的图片

        # 设置数据标签
        label_names = sorted(item.name for item in data_path.glob('*/') if item.is_dir())

        label_to_index = dict((name, index) for index, name in enumerate(label_names))

        all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                            for path in all_image_paths]

        train_image_paths = all_image_paths[:math.floor(len(all_image_paths) * (1 - scale))]
        train_iamge_labels = all_image_labels[:math.floor(len(all_image_labels) * (1 - scale))]

        test_image_paths = all_image_paths[math.floor(len(all_image_paths) * (1 - scale)):]
        test_iamge_labels = all_image_labels[math.floor(len(all_image_labels) * (1 - scale)):]
        # 返回数据和标签
        return train_image_paths, train_iamge_labels, \
               test_image_paths, test_iamge_labels

    def augmentScaleTrain(self, isAug, paths, labels, batch,
                          funcForData, img_height=300,
                          img_width=300, shuffle=True):
        index = 0
        DataAugmen = DataAugmentation(img_height, img_width)

        while True:
            batch_images = []
            batch_labels = []
            for i in range(batch):
                # 读取数据
                Myimage = np.array(Image.open(paths[index]))
                if isAug:
                    Myimage = DataAugmen(Myimage)  # 对数据进行增强
                else:
                    Myimage = ResizeRandomInterp(height=img_height,
                                                 width=img_width,
                                                 interpolation_modes=[cv2.INTER_NEAREST,
                                                                      cv2.INTER_LINEAR,
                                                                      cv2.INTER_CUBIC,
                                                                      cv2.INTER_AREA,
                                                                      cv2.INTER_LANCZOS4])(Myimage)

                # 对数据进行预处理
                Myimage = funcForData(Myimage)
                batch_images.append(Myimage)
                batch_labels.append(labels[index])
                index += 1

                if (index == len(paths)):
                    if shuffle:
                        # 重新打乱
                        print("打乱数据")
                        # 打乱数据
                        state = np.random.get_state()
                        np.random.shuffle(paths)
                        np.random.set_state(state)
                        np.random.shuffle(labels)
                    index = 0
                    break
            batch_images = np.array(batch_images)
            batch_labels = np.array(batch_labels)

            yield (batch_images, batch_labels)

    def augmentScaleTest(self, paths, labels, batch,
                         funcForData, img_height=300,
                         img_width=300):
        index = 0
        while True:
            batch_images = []
            batch_labels = []
            for i in range(batch):
                # 读取数据
                Myimage = np.array(Image.open(paths[index]))
                Myimage = ResizeRandomInterp(height=img_height,
                                             width=img_width,
                                             interpolation_modes=[cv2.INTER_NEAREST,
                                                                  cv2.INTER_LINEAR,
                                                                  cv2.INTER_CUBIC,
                                                                  cv2.INTER_AREA,
                                                                  cv2.INTER_LANCZOS4])(Myimage)

                # 对数据进行预处理
                Myimage = funcForData(Myimage)
                batch_images.append(Myimage)
                batch_labels.append(labels[index])
                index += 1

                if (index == len(paths)):
                    index = 0
                    break
            batch_images = np.array(batch_images)
            batch_labels = np.array(batch_labels)

            yield (batch_images, batch_labels)

#######################################################################################

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    '''
    path = '../../../../datasets/traindatas/flower_data/train'  # 读入数据的路径
    # ==============验证自己的方法===============
    datagen = dataGenerator()
    for i in range(100):
        for images, labels in datagen.augment(False, path, 10, False, PreDataForVGG):
        # images, labels = next(datagen.augment(False, path, 10, False, PreDataForVGG))
            for image in images:
                plt.imshow(image)
                plt.axis('off')  # 去掉坐标轴
                plt.title("image")  # 添加标题
                plt.show()
        break
    '''
####################################验证按比例分配数据方法#################################
    '''
    path = '../../../../datasets/traindatas/flowers/'
    datagen = dataGenerator()
    train_image_paths, train_iamge_labels, \
    test_image_paths, test_iamge_labels = datagen.readDataScale(path)
    g = datagen.augmentScaleTrain(False, train_image_paths,
                             train_iamge_labels, 20, PreDataForVGG)
    for i in range(200):
        for images, labels in datagen.augmentScaleTrain(False, train_image_paths,
                             train_iamge_labels, 20, PreDataForVGG):
            for image in images:
                plt.imshow(image)

                plt.axis('off')  # 去掉坐标轴
                plt.title("image")  # 添加标题
                plt.show()
    '''
#########################################验证集成的方法######################################
    path = '../../../../datasets/traindatas/flower_data/train'  # 读入数据的路径
    params = dict(  # rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        preprocessing_function=PreDataForVGG)  # 配置参数

    datagen = dataGenerator(**params)
    for i in range(10):
        for images, labels in datagen.flow_from_directory(path, batch_size=5, target_size=(300, 300)):
            for image in images:
                plt.imshow(image)

                plt.axis('off')  # 去掉坐标轴
                plt.title("image")  # 添加标题
                plt.show()