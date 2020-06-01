import os
from DeepLearning.CV.tools.SegAug import ImageGenerator
from PIL import Image
import random
import numpy as np

def PreDataForCaffe(image): #resnet的数据预处理,应用的是caffe模式
    """
    :param image:图片数据
    :param label: 图片的标签
    :return: 处理后的数据和标签
    """
    x = image.astype(np.float32)
    #进行归一化处理
    x = x[..., ::-1]
    mean = [103.939, 116.779, 123.68]
    std = None
    x[..., 0] -= mean[0]
    x[..., 1] -= mean[1]
    x[..., 2] -= mean[2]
    if std is not None:
        x[..., 0] /= std[0]
        x[..., 1] /= std[1]
        x[..., 2] /= std[2]
    return x

def PreDataForTF(image): #vgg的数据预处理,应用的是tf模式
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

def transform(image, mask, image_w, image_h, is_aug):
    #图片尺寸调整
    image = np.array(image.resize((image_w, image_h)))
    mask = np.array(mask.resize((image_w, image_h)))
    #对mask最后一维增加一维
    mask = np.expand_dims(mask, -1)

    #进行数据增广
    if is_aug:
        #1、旋转
        ran1 = random.random()
        if(ran1 < 0.2):
            image, mask = ImageGenerator.random_rotation(image, mask)
        #2、加噪声
        if (ran1 < 0.3 and ran1 > 0.2):
            image = ImageGenerator.random_noise(image)
        #3、锐化
        if (ran1 < 0.7 and ran1 > 0.6):
            image = ImageGenerator.color(image)
        #4、椒盐噪声
        if (ran1 < 0.4 and ran1 > 0.3):
            image = ImageGenerator.arithmetic(image)
        #5、对比度
        if (ran1 < 0.5 and ran1 > 0.4):
            image = ImageGenerator.brightness(image)
        #6、对调
        if (ran1 < 0.6 and ran1 > 0.5):
            image, mask = ImageGenerator.horizontal_flip(image, mask)
    return image, mask

def augment(isAug, path, path_mask, batch, shuffle, funcForData, img_width=640, img_height=512):
    index = 0
    file_names = []
    file_masks = []
    for file in os.listdir(path):   #读取
        file_mask = file.split('.')[0] + '_mask.gif'
        file_names.append(os.path.join(path, file))
        file_masks.append(os.path.join(path_mask, file_mask))

    while True:
        batch_images = []
        batch_masks = []
        for i in range(batch):
            # 读取image和mask
            Myimage = Image.open(file_names[index])
            Mymask = Image.open(file_masks[index])

            Myimage, Mymask = transform(Myimage, Mymask, img_width, img_height, isAug)  # 对数据进行增强


            # 对数据进行预处理
            Myimage = funcForData(Myimage)
            batch_images.append(Myimage)
            batch_masks.append(Mymask)
            index += 1

            if (index == len(file_names)):
                if shuffle:
                    # 重新打乱
                    state = np.random.get_state()
                    np.random.shuffle(file_names)
                    np.random.set_state(state)
                    np.random.shuffle(file_masks)
                index = 0
                break
        batch_images = np.array(batch_images)
        batch_masks = np.array(batch_masks)
        yield batch_images, batch_masks

if __name__ == '__main__':
    from DeepLearning.CV.tools.SegAug.ImageGenerator import plot_im_mask

    path = "../../../../datasets/traindatas/cars/train"  # 样本数据集
    path_mask = "../../../../datasets/traindatas/cars/train_masks"  # 样本mask

    while True:
        datas = augment(True, path, path_mask, 16, True, PreDataForTF)
        X, Y = next(datas)
        X = np.array(X)
        Y = np.array(Y)

        for i in range(len(X)):
            plot_im_mask(X[i], Y[i])