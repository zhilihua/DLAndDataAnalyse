from model import UNet16
from DeepLearning.CV.tools.SegAug import ImageGenerator
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.layers import Input
from data_generater import PreDataForCaffe

height = 256
width = 320
path = "model_seg.h5"    #预训练数据的参数

#获取模型
input_img = Input((height, width, 3), name='img')
model = UNet16(input_img)
#加载参数
model.load_weights(path, by_name=True)

def predict(model, path_img):
    image = cv2.imread(path_img)

    inputs = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  #转换为输入格式
    #缩放到指定大小
    o_input = np.array(inputs.resize((width, height)))
    inputs = PreDataForCaffe(np.array(inputs.resize((width, height))))
    #维度增加
    inputs = np.expand_dims(inputs, 0)

    #预测
    mask = model.predict(inputs)

    #画图
    ImageGenerator.plot_im_mask(o_input, mask[0])

if __name__ == '__main__':
    path_imgs = "../../../../datasets/testdatas/cars/"
    import os
    images_path = os.listdir(path_imgs)
    for file_path in images_path:
        file = os.path.join(path_imgs, file_path)
        predict(model, file)