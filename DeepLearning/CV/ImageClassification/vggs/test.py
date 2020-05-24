import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from DeepLearning.CV.tools.ClassAug.readDir import PreDataForTF, PreDataForCaffe
import matplotlib.pyplot as plt
from PIL import Image

def predict(model, img, target_size):
    if img.size != target_size:
        img = img.resize(target_size)

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = PreDataForTF(x)    #重新训练用
    # x = PreDataForCaffe(x)    #迁移学习用
    preds = model.predict(x)
    return preds[0]

def plot_preds(image, preds, p):
    index = np.argmax(preds)

    labels = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

    plt.imshow(image)
    plt.axis('off')
    plt.title("come from %s==>" %p+labels[index]+':'+'%s'%preds[index])
    plt.show()

if __name__ == '__main__':
    print("这个是vgg")
    import os
    target_size = (224, 224)

    imagesPath = []
    image_path = '../../../../datasets/testdatas/flowers'
    for root, dirs, files in os.walk(image_path):
        for file in files:
            path = os.path.join(image_path, file)
            imagesPath.append(path)

    # path_model = 'saveWeights/vgg11Model.h5'   #加载自己学习的vgg11模型
    path_model = 'saveWeights/vgg11Model.h5'    #加载自己学习的vgg16迁移模型
    model = load_model(path_model)

    for p in imagesPath:
        img = Image.open(p)
        preds = predict(model, img, target_size)
        #print(preds)
        p = p.split('\\')[-1]
        plot_preds(img, preds, p)