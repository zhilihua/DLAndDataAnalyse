import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
import skimage as sk
from imgaug import augmenters as iaa

def random_rotation(image, mask):
    random_degree = random.uniform(-50, 50)
    return sk.transform.rotate(image, random_degree, preserve_range=True).astype(np.uint8), \
           sk.transform.rotate(mask, random_degree, preserve_range=True).astype(np.uint8)

def random_noise(image_array):
    seq = iaa.Sequential([
        iaa.GaussianBlur(sigma=(1.0, 3.0))
    ])
    return seq.augment_image(image_array)

def color(image_array):
    seq = iaa.Sequential([
        iaa.Sharpen(lightness=1.00, alpha=1)
    ])
    return seq.augment_image(image_array)


def arithmetic(image_array):
    seq = iaa.Sequential([
        iaa.Salt(p=0.03)
    ])
    return seq.augment_image(image_array)


def brightness(image):
    i = random.randint(0, 3)
    items = [0.25, 1.00, 0.5, 1.5]
    seq = iaa.Sequential([
        iaa.Multiply(mul=items[i]),
        iaa.Sharpen(lightness=1.00, alpha=1)
    ])
    return seq.augment_image(image)


def horizontal_flip(image_array, mask_array):
    return image_array[:, ::-1], mask_array[:, ::-1]


def plot_im_mask(im, im_mask):
    im_mask = im_mask > 0.5
    # im = (im+1)*127.5
    im = np.array(im, dtype=np.uint8)
    im_mask = np.array(im_mask, dtype=np.uint8)
    plt.subplot(1, 3, 1)
    plt.imshow(im)
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(im_mask[:, :, 0])
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.bitwise_and(im, im, mask=im_mask))
    plt.axis('off')
    plt.show()