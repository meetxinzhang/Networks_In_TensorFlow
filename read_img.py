import numpy as np
from PIL import Image
import tensorflow


def get_img_numpy(imagePath, h, w):

    features = np.asarray(Image.open(imagePath))

    # if imagePath == 'pictures/smoke/119 (2).jpg':
    #     print('yuanlai', np.shape(features))
    #     print(features)

    shape = np.shape(features)
    if len(shape) < 3:
        features = add(features, 0)
    elif shape[-1] == 1:
        features = add(features, 1)
    elif shape[-1] == 2:
        features = add(features, 2)
    elif shape[-1] > 3:
        features = add(features, 4)

    # features = tensorflow.image.per_image_standardization(features)
    features = tensorflow.image.resize_image_with_crop_or_pad(features, target_height=h, target_width=w)

    # print(np.shape(features), imagePath)
    return features.numpy()


def add(x, n=3):
    print('添加颜色通道: ', n)
    # features = np.expand_dims(features, axis=0)
    if n == 0:
        x = [x, x, x]
    elif n == 1:
        x = np.squeeze(x)
        x = [x, x, x]
    elif n == 2:
        x = [x[:, :, 0], x[:, :, 1], x[:, :, 1]]
    elif n == 4:
        x = [x[:, :, 0], x[:, :, 1], x[:, :, 2]]
    print(np.shape(x), '00000000000000')
    x = np.transpose(x, (1, 2, 0))
    print(np.shape(x), '111111111111111')
    return x
