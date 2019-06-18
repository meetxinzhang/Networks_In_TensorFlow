import numpy as np
from PIL import Image
import tensorflow


def get_img_numpy(imagePath, h, w):

    features = np.asarray(Image.open(imagePath))
    features = tensorflow.image.per_image_standardization(features)
    features = tensorflow.image.resize_image_with_crop_or_pad(features, target_height=h, target_width=w)

    return features.numpy()