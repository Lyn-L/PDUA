import numpy as np
from skimage.io import imread
from skimage.transform import resize
import tensorflow as tf

def downsample(inp):
    return np.reshape(inp[1:-2,1:-2,:], [1,224,224,3])

def upsample(inp):
    out = np.zeros([227,227,3])
    out[1:-2,1:-2,:] = inp
    out[0,1:-2,:] = inp[0,:,:]
    out[-2,1:-2,:] = inp[-1,:,:]
    out[-1,1:-2,:] = inp[-1,:,:]
    out[:,0,:] = out[:,1,:]
    out[:,-2,:] = out[:,-3,:]
    out[:,-1,:] = out[:,-3,:]
    return np.reshape(out,[1,227,227,3])

def old_img_preprocess(img_path, size=224):
    mean = [103.939, 116.779, 123.68]
    img = imread(img_path)
    # img = np.asarray(img_path)
    img = resize(img, (size, size))*255.0
    if len(img.shape) == 2:
        img = np.dstack([img, img, img])
    img[:, :, 0] -= mean[2]
    img[:, :, 1] -= mean[1]
    img[:, :, 2] -= mean[0]
    img[:, :, [0, 1, 2]] = img[:, :, [2, 1, 0]]
    img = np.reshape(img, [1, size, size, 3])
    return img
# Preprocessing for Inception V3


def v3_preprocess(img_path):
    img = imread(img_path)
    # img = np.asarray(img_path)
    img = resize(img, (299, 299), preserve_range=True)
    img = (img - 128) / 128
    if len(img.shape) == 2:
        img = np.dstack([img, img, img])
    img = np.reshape(img, [1, 299, 299, 3])
    return img

# Image preprocessing format
# Fog VGG models.


def vgg_preprocess(img_path, size=224):
    mean = [103.939, 116.779, 123.68]
    img = imread(img_path)
    # img = np.asarray(img_path)
    if len(img.shape) == 2:
        img = np.dstack([img, img, img])
    resFac = 256.0/min(img.shape[:2])
    newSize = list(map(int, (img.shape[0]*resFac, img.shape[1]*resFac)))
    img = resize(img, newSize, mode='constant', preserve_range=True)
    offset = [newSize[0]/2.0 -
              np.floor(size/2.0), newSize[1]/2.0-np.floor(size/2.0)]
    # print(offset,size)
    img = img[int(offset[0]):int(offset[0])+size,
              int(offset[1]):int(offset[1])+size, :]
    img[:, :, 0] -= mean[2]
    img[:, :, 1] -= mean[1]
    img[:, :, 2] -= mean[0]
    img[:, :, [0, 1, 2]] = img[:, :, [2, 1, 0]]
    img = np.reshape(img, [1, size, size, 3])
    return img

# For Resnets,Caffenet and Googlenet
# From Caffe-tensorflow


def img_preprocess(img, scale=256, isotropic=False, crop=227, mean=np.array([103.939, 116.779, 123.68])):
    '''Crops, scales, and normalizes the given image.
    scale : The image wil be first scaled to this size.
            If isotropic is true, the smaller side is rescaled to this,
            preserving the aspect ratio.
    crop  : After scaling, a central crop of this size is taken.
    mean  : Subtracted from the image
    '''
    # Rescale
    if isotropic:
        img_shape = tf.to_float(tf.shape(img)[:2])
        min_length = tf.minimum(img_shape[0], img_shape[1])
        new_shape = tf.to_int32((scale / min_length) * img_shape)
    else:
        new_shape = tf.stack([scale, scale])
    img = tf.image.resize_images(img, new_shape)
    # Center crop
    # Use the slice workaround until crop_to_bounding_box supports deferred tensor shapes
    # See: https://github.com/tensorflow/tensorflow/issues/521
    offset = (new_shape - crop) / 2
    img = tf.slice(img, begin=tf.stack(
        [offset[0], offset[1], 0]), size=tf.stack([crop, crop, -1]))
    # Mean subtraction
    return tf.to_float(img) - mean


def load_image():
    # Read the file
    image_path = tf.placeholder(tf.string, None)
    file_data = tf.read_file(image_path)
    # Decode the image data
    img = tf.image.decode_jpeg(file_data, channels=3)
    img = tf.reverse(img, [-1])
    return img, image_path


def loader_func(network_name, sess, isotropic, size):
    if network_name == 'inceptionv3':
        def loader(image_name):
            im = v3_preprocess(image_name)
            return im
    elif 'vgg' in network_name:
        def loader(image_name):
            im = vgg_preprocess(image_name)
            return im
    else:
        img_tensor, image_path_tensor = load_image()
        processed_img = img_preprocess(
            img=img_tensor, isotropic=isotropic, crop=size)

        def loader(image_name, processed_img=processed_img, image_path_tensor=image_path_tensor, sess=sess):
            im = sess.run([processed_img], feed_dict={
                          image_path_tensor: image_name})
            return im
    return loader


def get_params(net_name):
    isotropic = False
    if net_name == 'caffenet':
        size = 227
    elif net_name == 'inceptionv3':
        size = 299
    else:
        size = 224
        if not net_name == 'googlenet':
            isotropic = True
    return isotropic, size

k = np.float32([1, 4, 6, 4, 1])
# k = np.float32([5, 1, 0.5, 1, 5])
# k = np.float32([5, 3, 1, 3, 5])
# k = np.float32([1, 4, 0, 4, 1])
k = np.outer(k, k)
k5x5 = k[:, :, None, None] / k.sum() * np.eye(3, dtype=np.float32)

def lap_split(img):
    with tf.name_scope('split'):
        lo = tf.nn.conv2d(img, k5x5, [1, 2, 2, 1], 'SAME')
        lo2 = tf.nn.conv2d_transpose(lo, k5x5 * 4, tf.shape(img), [1, 2, 2, 1])
        hi = img - lo2
    return lo, hi

def lap_split_n(img, n):
    levels = []
    for i in range(n):
        img, hi = lap_split(img)
        levels.append(hi)
    levels.append(img)
    return levels[::-1]

def lap_merge(levels):
    img = levels[0]
    for hi in levels[1:]:
        with tf.name_scope('merge'):
            img = tf.nn.conv2d_transpose(img, k5x5 * 4, tf.shape(hi), [1, 2, 2, 1]) + hi
    return img


def normalize_std(img, eps=1e-10):
    with tf.name_scope('normalize'):
        std = tf.sqrt(tf.reduce_mean(tf.square(img)))
        return img / tf.maximum(std, eps)

def lap_normalize(img, scale_n=2):
    # img = tf.expand_dims(img, 0)
    # img = img - tf.reduce_mean(img)
    tlevels = lap_split_n(img, scale_n)
    tlevels = list(map(normalize_std, tlevels))
    out = lap_merge(tlevels)
    return out