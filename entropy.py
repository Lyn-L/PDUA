'''
This code generated a universal adversarial network for a given network
'''
# python evaluate.py --network vgg16 --adv_im perturbations/vgg16_o_no_data.npy --img_list ~/dataset/image_name.txt --gt_labels ~/dataset/gt_labels.txt


from nets.vgg_f import vggf
from nets.caffenet import caffenet
from nets_new.vgg_16 import vgg16
from nets.vgg_19 import vgg19
from nets.googlenet import googlenet
from nets.resnet_50 import resnet50
from nets.resnet_152 import resnet152
from misc.utils import *
import tensorflow as tf
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import argparse

import time

import os

os.environ['CUDA_VISIBLE_DEVICES']='1'

def validate_arguments(args):
    nets = ['vggf', 'caffenet', 'vgg16', 'vgg19', 'googlenet', 'googlenet',  'resnet50', 'resnet152']
    if not(args.network in nets):
        print ('invalid network')
        exit (-1)

def choose_net(network):
    MAP = {
        'vggf'     : vggf,
        'caffenet' : caffenet,
        'vgg16'    : vgg16,
        'vgg19'    : vgg19,
        'googlenet': googlenet,
        'resnet50': resnet50,
        'resnet152': resnet152
    }
    if network == 'caffenet':
        size = 227
    else:
        size = 224
    # placeholder to pass image
    input_image = tf.placeholder(shape=[None, size, size, 3],dtype='float32', name='input_image')

    # input_batch = tf.concat([input_image, tf.add(input_image,adv_image)], 0)


    return  MAP[network](input_image), input_image

def entropy(x):
    n= x.shape[1]
    ret=0.0
    for i in range(n):
        ret=ret+(-1*x[0, i])*np.log2(x[0, i] + 1e-7)
    return ret

# def train(adv_net, d_net, net, in_im, ad_im, d_im, opt_layers, net_name):
def train(net, in_im, net_name, im_list,):


    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    imgs = open(im_list).readlines()
    isotropic, size = get_params(net_name)
    batch_size = 1
    batch_im = np.zeros((batch_size, size, size, 3))
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        img_loader = loader_func(net_name, sess, isotropic, size)
        for i in range(len(imgs)):
            # im = img_loader(imgs[i].strip())
            batch_im = np.load(imgs[i].strip())
            softmax_scores = sess.run(net['prob'], feed_dict={in_im: batch_im})
            a = np.array(softmax_scores)
            print imgs[i].strip()
            print entropy(a)
            # if i == 0:
            #     print imgs[i].strip()
            #     print entropy(a)
            # elif i == 1:
            #     print imgs[i].strip()
            #     print entropy(a)
            # elif i == 2:
            #     print imgs[i].strip()
            #     print entropy(a)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', default='googlenet',
                        help='The network eg. googlenet')
    parser.add_argument('--prior_type', default='no_data',
                        help='Which kind of prior to use')
    parser.add_argument('--img_list', default='None',
                        help='In case of providing data priors,list of image-files')
    parser.add_argument('--batch_size', default=25,
                        help='The batch size to use for training and testing')
    args = parser.parse_args()
    if args.img_list == 'None':
        args.img_list = None
    validate_arguments(args)
    # adv_net, net, inp_im, ad_im = choose_net(args.network)
    net, inp_im  = choose_net(args.network)
    # opt_layers = not_optim_layers(args.network)
    train(net, inp_im ,  args.network, args.img_list)

if __name__ == '__main__':
    main()
