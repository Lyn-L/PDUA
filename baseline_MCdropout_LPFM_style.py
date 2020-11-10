'''
This code generated a universal adversarial network for a given network
'''
# python evaluate.py --network resnet50 --adv_im perturbations/resnet50_md_lpfm_s_with_data.npy --img_list ~/dataset/sub_set.txt --gt_labels ~/dataset/sub_gt_labels.txt

from nets.vgg_f import vggf
from nets.caffenet import caffenet
from nets_new.vgg_16 import vgg16
from nets.vgg_19 import vgg19
from nets.googlenet import googlenet
from nets.resnet_50 import resnet50
from nets.resnet_152 import resnet152
from urllib import urlretrieve
from misc.losses import *
import tensorflow as tf
import numpy as np
import argparse
import time
import math
import os
import misc.losses as losses
import utils.functions as func
from misc.utils import lap_normalize
from AdaBound import AdaBoundOptimizer
from AMSGrad import AMSGrad
# import utils.losses as losses
import warnings
warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES']='0,1'

def validate_arguments(args):
    nets = ['vggf', 'caffenet', 'vgg16', 'vgg19', 'googlenet', 'googlenet',  'resnet50', 'resnet152']
    if not(args.network in nets):
        print ('invalid network')
        exit (-1)

def choose_net(network, train_type):
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

    style_network = 'vggf'
    # placeholder to pass image
    input_image = tf.placeholder(shape=[None, size, size, 3],dtype='float32', name='input_image')
    # # initializing adversarial image
    adv_image = tf.Variable(tf.random_uniform(
        [1,size,size,3],minval=-10,maxval=10), name='noise_image', dtype='float32')
    adv_image = tf.clip_by_value(adv_image,-10,10)

    image_i = './prior/circle.npy'

    # adv_image = tf.Variable(initial_value=np.load(image_i), name='noise_image', dtype='float32')
    # adv_image = tf.clip_by_value(adv_image,-10,10)

    adv_image1 = tf.Variable(initial_value=np.load(image_i), name='noise_image1', dtype='float32', trainable = False)
    adv_image1 = tf.clip_by_value(adv_image1,-10,10)

    input_batch = tf.concat([input_image, tf.add(input_image,adv_image)], 0)
    input_batch1 = tf.concat([input_image, adv_image1], 0)

    test_net = MAP[network](input_batch)
    style_net2 = MAP[style_network](adv_image1)
    with tf.name_scope("train_net"):
        train_net = MAP[network](tf.add(input_image, adv_image))
        # style_net1 = MAP[style_network](adv_image)
        style_net1 = MAP[style_network](tf.add(input_image, adv_image))
        
    return train_net, test_net, style_net1, style_net2, input_image, adv_image

def not_optim_layers(network):
# Layers at which are excluded from optimization
    if network == 'vggf':
        return ['norm1', 'pool1', 'norm2', 'pool2', 'pool5', 'fc6', 'fc7', 'fc8', 'prob']
    if network == 'caffenet':
        return ['norm1', 'pool1', 'norm2', 'pool2', 'pool5', 'fc6', 'fc7', 'fc8', 'prob']
    elif network == 'vgg16':
        return ['pool1', 'pool2', 'pool3', 'pool4', 'pool5', 'fc6', 'fc7', 'fc8', 'prob']
    elif network == 'vgg19':
        return ['pool1', 'pool2', 'pool3', 'pool4', 'pool5', 'prob']
    elif network == 'googlenet':
        return ['pool1_3x3_s2', 'pool1_norm1', 'conv2_norm2', 'pool2_3x3_s2', 'pool3_3x3_s2', 'pool4_3x3_s2', 'pool5_7x7_s1', 'loss3_classifier', 'prob']
    elif network == 'resnet50':
        return ['bn_conv1', 'pool1', 'pool5', 'pool5_r', 'fc1000', 'prob']
    elif network == 'resnet152':
        return ['bn_conv1', 'pool1', 'pool5', 'pool5_r', 'fc1000', 'prob']
    else:
        return ['pool1_3x3_s2', 'pool1_norm1', 'conv2_norm2', 'pool2_3x3_s2', 'pool3_3x3_s2', 'pool4_3x3_s2', 'pool5_7x7_s1', 'loss3_classifier', 'prob']

def rescale_checker_function(check, sat, sat_change, sat_min):
    value = (sat_change < check and sat > sat_min)
    return value


def get_update_operation_func(train_type, in_im, sess, update, batch_size, size, img_list):
    if train_type == 'no_data':
        def updater(noiser, sess=sess, update=update):
            sess.run(update, feed_dict={in_im: noiser})
    elif train_type == 'with_range':
        def updater(noiser, sess=sess, update=update, in_im=in_im, batch_size=batch_size, size=size):
            image_i = 'data/gaussian_noise.png'
            for j in range(batch_size):
                noiser[j:j+1] = np.copy(func.img_preprocess(image_i,
                                                            size=size, augment=True))
            sess.run(update, feed_dict={in_im: noiser})
    elif train_type == 'with_data':
        def updater(noiser, sess=sess, update=update, in_im=in_im, batch_size=batch_size, size=size, img_list=img_list):
            rander = np.random.randint(low=0, high=(len(img_list)-batch_size))
            for j in range(batch_size):
                noiser[j:j+1] = np.copy(func.img_preprocess(
                    img_list[rander+j].strip(), size=size, augment=False))
            sess.run(update, feed_dict={in_im: noiser})
    return updater

# def train(adv_net, d_net, net, in_im, ad_im, d_im, opt_layers, net_name):
def train(adv_net, net, s_net1, s_net2, in_im, ad_im, opt_layers, net_name, train_type, batch_size, rho, K_mc, img_list_file=None):

    # Vanilla Version
    # K_mc = 10
    cost = 0
    for i in range(K_mc):
        cost += - losses.drop_loss_dropout(adv_net, opt_layers, 0.1)
    s_cost = losses.style_loss_relative(s_net1, s_net2, 'conv3')
    f_cost = cost / K_mc + rho * s_cost

    tvars = tf.trainable_variables()
    # optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
    optimizer = AdaBoundOptimizer(learning_rate=0.05, final_lr=0.1, beta1=0.85, beta2=0.999, amsbound=False)
    # optimizer = AMSGrad(learning_rate=0.15, beta1=0.9, beta2=0.99, epsilon=1e-8)
    # optimizer = tf.train.MomentumOptimizer(learning_rate=0.1, momentum=0.9)
    grads = optimizer.compute_gradients(f_cost, tvars)
    # capped_gvs = [(lap_normalize(grad), var) for grad, var in grads]
    update = optimizer.apply_gradients(grads)


    size = 224
    # getting the validation set
    if net_name == 'caffenet':
        data_path = os.path.join('data', 'preprocessed_images_227.npy')
        download_link = "https://www.dropbox.com/s/0v8pnnumbytb378/preprocessed_images_227.npy?raw=1"
    else:
        # data_path = os.path.join('data', 'preprocessed_images_224.npy')
        data_path = os.path.join('data', 'vgg_preprocessed.npy')
        download_link = "https://www.dropbox.com/s/k4tamvdjndyvgws/preprocessed_images_224.npy?raw=1"
    if os.path.isfile(data_path) == 0:
        print('Downloading validation data (1K images)...')
        urlretrieve (download_link, data_path)

    imgs = np.load(data_path)
    print('Loaded mini Validation Set')

    # constants
    fool_rate = 0  # current fooling rate
    max_iter = 20000  # better safe than looped into eternity
    stopping = 0  # early stopping condition
    t_s = time.time()
    # New constants
    check = 0.00001
    prev_check = 0
    rescaled = False
    stop_check = False
    noiser = np.zeros((batch_size, size, size, 3))
    rescaled = False
    if train_type == 'with_data':
        img_list = open(img_list_file).readlines()
    else:
        img_list = None

    print "Starting {:} training...".format(net_name)

    # Saturation Measure
    saturation = tf.div(tf.reduce_sum(tf.to_float(
        tf.equal(tf.abs(ad_im), 10))), tf.to_float(tf.size(ad_im)))
    # rate of change of percentage change
    sat_prev = 0
    sat = 0
    sat_change = 0
    sat_min = 0.5  # For checking sat difference only after its sensible

    # rescaler
    assign_op = tvars[0].assign(tf.divide(tvars[0], 2.0))

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    with tf.Session(config=config) as sess:
        update_op = get_update_operation_func(
            train_type, in_im, sess, update, batch_size, size, img_list)
        sess.run(tf.global_variables_initializer())
        for i in range(max_iter):
            update_op(noiser)
            # calculate variables
            sat_prev = np.copy(sat)
            sat = sess.run(saturation)
            sat_change = abs(sat-sat_prev)
            check_dif = i - prev_check
            if i % 100 == 0:
                print('iter', i, 'current_saturation',
                      sat, 'sat_change', sat_change)

            # check for saturation
            if rescale_checker_function(check, sat, sat_change, sat_min):
                rescaled = True
            # validation time
            if not stop_check and ((check_dif > 200 and rescaled == True) or check_dif == 400):
                iters = int(math.ceil(1000/float(batch_size)))
                temp = 0
                prev_check = i
                for j in range(iters):
                    l = j*batch_size
                    L = min((j+1)*batch_size, 1000)
                    softmax_scores = sess.run(
                        net['prob'], feed_dict={in_im: imgs[l:L]})
                    true_predictions = np.argmax(
                        softmax_scores[:batch_size], axis=1)
                    ad_predictions = np.argmax(
                        softmax_scores[batch_size:], axis=1)
                    not_flip = np.sum(true_predictions == ad_predictions)
                    temp += not_flip
                current_rate = (1000-temp)/1000.0
                print('current_val_fooling_rate',
                      current_rate, 'current_iter', i)

                if current_rate > fool_rate:
                    print('best_performance_till_now')
                    stopping = 0
                    fool_rate = current_rate
                    im = sess.run(ad_im)
                    name = 'perturbations/'+net_name+'_md_lpfm_s_'+train_type+'.npy'
                    np.save(name, im)
                else:
                    stopping += 1
                if stopping == 10:
                    print('Val best out')
                    stop_check = True
                    break

            if rescale_checker_function(check, sat, sat_change, sat_min):
                sess.run(assign_op)
                print('reached_saturation', sat, sat_change,
                      'criteria', check, 'iter', i)
                rescaled = False
                prev_check = i
        print('training_done', time.time()-t_s)

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
    parser.add_argument('--rho', default=1e-3,
                        help='The batch size to use for training and testing')
    parser.add_argument('--K_mc', default=10,
                        help='The batch size to use for training and testing')

    args = parser.parse_args()
    if args.img_list == 'None':
        args.img_list = None
    validate_arguments(args)
    adv_net, net, s_net1, s_net2, inp_im, ad_im = choose_net(args.network, args.prior_type)
    opt_layers = not_optim_layers(args.network)
    train(adv_net, net, s_net1, s_net2, inp_im, ad_im, opt_layers, args.network,
          args.prior_type, int(args.batch_size), float(args.rho), int(args.K_mc),args.img_list)

if __name__ == '__main__':
    main()
