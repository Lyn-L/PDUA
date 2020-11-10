#definition of different custom loss functions
import tensorflow as tf
import tensorflow_probability as tfp
# from keras import backend as K
# from keras.layers import Input, Dense, Lambda, Activation, Flatten, Convolution2D, MaxPooling2D
# from keras.models import Model
# from keras.regularizers import l2
# from keras.utils import np_utils
import numpy as np
import ConcreteDropout as CD 
layers = tf.contrib.layers
tfd = tfp.distributions

#to maximise activations
def activations(network, layers):
    loss = 0
    for i in network.keys():
        if i not in layers:
            try:
                for j in network[i].keys():
                    if j == 'concat':
                        loss += tf.log(tf.reduce_mean(tf.abs(network[i][j]))) #total blob activations
            except:
                loss += tf.log(tf.reduce_mean(tf.abs(network[i]))) #total blob activations
    return loss
    
# Loss as defined for DG_UAP
# def Dropout_mc(p):
#     layer = Lambda(lambda x: K.dropout(x, p), output_shape=lambda shape: shape)
#     return layer

def apply_layers(inp, layers):
    output = inp
    for layer in layers:
        output = layer(output)
    return output

def GenerateMCSamples(inp, layers, K_mc=20):
    if K_mc == 1:
        return apply_layers(inp, layers)
    output_list = []
    for _ in xrange(K_mc):
        output_list += [apply_layers(inp, layers)]  # THIS IS BAD!!! we create new dense layers at every call!!!!
    def pack_out(output_list):
        #output = K.pack(output_list) # K_mc x nb_batch x nb_classes
        output = K.stack(output_list) # K_mc x nb_batch x nb_classes
        return K.permute_dimensions(output, (1, 0, 2)) # nb_batch x K_mc x nb_classes
    def pack_shape(s):
        s = s[0]
        assert len(s) == 2
        return (s[0], K_mc, s[1])
    out = Lambda(pack_out, output_shape=pack_shape)(output_list)
    return out

def l2_all(network, layers):
    loss = 0
    for i in network.keys():
        if i not in layers:
            try:
                for j in network[i].keys():
                    print(i, j)
                    loss +=  tf.log(tf.nn.l2_loss(tf.abs(network[i][j])))
            except:
                print i
                loss += tf.log(tf.nn.l2_loss(tf.abs(network[i])))
    return loss

def drop_loss_dropout(network, layers, prob):
    loss = 0
    for i in network.keys():
        if i not in layers:
            try:
                for j in network[i].keys():
                    if j == 'concat':
                        tmp = tf.abs(tf.nn.dropout(network[i][j], keep_prob= prob))
                        loss += tf.log(tf.nn.l2_loss(tf.nn.l2_loss(tmp)))
            except:
                tmp = tf.abs(tf.nn.dropout(network[i], keep_prob= prob))
                loss += tf.log(tf.nn.l2_loss(tf.nn.l2_loss(tmp)))
    return loss

def drop_loss_concrete_dropout(network, layers):
    loss = 0
    for i in network.keys():
        if i not in layers:
            try:
                for j in network[i].keys():
                    if j == 'concat':
                        tmp = tf.abs(CD.ConcreteDropout(network[i][j]))
                        loss += tf.log(tf.nn.l2_loss(tf.nn.l2_loss(tmp)))
            except:
                tmp = tf.abs(CD.ConcreteDropout(network[i]))
                loss += tf.log(tf.nn.l2_loss(tf.nn.l2_loss(tmp)))
    return loss

def make_likelihood(event_prob):
    return tfd.Bernoulli(probs=event_prob,dtype=tf.float32)

def drop_loss_mcmc(network, layers, prob):
    loss = 0
    dtype = np.float32
    likelihood = make_likelihood(prob)
    for i in network.keys():
        if i not in layers:
            try:
                for j in network[i].keys():
                    if j == 'concat':
                        tmp = tf.reduce_mean(tf.abs(network[i][j]), axis = 0)
                        states, _ = tfp.mcmc.sample_chain(num_results=2,current_state=tmp,kernel = tfp.mcmc.HamiltonianMonteCarlo(target_log_prob_fn=likelihood.log_prob,step_size=0.1, num_leapfrog_steps=2))#, num_burnin_steps=200,  num_steps_between_results=1,           parallel_iterations=1)
                        sample_mean = tf.nn.l2_loss(tf.abs(states))
                        loss += tf.log(sample_mean)
            except:
                tmp = tf.reduce_mean(tf.abs(network[i]), axis = 0)
                states, _ = tfp.mcmc.sample_chain(num_results=2,current_state=tmp,kernel = tfp.mcmc.HamiltonianMonteCarlo(target_log_prob_fn=likelihood.log_prob,step_size=0.1, num_leapfrog_steps=2))#, num_burnin_steps=200,  num_steps_between_results=1,           parallel_iterations=1)
                sample_mean = tf.nn.l2_loss(tf.abs(states))
                loss += tf.log(sample_mean)
    return loss

def dis_loss(ad_im, prior_im):
    return tf.reduce_sum(tf.nn.l2_loss(ad_im - prior_im))

def gram_matrix(tensor):
    shape = tensor.get_shape()

    # Get the number of feature channels for the input tensor,
    # which is assumed to be from a convolutional layer with 4-dim.
    num_channels = int(shape[3])

    # Reshape the tensor so it is a 2-dim matrix. This essentially
    # flattens the contents of each feature-channel.
    matrix = tf.reshape(tensor, shape=[-1, num_channels])

    # Calculate the Gram-matrix as the matrix-product of
    # the 2-dim matrix with itself. This calculates the
    # dot-products of all combinations of the feature-channels.
    gram = tf.matmul(tf.transpose(matrix), matrix)

    return gram


def style_loss_relative(A, X, layer):

    a = A[layer]
    x = X[layer]

    A = gram_matrix(a)
    G = gram_matrix(x)
    loss = tf.div(tf.reduce_sum(tf.pow((G-A),2)), tf.reduce_sum(tf.pow(G,2)))
    return loss

def style_loss(A, X, layer):
    a = A[layer]
    x = X[layer]

    A = gram_matrix(a)
    G = gram_matrix(x)

    N = a.shape[1].value # M
    M = a.shape[2].value * a.shape[3].value # N
    number = 1. / (4 * N ** 2 * M ** 2)
    loss = tf.reduce_sum(tf.pow((G-A),2)) * number 
    return loss

def style_dis_loss(A, X, layer):
    a = A[layer]
    x = X[layer]
    return tf.reduce_sum(tf.nn.l2_loss(a - x))