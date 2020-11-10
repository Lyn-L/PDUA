# definition of different custom loss functions
import tensorflow as tf

# Loss as defined in Fast-feature-fool


def activations(network, layers):
    loss = 0
    for i in network.keys():
        if i not in layers:
            try:
                for j in network[i].keys():
                    if j == 'concat':
                        # total blob activations
                        loss += tf.log(tf.reduce_mean(tf.abs(network[i][j])))
            except:
                # total blob activations
                loss += tf.log(tf.reduce_mean(tf.abs(network[i])))
    return loss

# Loss as defined for DG_UAP


def l2_all(network, layers):
    loss = 0
    for i in network.keys():
        if i not in layers:
            try:
                for j in network[i].keys():
                    print(i, j)
                    loss += tf.log(tf.nn.l2_loss(tf.abs(network[i][j])))
            except:
                print i
                loss += tf.log(tf.nn.l2_loss(tf.abs(network[i])))
    return loss

def class_reg(network1, ad_im):
    loss = 0
    
    label1 = network1['prob']
    # label2 = network2['prob']

    g = tf.gradients(label1, ad_im)

    # ind_max = tf.argmax(label1, axis = 1)

    loss = tf.reduce_sum(tf.abs(1 - 0.5 * tf.reduce_sum(tf.multiply(g, ad_im)) * (1+label1)))
    # loss = tf.reduce_mean(1 - tf.multiply(g, ad_im))

    return loss
