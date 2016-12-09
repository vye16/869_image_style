import tensorflow as tf
import numpy as np

from scipy.misc import imresize

import vgg

'''
Each layer is a filter bank of nfilters filters
of height x width x nchannels
'''

CONTENT_LAYER = 'relu4_2'
STYLE_LAYERS = ['relu1_1', 'relu2_1', 'relu3_1',
                'relu4_1', 'relu5_1'];
NETWORK = 'imagenet-vgg-verydeep-19.mat'


def get_content_rep(content):
    shape = (1,) + content.shape
    content_features = {}

    # compute content features in feedforward mode
    g = tf.Graph()
    with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
        image = tf.placeholder('float', shape=shape)
        net, mean_pixel = vgg.net(NETWORK, image)
        content_pre = np.array([vgg.preprocess(content, mean_pixel)])
        content_features[CONTENT_LAYER] = net[CONTENT_LAYER].eval(
                feed_dict={image: content_pre})
    return content_features

def get_style_rep(style):
    style_features = {}
    style_shape = (1,) + style.shape
    # compute style features in feedforward mode
    g = tf.Graph()
    with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
        image = tf.placeholder('float', shape=style_shape)
        net, mean_pixel = vgg.net(NETWORK, image)
        style_pre = np.array([vgg.preprocess(style, mean_pixel)])
        for layer in STYLE_LAYERS:
            features = net[layer].eval(feed_dict={image: style_pre})
            features = np.reshape(features, (-1, features.shape[3]))
            gram = np.matmul(features.T, features) / features.size
            style_features[layer] = gram
    return style_features

def content_loss_op(net, content_features, content_weight):
    return content_weight * (2 * tf.nn.l2_loss(
            net[CONTENT_LAYER] - content_features[CONTENT_LAYER]) /
            content_features[CONTENT_LAYER].size)

def style_loss_op(net, style_features, style_weight):
    style_loss = 0
    style_losses = []
    for style_layer in STYLE_LAYERS:
        layer = net[style_layer]
        _, height, width, number = map(lambda i: i.value, layer.get_shape())
        size = height * width * number
        feats = tf.reshape(layer, (-1, number))
        gram = tf.matmul(tf.transpose(feats), feats) / size
        style_gram = style_features[style_layer]
        style_losses.append(2 * tf.nn.l2_loss(gram - style_gram) / style_gram.size)
    return style_weight * reduce(tf.add, style_losses)

def smoothing_loss_op(image, smooth_weight):
    _, ydim, xdim, _ = image.get_shape()
    nrows = ydim.value
    ncols = xdim.value
    y_size = _tensor_size(image[:,1:,:,:])
    x_size = _tensor_size(image[:,:,1:,:])
    return smooth_weight * 2 * (
            (tf.nn.l2_loss(image[:,1:,:,:] - image[:,:nrows-1,:,:]) /
                y_size) +
            (tf.nn.l2_loss(image[:,:,1:,:] - image[:,:,:ncols-1,:]) /
                x_size))

def do_shit(content, style,
        iterations=1000, learning_rate=1e0,
        content_weight=1, style_weight=1e2, smooth_weight=1e2):

    content_im = imresize(content, [256, 256])
    style_im = imresize(style, [256, 256])
    content_rep = get_content_rep(content_im)
    style_rep = get_style_rep(style_im)

    # optimizing the image
    shape = (1,) + content_im.shape
    image = tf.Variable(tf.random_normal(shape) * 0.2, name='image')
    net, channel_avg = vgg.net(NETWORK, image)
 
    content_loss = content_loss_op(net, content_rep, content_weight)
    style_loss = style_loss_op(net, style_rep, style_weight)
    tv_loss = smoothing_loss_op(image, smooth_weight)
    loss = content_loss + style_loss + tv_loss

    # optimizer setup
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    im_out = None
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for i in xrange(iterations):
            print 'Iteration', i
            train_step.run()
        im_out = image.eval().squeeze() + channel_avg
    return im_out


def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)

if __name__=="__main__":

    from scipy.misc import imread, imsave

    content = imread('in/stata.jpg')
    style = imread('in/style.jpg')

    out = do_shit(content, style)
    imsave('out/goddamn.jpg', out)
