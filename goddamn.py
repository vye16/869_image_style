import tensorflow as tf
import numpy as np

from scipy.misc import imresize

import vgg
#from network import get_network

'''
Each layer is a filter bank of nfilters filters
of height x width x nchannels
'''

CONTENT_LAYER = 'relu4_2'
STYLE_LAYERS = ['relu1_1', 'relu2_1', 'relu3_1',
                'relu4_1', 'relu5_1'];
NETWORK = 'imagenet-vgg-verydeep-19.mat'

def do_shit(content, style,
        iterations=1000, learning_rate=1e0,
        content_weight=5, style_weight=1e2, smooth_weight=1e2):

    content = imresize(content, [256, 256])
    style = imresize(style, [256, 256])

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

    # optimizing the image
    with tf.Graph().as_default():
        image = tf.Variable(tf.random_normal(shape) * 0.256)
        net, channel_avg = vgg.net(NETWORK, image)

        content_loss = content_weight * (2 * tf.nn.l2_loss(
            net[CONTENT_LAYER] - content_features[CONTENT_LAYER]) /
            content_features[CONTENT_LAYER].size)

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
        style_loss = style_weight * reduce(tf.add, style_losses)

        y_size = _tensor_size(image[:,1:,:,:])
        x_size = _tensor_size(image[:,:,1:,:])
        smooth_loss = smooth_weight * 2 * (
                (tf.nn.l2_loss(image[:,1:,:,:] - image[:,:shape[1]-1,:,:]) /
                    y_size) +
                (tf.nn.l2_loss(image[:,:,1:,:] - image[:,:,:shape[2]-1,:]) /
                    x_size))    

        loss = content_loss + style_loss + smooth_loss

        # optimizer setup
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        im_out = None
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for i in xrange(iterations):
                print 'Iteration', i
                train_step.run()
            im = image.eval()
    return vgg.unprocess(im.reshape(shape[1:]), channel_avg)


def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)

if __name__=="__main__":

    from scipy.misc import imread, imsave

    content = imread('in/stata.jpg')
    style = imread('in/style.jpg')

    out = do_shit(content, style)
    imsave('out/goddamn.jpg', out)
