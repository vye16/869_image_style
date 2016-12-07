import tensorflow as tf
import numpy as np

from scipy.misc import imresize

from network import get_network
'''
Each layer is a filter bank of nfilters filters
of height x width x nchannels
'''

CONTENT_LAYER = 'relu4_2'
STYLE_LAYERS = ['relu1_1', 'relu2_1', 'relu3_1',
                'relu4_1', 'relu5_1'];


def get_content_rep(content_im):
    nrows, ncols, nchannels = content_im.shape
    content_rep = {}
    g = tf.Graph()
    with g.as_default(), tf.Session() as sess:
        image = tf.placeholder('float',
                shape=(1,nrows,ncols,nchannels))
        net, channel_avg = get_network(image)
        # get network responses for our content image
        # output size is 1 x nrows x ncols x nfilters
        content_rep[CONTENT_LAYER] = net[CONTENT_LAYER].eval(
                feed_dict={image:[content_im - channel_avg]})
    return content_rep


def get_style_rep(style_im):
    nrows, ncols, nchannels = style_im.shape
    style_rep = {}
    g = tf.Graph()
    with g.as_default(), tf.Session() as sess:
        image = tf.placeholder('float',
                shape=(1,nrows,ncols,nchannels))
        net, channel_avg = get_network(image)
        # get network responses of style image
        for layer in STYLE_LAYERS:
            # output size is 1 x nrows x ncols x nfilters
            responses = net[layer].eval(
                    feed_dict={image:[style_im - channel_avg]})
            # compute cross correlations between filter responses
            nfilters = responses.shape[-1]
            responses = responses.reshape((-1, nfilters))
            gram = responses.T.dot(responses)
            style_rep[layer] = gram / responses.size
    return style_rep


def content_loss_op(net, content_rep, content_weight):
    # content loss operator
    return content_weight * (tf.nn.l2_loss(
                net[CONTENT_LAYER] - content_rep[CONTENT_LAYER])/
                content_rep[CONTENT_LAYER].size)


def style_loss_op(net, style_rep, style_weight):
    # style loss operator
    loss = 0
    g = tf.Graph()
    with g.as_default(), tf.Session() as sess:
        for layer in STYLE_LAYERS:
            # output of conv_respons is 1 x nrows x ncols x nfilters
            responses = net[layer]
            nfilters = responses.get_shape()[-1].value
            size = np.prod(responses.get_shape()[1:]).value
            resp = tf.reshape(responses, (-1, nfilters))
            gram = tf.matmul(tf.transpose(resp), resp) / size
            loss += tf.nn.l2_loss(gram - style_rep[layer]) / size
    return style_weight * loss


# smoothing/image denoising
def smoothing_loss_op(image, smooth_weight):
    _, ydim, xdim, _ = image.get_shape()
    nrows = ydim.value
    ncols = xdim.value
    ysize = np.prod(image[:,1:,:,:].get_shape()).value
    xsize = np.prod(image[:,:,1:,:].get_shape()).value
    return smooth_weight * (
            tf.nn.l2_loss(image[:,1:,:,:]
                - image[:,:nrows-1,:,:])/ysize
            + tf.nn.l2_loss(image[:,:,1:,:]
                - image[:,:,:ncols-1,:])/xsize)


def synthesize(content, style,
        iterations=1000, learning_rate=1e0,
        content_weight=5, style_weight=1e2, smooth_weight=1e2):

    # XXX just to make faster on my machine
    content_im = imresize(content, [256, 256])
    style_im = imresize(style, [256, 256])
    content_rep = get_content_rep(content_im)
    style_rep = get_style_rep(style_im)

    # optimizing the image
    shape = (1,) + content_im.shape
    image = tf.Variable(tf.random_normal(shape) * 0.2, name='image')
    net, channel_avg = get_network(image)
    total_loss = content_loss_op(net, content_rep, content_weight)\
                    + style_loss_op(net, style_rep, style_weight)\
                    + smoothing_loss_op(image, smooth_weight)

    train_step = tf.train.AdamOptimizer(learning_rate).\
                            minimize(total_loss)

    im_out = None
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for i in xrange(iterations):
            print 'Iteration', i
            train_step.run()
        im_out = image.eval().squeeze() + channel_avg
    return im_out, content_im, style_im

if __name__=="__main__":

    from scipy.misc import imread, imsave

    content = imread('in/content.jpg')
    style = imread('in/style.jpg')

    out, _, _ = synthesize(content, style)
    imsave('out/out.jpg', out)
