# Copyright (c) 2015-2016 Anish Athalye. Released under GPLv3.

import numpy as np
import scipy.misc


def transfer_color(content, styles):
    shape = (1,) + content.shape
    style_shapes = [(1,) + style.shape for style in styles]
    
    content_mean = np.mean(content, axis=(0,1))
    diff = (content - content_mean).reshape((-1, 3))
    content_cov = np.cov(diff.T)
    cvals, cvecs = np.linalg.eig(content_cov)
    content_sqrt_cov = cvecs.dot(np.diag(np.sqrt(cvals))).dot(cvecs.T) 

    new_styles = []
    for i in range(len(styles)):
        style = styles[i]
        style_shape = style_shapes[i]
        style_mean = np.mean(style, axis=(0,1))
        diff = (style - style_mean).reshape((-1, 3))
        style_cov = np.cov(diff.T)
        svals, svecs = np.linalg.eig(style_cov)
        style_sqrt_cov = svecs.dot(np.diag(np.sqrt(svals))).dot(svecs.T)
        style_inv = np.linalg.inv(style_sqrt_cov)
        A = content_sqrt_cov.dot(style_inv)
        b = content_mean - A.dot(style_mean)
        new_style = np.zeros(style.shape)
        for r in range(style.shape[0]):
            for c in range(style.shape[1]):
                new_style[r,c,:] = A.dot(style[r,c,:]) + b
        new_styles.append(new_style)
    return new_styles


def rgb2yiq(im):
    newim = np.zeros(im.shape)
    nrows, ncols, _ = im.shape
    A = np.array(
            [[0.299, 0.587, 0.114],
             [0.596, -0.274, -0.322],
             [0.211, -0.523, 0.312]])
    for i in range(nrows):
        for j in range(ncols):
            newim[i,j,:] = A.dot(im[i,j,:])
    return newim

def yiq2rgb(im):
    newim = np.zeros(im.shape)
    nrows, ncols, _ = im.shape
    A = np.array(
            [[1, 0.956, 0.621],
             [1, -0.272, -0.647],
             [1, -1.106, 1.703]])
    for i in range(nrows):
        for j in range(ncols):
            newim[i,j,:] = A.dot(im[i,j,:])
    return newim

def imread(path):
    return scipy.misc.imread(path).astype(np.float)

def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(path, img)

if __name__=="__main__":

    content = imread('in/stata.jpg')
    styles = [imread('in/style.jpg')]
    new_styles = transfer_color(content, styles)
    imsave('in/new_style.jpg', new_styles[0])
