from process_image import *
import vgg_v19

class CONFIG:
    IMAGE_WIDTH = 400
    IMAGE_HEIGHT = 300
    COLOR_CHANNELS = 3
    VGG_MODEL = 'pretrained-model/imagenet-vgg-verydeep-19.mat'
    # Pick the VGG 19-layer model by from the paper "Very Deep Convolutional Networks for Large-Scale Image Recognition".

STYLE_LAYERS = [
    'conv1_1',
    'conv2_1',
    'conv3_1',
    'conv4_1',
    'conv5_1']

def gram(A):
    shape = tf.shape(A)
    num_images = shape[0]
    width = shape[1]
    height = shape[2]
    num_filters = shape[3]
    filters = tf.reshape(A, tf.stack([num_images, -1, num_filters]))
    grams = tf.matmul(filters, filters, transpose_a=True) / tf.to_float(width * height * num_filters)
    return grams

def get_style_feature(path,hight,width):
    style_feature = {};
    image = style_image(path, hight, width)
    vgg_model = vgg_v19.load_vgg_model(CONFIG.VGG_MODEL, CONFIG.IMAGE_HEIGHT,
                                    CONFIG.IMAGE_WIDTH, CONFIG.COLOR_CHANNELS,image);
    with tf.Session() as sess:
        vgg_f = sess.run(vgg_model)
        for layer in STYLE_LAYERS:
            layer_a = vgg_f[layer]
            layer_a = tf.squeeze(gram(layer_a), [0])
            style_feature[layer] = layer_a
    return style_feature

def get_style_loss(style_feature,g_model):
    sum_loss = 0;
    for layer in STYLE_LAYERS:
        layer_a,_ = tf.split(g_model[layer], 2, 0)
        size = tf.size(layer_a)
        loss = tf.nn.l2_loss(tf.subtract(gram(layer_a),style_feature[layer]))*2/tf.to_float(size)
        sum_loss+=loss
    return sum_loss

def get_content_loss(g_model):
    sum_loss = 0;
    for layer in STYLE_LAYERS:
        layer_g,layer_c = tf.split(g_model[layer], 2, 0)
        size = tf.size(layer_g)
        loss = tf.nn.l2_loss(tf.subtract(layer_g,layer_c))*2/tf.to_float(size)
        sum_loss+=loss
    return sum_loss