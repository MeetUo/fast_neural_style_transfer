import tensorflow as tf
from os import listdir
from os.path import isfile, join
import numpy as np

class CONFIG:
    MEANS = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 3))

def reshape_and_normalize_image(image):
    """
    Reshape and normalize the input image (content or style)
    """
    # Substract the mean to match the expected input of VGG16
    image = tf.subtract(image,CONFIG.MEANS)

    return image

def preprocess_fn(image,height,width):
    image = tf.image.resize_images(image,[height,width],method=0)
    return image

def train_image(batch_size, height, width, path, epochs=2, shuffle=True):
    filenames = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    if not shuffle:
        filenames = sorted(filenames)
    png = filenames[0].lower().endswith('png')  # If first file is a png, assume they all are

    filename_queue = tf.train.string_input_producer(filenames, shuffle=shuffle, num_epochs=epochs)
    reader = tf.WholeFileReader()
    _, img_bytes = reader.read(filename_queue)
    image = tf.image.decode_png(img_bytes, channels=3) if png else tf.image.decode_jpeg(img_bytes, channels=3)
    processed_image = preprocess_fn(image, height, width)
    processed_image = reshape_and_normalize_image(processed_image)
    return tf.train.batch([processed_image], batch_size, dynamic_pad=True)

def style_image(path,height, width):
    img_bytes = tf.read_file(path)
    if path.lower().endswith('png'):
        image = tf.image.decode_png(img_bytes)
    else:
        image = tf.image.decode_jpeg(img_bytes)
    processed_image = preprocess_fn(image, height, width)
    processed_image = reshape_and_normalize_image(processed_image)
    image = tf.expand_dims(processed_image, 0)
    return image